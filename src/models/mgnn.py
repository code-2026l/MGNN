"""
MGNN: Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection.

Implements the three-view fusion architecture (paper, Sec. 4.3-4.4):

  - Sequence view:    BiLSTM over the 3-channel packet sequence
                      (length, direction, inter-arrival time), max-pooled
                      over time, sequences truncated at 100 packets.
  - Statistical view: two-layer MLP with batch normalization over the
                      23-dimensional distributional flow features.
  - Interaction view: 2-layer, 4-head GAT with type-specific linear
                      projections over the heterogeneous flow/IP graph.

Cross-view attention (a linear layer followed by tanh, then softmax) learns
per-flow fusion weights, enabling post-hoc interpretability (paper, Sec. 4.4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class SeqEncoder(nn.Module):
    """BiLSTM encoder for the packet-sequence view (paper, Sec. 4.3)."""

    def __init__(self, input_dim=3, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden // 2, num_layers=2,
                            batch_first=True, bidirectional=True)

    def forward(self, seq):
        # seq: (batch, 100, 3)  [length, direction, inter-arrival time]
        h, _ = self.lstm(seq)
        return h.max(dim=1).values


class StatEncoder(nn.Module):
    """Two-layer MLP with batch normalization (paper, Sec. 4.3)."""

    def __init__(self, stat_dim=23, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(stat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, stat):
        return self.mlp(stat)


class HeteroGATEncoder(nn.Module):
    """2-layer, 4-head GAT with type-specific linear projections.

    Flow nodes are projected from their statistical features; IP nodes are
    looked up from a learnable embedding table (one entry per unique IP
    address). Both node types then propagate through shared GAT layers over
    the heterogeneous graph (paper, Sec. 4.3).
    """

    def __init__(self, in_dim, hidden=128, heads=4, n_ips=0):
        super().__init__()
        self.hidden = hidden
        self.flow_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.ip_emb = nn.Embedding(max(n_ips, 1), hidden) if n_ips > 0 else None
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads)
        self.gat2 = GATConv(hidden, hidden, heads=1)

    def forward(self, x, edge_index, node_type, ip_index):
        flow_mask = node_type == 0
        h = torch.zeros(x.size(0), self.hidden, device=x.device)
        if flow_mask.any():
            h[flow_mask] = self.flow_proj(x[flow_mask])
        ip_mask = ~flow_mask
        if ip_mask.any() and self.ip_emb is not None:
            h[ip_mask] = self.ip_emb(ip_index[ip_mask].clamp(min=0))
        h = F.relu(self.gat1(h, edge_index))
        h = F.relu(self.gat2(h, edge_index))
        return h


class MGNN(nn.Module):
    """Multi-View Graph Neural Network with configurable fusion.

    Args:
        fusion: Fusion strategy, one of 'concat', 'avg', 'attn', 'attn_align'
                (paper, Table 6).
        seq_len: Length of the packet sequences (100 in the paper).
        stat_dim: Dimensionality of the statistical features (23).
        hidden: Hidden dimension of every encoder (128).
        n_ips: Number of unique IP addresses in the dataset (embedding size).
        gat_heads: Number of GAT attention heads (4).
    """

    def __init__(self, fusion='attn_align', seq_len=100, stat_dim=23,
                 hidden=128, n_ips=0, gat_heads=4, drop_views=None):
        super().__init__()
        self.fusion = fusion
        self.hidden = hidden
        # drop_views: tuple of view names to disable in the fusion
        # ('sequence', 'statistical', 'interaction'), used by the ablation.
        self.drop_views = set() if drop_views is None else set(drop_views)

        self.seq_encoder = SeqEncoder(input_dim=3, hidden=hidden)
        self.stat_encoder = StatEncoder(stat_dim=stat_dim, hidden=hidden)
        self.inter_encoder = HeteroGATEncoder(
            in_dim=stat_dim, hidden=hidden, heads=gat_heads, n_ips=n_ips)

        if fusion == 'concat':
            proj_dim = hidden * 3
        elif fusion == 'avg':
            proj_dim = hidden
        elif fusion in ('attn', 'attn_align'):
            # Cross-view attention (paper, Sec. 4.4): a single linear layer
            # scores each view, tanh activates the scores, softmax normalizes
            # them into per-flow fusion weights.
            self.attn_score = nn.Linear(hidden, 1, bias=False)
            proj_dim = hidden
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        self.use_align = (fusion == 'attn_align')

    def fuse_views(self, views):
        """Fuse the (B, 3, hidden) view stack with the configured strategy.

        For attention fusion also returns the per-flow view weights.
        """
        if self.fusion == 'concat':
            return views.reshape(views.size(0), -1), None
        if self.fusion == 'avg':
            return views.mean(dim=1), None
        scores = torch.tanh(self.attn_score(views)).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        return (weights.unsqueeze(-1) * views).sum(dim=1), weights

    def forward_all(self, x_seq, x_stat, x_all, edge_index, node_type,
                    ip_index, flow_mask):
        """Single-pass forward over a NeighborLoader batch.

        Encodes the three views, fuses them, and returns
        (logits, fused_hidden, views) so the classifier, the alignment loss
        and the decision-boundary regularizer share one encoder pass.

        Args:
            x_seq:   (B, 100, 3) packet sequences of flow nodes in the batch.
            x_stat:  (B, 23) statistical features of flow nodes in the batch.
            x_all:   (B_all, 23) features of all sampled nodes (flows + IPs).
            edge_index: (2, E) sampled heterogeneous edges.
            node_type:  (B_all,) 0 = flow, 1 = IP.
            ip_index:   (B_all,) global embedding id for IP nodes (-1 else).
            flow_mask:  (B_all,) boolean mask selecting flow nodes.
        """
        h_seq = self.seq_encoder(x_seq)
        h_stat = self.stat_encoder(x_stat)
        h_all = self.inter_encoder(x_all, edge_index, node_type, ip_index)
        h_inter = h_all[flow_mask]
        if "sequence" in self.drop_views:
            h_seq = torch.zeros_like(h_seq)
        if "statistical" in self.drop_views:
            h_stat = torch.zeros_like(h_stat)
        if "interaction" in self.drop_views:
            h_inter = torch.zeros_like(h_inter)
        views = torch.stack([h_seq, h_stat, h_inter], dim=1)
        h, _ = self.fuse_views(views)
        logits = self.classifier(h).squeeze(-1)
        return logits, h, views

    def forward(self, x_seq, x_stat, x_all, edge_index, node_type,
                ip_index, flow_mask):
        logits, _, _ = self.forward_all(x_seq, x_stat, x_all, edge_index,
                                        node_type, ip_index, flow_mask)
        return logits

    def get_attention_weights(self, x_seq, x_stat, x_all, edge_index,
                              node_type, ip_index, flow_mask):
        """Per-flow view weights for interpretability (paper, Sec. 5.5)."""
        if self.fusion not in ('attn', 'attn_align'):
            return None
        with torch.no_grad():
            h_seq = self.seq_encoder(x_seq)
            h_stat = self.stat_encoder(x_stat)
            h_all = self.inter_encoder(x_all, edge_index, node_type, ip_index)
            h_inter = h_all[flow_mask]
            views = torch.stack([h_seq, h_stat, h_inter], dim=1)
            _, weights = self.fuse_views(views)
        return weights  # (B, 3): [sequence, statistical, interaction]
