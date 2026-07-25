"""
MGNN: Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection.

Implements the three-view fusion architecture:
  - Sequence view: BiLSTM encoder for packet-length sequences
  - Statistical view: MLP encoder for distributional features
  - Interaction view: GAT encoder on k-NN graph of flow features

Supports four fusion strategies: concat, avg, attn, attn_align.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATInteractionEncoder(nn.Module):
    """Graph Attention Network encoder for the interaction view.

    Builds a k-NN graph from batch feature vectors and applies
    multi-head GAT convolution to capture flow-level interactions.
    Falls back to a linear projection when PyTorch Geometric is unavailable.
    """

    def __init__(self, in_dim, hidden=128, heads=4, use_pyg=True):
        super().__init__()
        self.use_pyg = use_pyg
        self.hidden = hidden

        if use_pyg:
            try:
                from torch_geometric.nn import GATConv
                self.gat_conv1 = GATConv(in_dim, hidden // heads, heads=heads)
                self.gat_conv2 = GATConv(hidden, hidden, heads=1)
                self._gat_available = True
            except ImportError:
                print("  [GATInteractionEncoder] PyTorch Geometric not found, "
                      "falling back to linear projection.")
                self._gat_available = False
        else:
            self._gat_available = False

        if not self._gat_available:
            self.fallback_proj = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
            )

    def _build_knn_graph(self, x, k=5):
        """Build k-NN graph from feature vectors (batched)."""
        n = x.size(0)
        if n <= 1:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            return edge_index

        # Compute pairwise cosine similarity
        x_norm = F.normalize(x, dim=-1)
        sim = x_norm @ x_norm.t()

        # Get top-k neighbours (excluding self)
        k_eff = min(k + 1, n)
        _, idx = sim.topk(k_eff, dim=-1)

        src = torch.arange(n, device=x.device).unsqueeze(1).expand(-1, k_eff - 1)
        dst = idx[:, 1:]  # exclude self
        edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
        return edge_index

    def forward(self, x):
        if self._gat_available and x.size(0) > 1:
            edge_index = self._build_knn_graph(x)
            if edge_index.size(1) == 0:
                return self.fallback_proj(x)
            h = self.gat_conv1(x, edge_index)
            h = F.relu(h)
            h = self.gat_conv2(h, edge_index)
            return F.relu(h)
        else:
            return self.fallback_proj(x)


class SeqEncoder(nn.Module):
    """BiLSTM encoder for packet-length sequences."""

    def __init__(self, input_dim=1, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden // 2, batch_first=True, bidirectional=True)

    def forward(self, seq):
        # seq: (batch, seq_len, input_dim)
        h, _ = self.lstm(seq)
        h_seq, _ = torch.max(h, dim=1)  # max-over-time pooling
        return h_seq


class StatEncoder(nn.Module):
    """MLP encoder for statistical features."""

    def __init__(self, stat_dim=23, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(stat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, hidden),
        )

    def forward(self, stat):
        return self.mlp(stat)


class MGNN(nn.Module):
    """Multi-View Graph Neural Network with configurable fusion.

    Args:
        fusion: Fusion strategy ('concat', 'avg', 'attn', 'attn_align').
        seq_len: Length of input packet-length sequences.
        stat_dim: Dimensionality of statistical features.
        hidden: Hidden dimension for all encoders.
        use_gat: Whether to use GAT for the interaction view.
        gat_heads: Number of GAT attention heads.
    """

    def __init__(self, fusion='attn_align', seq_len=100, stat_dim=23,
                 hidden=128, use_gat=True, gat_heads=4):
        super().__init__()
        self.fusion = fusion
        self.hidden = hidden

        # Three-view encoders
        self.seq_encoder = SeqEncoder(input_dim=1, hidden=hidden)
        self.stat_encoder = StatEncoder(stat_dim=stat_dim, hidden=hidden)
        self.inter_encoder = GATInteractionEncoder(
            in_dim=stat_dim, hidden=hidden, heads=gat_heads, use_pyg=use_gat,
        )

        # Fusion-specific components
        if fusion == 'concat':
            proj_dim = hidden * 3
        elif fusion == 'avg':
            proj_dim = hidden
        elif fusion in ('attn', 'attn_align'):
            self.attn_q = nn.Linear(hidden, 64)
            self.attn_w = nn.Linear(64, 1, bias=False)
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

    def encode_views(self, seq, stat):
        """Encode all three views and return individual view representations."""
        h_seq = self.seq_encoder(seq)
        h_stat = self.stat_encoder(stat)
        h_inter = self.inter_encoder(stat)
        return h_seq, h_stat, h_inter

    def fuse_views(self, h_seq, h_stat, h_inter):
        """Fuse three view representations using the configured strategy."""
        if self.fusion == 'concat':
            return torch.cat([h_seq, h_stat, h_inter], dim=-1)
        elif self.fusion == 'avg':
            return (h_seq + h_stat + h_inter) / 3.0
        else:  # attn / attn_align
            views = torch.stack([h_seq, h_stat, h_inter], dim=1)  # (batch, 3, hidden)
            scores = self.attn_w(torch.tanh(self.attn_q(views))).squeeze(-1)
            weights = F.softmax(scores, dim=1)
            return torch.sum(weights.unsqueeze(-1) * views, dim=1), weights

    def forward(self, seq, stat):
        h_seq, h_stat, h_inter = self.encode_views(seq, stat)
        if self.fusion in ('attn', 'attn_align'):
            h, _ = self.fuse_views(h_seq, h_stat, h_inter)
        else:
            h = self.fuse_views(h_seq, h_stat, h_inter)
        return self.classifier(h).squeeze(-1)

    def get_views(self, seq, stat):
        """Return stacked view representations for alignment loss computation."""
        with torch.no_grad():
            h_seq, h_stat, h_inter = self.encode_views(seq, stat)
        return torch.stack([h_seq, h_stat, h_inter], dim=1)

    def get_attention_weights(self, seq, stat):
        """Return per-view attention weights for interpretability."""
        if self.fusion not in ('attn', 'attn_align'):
            return None
        with torch.no_grad():
            h_seq, h_stat, h_inter = self.encode_views(seq, stat)
            _, weights = self.fuse_views(h_seq, h_stat, h_inter)
        return weights  # (batch, 3): [seq_weight, stat_weight, inter_weight]
