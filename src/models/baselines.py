"""
Baseline models for the paper's comparison tables.

Graph baselines (GCN, GraphSAGE, GAT, E-GraphSAGE, GDN, FN-GNN) run on a
flow-level graph; deep baselines (DNN, LSTM, BiLSTM, CNN-LSTM) consume the
statistical and sequence views directly. All follow the hyperparameters of
the paper's implementation section (Adam, lr=1e-3, weight_decay=1e-5,
hidden=128).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


def _require_pyg():
    try:
        import torch_geometric.nn  # noqa: F401
        return torch_geometric.nn
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Graph baselines need PyTorch Geometric. Install it with:  "
            "pip install torch_geometric") from e


class GCN(nn.Module):
    """2-layer Graph Convolutional Network."""

    def __init__(self, in_dim, hidden=128, out_dim=1):
        super().__init__()
        gnn = _require_pyg()
        self.conv1 = gnn.GCNConv(in_dim, hidden)
        self.conv2 = gnn.GCNConv(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return self.out(x).squeeze(-1)


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE model."""

    def __init__(self, in_dim, hidden=128, out_dim=1):
        super().__init__()
        gnn = _require_pyg()
        self.conv1 = gnn.SAGEConv(in_dim, hidden)
        self.conv2 = gnn.SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return self.out(x).squeeze(-1)


class GATNet(nn.Module):
    """2-layer, 4-head Graph Attention Network."""

    def __init__(self, in_dim, hidden=128, out_dim=1, heads=4):
        super().__init__()
        gnn = _require_pyg()
        self.conv1 = gnn.GATConv(in_dim, hidden // heads, heads=heads)
        self.conv2 = gnn.GATConv(hidden, hidden, heads=1)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return self.out(x).squeeze(-1)


class EGraphSAGE(nn.Module):
    """E-GraphSAGE (Gao et al., 2021): GraphSAGE augmented with edge features.

    Edge attributes [cosine similarity, inverse L2 distance] are
    mean-aggregated per node and concatenated to the node features before
    each SAGE layer, implemented with index ops so it runs on any PyG
    version without torch_sparse/torch_scatter.
    """

    def __init__(self, in_dim, hidden=128, out_dim=1, edge_feat_dim=2):
        super().__init__()
        gnn = _require_pyg()
        self.conv1 = gnn.SAGEConv(in_dim + edge_feat_dim, hidden)
        self.conv2 = gnn.SAGEConv(hidden + edge_feat_dim, hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.edge_feat_dim = edge_feat_dim

    def _edge_features(self, x, edge_index):
        row, col = edge_index
        x_norm = F.normalize(x, dim=-1)
        cos = (x_norm[row] * x_norm[col]).sum(dim=-1, keepdim=True)
        l2 = torch.norm(x[row] - x[col], dim=-1, keepdim=True)
        inv = 1.0 / (l2 + 1e-8)
        return torch.cat([cos, inv], dim=-1)

    def _aggregate(self, x, edge_index):
        d = self.edge_feat_dim
        e = self._edge_features(x, edge_index)
        row = edge_index[0]
        agg = torch.zeros(x.size(0), d, device=x.device)
        agg.index_add_(0, row, e)
        deg = torch.bincount(row, minlength=x.size(0)).clamp(min=1).float()
        agg = agg / deg.unsqueeze(1)
        return torch.cat([x, agg], dim=-1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(self._aggregate(x, edge_index), edge_index))
        x = F.relu(self.conv2(self._aggregate(x, edge_index), edge_index))
        return self.out(x).squeeze(-1)


class GDN(nn.Module):
    """Graph Deviation Network (Zheng et al., 2022).

    Node embeddings are enriched by graph-attention aggregation over the
    flow graph, then scored by a deviation-based head. The attention uses
    the scaled-dot-product formulation of the original GDN.
    """

    def __init__(self, in_dim, hidden=128, out_dim=1, heads=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.attn_q = nn.Linear(hidden, hidden)
        self.attn_k = nn.Linear(hidden, hidden)
        self.attn_v = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, out_dim)

    def forward(self, data):
        h = self.embed(data.x)
        row, col = data.edge_index
        q = self.attn_q(h)
        k = self.attn_k(h)
        v = self.attn_v(h)
        alpha = (q[row] * k[col]).sum(-1) / (h.size(-1) ** 0.5)
        # softmax over in-neighbours per node
        alpha = alpha.softmax(dim=0)
        agg = torch.zeros_like(v)
        agg.index_add_(0, row, alpha.unsqueeze(-1) * v[col])
        return self.score(h + agg).squeeze(-1)


class FNGNN(nn.Module):
    """FN-GNN (Tran and Park, 2024): feature-embedded GNN with residuals.

    Statistical features are embedded, propagated through two GCN layers
    with residual connections, and scored.
    """

    def __init__(self, in_dim, hidden=128, out_dim=1):
        super().__init__()
        gnn = _require_pyg()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.conv1 = gnn.GCNConv(hidden, hidden)
        self.conv2 = gnn.GCNConv(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x = self.embed(data.x)
        x = F.relu(self.conv1(x, data.edge_index)) + x
        x = F.relu(self.conv2(x, data.edge_index)) + x
        return self.out(x).squeeze(-1)


class DNN(nn.Module):
    """Multi-layer perceptron over the statistical view."""

    def __init__(self, in_dim, hidden=128, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTMNet(nn.Module):
    """LSTM over the packet sequence (max-pooled over time)."""

    def __init__(self, in_dim=3, hidden=128, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, seq):
        h, _ = self.lstm(seq)
        return self.out(h.max(dim=1).values).squeeze(-1)


class BiLSTMNet(nn.Module):
    """Bidirectional LSTM over the packet sequence."""

    def __init__(self, in_dim=3, hidden=128, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden // 2, batch_first=True,
                            bidirectional=True)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, seq):
        h, _ = self.lstm(seq)
        return self.out(h.max(dim=1).values).squeeze(-1)


class CNNLSTMNet(nn.Module):
    """1D CNN over the packet sequence followed by an LSTM."""

    def __init__(self, in_dim=3, hidden=128, out_dim=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, seq):
        # seq: (B, 100, 3) -> (B, 64, 100) -> (B, 100, 64)
        x = self.conv(seq.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(x)
        return self.out(h.max(dim=1).values).squeeze(-1)
