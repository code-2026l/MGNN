"""
Baseline graph models for comparison.

Requires PyTorch Geometric (torch-geometric>=2.4). If it is not installed a
clear error is raised, telling the user how to enable these baselines. The
MGNN interaction encoder itself does NOT require PyG (it falls back to a
linear projection) — only these graph baselines do.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_pyg():
    try:
        import torch_geometric.nn  # noqa: F401
        return torch_geometric.nn
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Graph baselines (GCN/GraphSAGE/E-GraphSAGE) need PyTorch Geometric. "
            "Install it with:  pip install torch_geometric  (see "
            "https://pytorch-geometric.readthedocs.io/). The MGNN model itself "
            "does not require PyG — run experiments/run_table6.py without it."
        ) from e


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


class EGraphSAGE(nn.Module):
    """GraphSAGE augmented with edge attributes.

    Edge features are [cosine similarity, inverse L2 distance]. They are
    aggregated per node (mean over incident edges) and concatenated to the
    node features before each SAGE layer, so the message passing actually
    consumes them. This is implemented with index ops only, so it works on
    any PyG version without torch_sparse or torch_scatter.
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
        """Mean-aggregate per-node edge features and cat to node features."""
        d = self.edge_feat_dim
        e = self._edge_features(x, edge_index)          # (E, d)
        row = edge_index[0]
        agg = torch.zeros(x.size(0), d, device=x.device)
        agg.index_add_(0, row, e)                       # sum into source node
        deg = torch.bincount(row, minlength=x.size(0)).clamp(min=1).float()
        agg = agg / deg.unsqueeze(1)                     # -> mean
        return torch.cat([x, agg], dim=-1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(self._aggregate(x, edge_index), edge_index))
        x = F.relu(self.conv2(self._aggregate(x, edge_index), edge_index))
        return self.out(x).squeeze(-1)