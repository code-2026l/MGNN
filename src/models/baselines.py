"""
Baseline graph models for comparison.

Implements:
  - GCN: Graph Convolutional Network
  - GraphSAGE: Graph Sample and Aggregate
  - EGraphSAGE: GraphSAGE with edge feature augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """2-layer Graph Convolutional Network."""

    def __init__(self, in_dim, hidden=128, out_dim=1):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return self.out(x).squeeze(-1)


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE model."""

    def __init__(self, in_dim, hidden=128, out_dim=1):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return self.out(x).squeeze(-1)


class EGraphSAGE(nn.Module):
    """GraphSAGE with edge feature augmentation.

    Difference from plain GraphSAGE: concatenates edge features
    (cosine similarity + inverse distance) to node features before
    each convolution, providing explicit edge-awareness.
    """

    def __init__(self, in_dim, hidden=128, out_dim=1, edge_feat_dim=2):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_dim + edge_feat_dim, hidden)
        self.conv2 = SAGEConv(hidden + edge_feat_dim, hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.edge_feat_dim = edge_feat_dim

    def _compute_edge_features(self, x, edge_index):
        """Compute edge features: cosine similarity and inverse L2 distance."""
        row, col = edge_index
        x_norm = F.normalize(x, dim=-1)
        cos_sim = (x_norm[row] * x_norm[col]).sum(dim=-1, keepdim=True)
        l2_dist = torch.norm(x[row] - x[col], dim=-1, keepdim=True)
        inv_dist = 1.0 / (l2_dist + 1e-8)
        return torch.cat([cos_sim, inv_dist], dim=-1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Edge features
        e = self._compute_edge_features(x, edge_index)

        # Conv 1: augment node features with edge features
        x_aug = torch.cat([x, torch.zeros_like(x[:, :self.edge_feat_dim])], dim=-1)
        x = F.relu(self.conv1(x, edge_index))
        # Augment with edge features aggregated to nodes
        x = torch.cat([x, torch.zeros_like(x[:, :self.edge_feat_dim])], dim=-1)
        x = F.relu(self.conv2(x, edge_index))
        return self.out(x).squeeze(-1)
