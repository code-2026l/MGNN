"""
Synthetic data generators for testing and development.

Generates data that mimics the statistical properties of CIC-IDS2017
(approx. 19.7% anomaly ratio) for quick pipeline testing.
"""

import torch
from torch.utils.data import TensorDataset

BENIGN_RATIO = 0.803
ANOMALY_RATIO = 0.197


def make_synthetic_graph_data(n_nodes=5000, feat_dim=78, edge_prob=0.005,
                               anom_ratio=ANOMALY_RATIO, seed=42):
    """Generate synthetic graph data mimicking CIC-IDS2017 statistics.

    Returns a torch_geometric Data object with train/val/test masks
    (60/20/20 split).
    """
    torch.manual_seed(seed)
    x = torch.randn(n_nodes, feat_dim)
    n_anom = int(n_nodes * anom_ratio)
    y = torch.zeros(n_nodes, dtype=torch.long)
    y[:n_anom] = 1
    x[:n_anom, :10] += 0.5

    n_edges = int(n_nodes * n_nodes * edge_prob)
    ei = torch.randint(0, n_nodes, (2, n_edges))
    mask = ei[0] != ei[1]
    ei = ei[:, mask]

    from torch_geometric.data import Data
    data = Data(x=x, edge_index=ei, y=y)

    perm = torch.randperm(n_nodes)
    n_train = int(0.6 * n_nodes)
    n_val = int(0.2 * n_nodes)
    data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.train_mask[perm[:n_train]] = True
    data.val_mask[perm[n_train:n_train + n_val]] = True
    data.test_mask[perm[n_train + n_val:]] = True
    return data


def make_synthetic_seqstat_data(n_samples=5000, seq_len=100, stat_dim=23,
                                  anom_ratio=ANOMALY_RATIO, seed=42):
    """Generate synthetic sequence + stat data for MGNN fusion experiments.

    Returns a TensorDataset of (seq, stat, labels).
    """
    torch.manual_seed(seed)
    n_anom = int(n_samples * anom_ratio)
    labels = torch.zeros(n_samples)
    labels[:n_anom] = 1

    seq = torch.randn(n_samples, seq_len, 1) * 0.1
    seq[:n_anom, :, 0] += 0.15

    stat = torch.randn(n_samples, stat_dim) * 0.1
    stat[:n_anom, :5] += 0.3

    perm = torch.randperm(n_samples)
    return TensorDataset(seq[perm], stat[perm], labels[perm])
