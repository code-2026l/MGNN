"""
Real-data graph construction for the graph-model comparison (Table 3).

Builds a graph whose nodes are real flows (from preprocessed features) and
whose edges come from data — not from a synthetic generator. Two edge types
mirror the paper's design intent:

  - communication edges   : flows sharing the same source IP/port trend
                            (approximated by feature-space neighbours)
  - temporal edges        : flows whose inter-arrival time falls below a
                            percentile threshold (requires a timestamp column)

Because the preprocessed .pt does not retain raw IPs/timestamps, we construct
the graph on the full feature matrix with a k-NN approximation. For CPython
workloads this is realised with scikit-learn for speed on large inputs.

Output is a torch_geometric Data object with train/val/test masks.
"""

import numpy as np
import torch

try:
    from sklearn.neighbors import NearestNeighbors
    _HAS_SKL = True
except Exception:  # pragma: no cover
    _HAS_SKL = False


def knn_adjacency(X, k, metric="cosine", batch=2048):
    """Build a k-NN adjacency edge_index on the full feature matrix.

    Uses the GPU when a CUDA device is available (blocked cosine similarity,
    seconds on an A100 even for millions of nodes), falling back to
    scikit-learn / numpy otherwise.
    """
    n = X.shape[0]
    device = None
    try:
        import torch as _t
        if _t.cuda.is_available():
            device = _t.device("cuda")
    except Exception:  # pragma: no cover
        device = None

    if device is not None:
        Xn = torch.from_numpy(
            X.astype(np.float32) / (np.linalg.norm(X, axis=1,
                                                   keepdims=True) + 1e-8)
        ).to(device)
        dst_all = []
        k_eff = min(k + 1, n)
        # Blocked similarity: iterate over source chunks to bound memory.
        for s in range(0, n, batch):
            sim = Xn[s:s + batch] @ Xn.t()          # (chunk, n)
            _, idx = sim.topk(k_eff, dim=-1)        # self first (cos=1)
            dst_all.append(idx[:, 1:k + 1].reshape(-1).cpu())
        dst = torch.cat(dst_all)
        src = torch.arange(n, device="cpu").unsqueeze(1).expand(-1, k).reshape(-1)
        return torch.stack([src, dst], dim=0)
    elif _HAS_SKL:
        nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric=metric,
                              algorithm="brute", n_jobs=-1)
        nn.fit(X)
        _, idx = nn.kneighbors(X)
    else:  # pragma: no cover - CPU-only fallback
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        sim = Xn @ Xn.T
        idx = np.argsort(-sim, axis=1)[:, : k + 1]
    # index 0 is the node itself (cosine self == 1); drop it.
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1: k + 1].reshape(-1)
    edge_index = np.stack([src, dst])
    return torch.tensor(edge_index, dtype=torch.long)


def build_graph(features, labels, k=5, val_frac=0.1, test_frac=0.2, seed=42):
    """Build a real k-NN flow graph with deterministic train/val/test masks.

    Returns a torch_geometric.data.Data object (or a lightweight dict when
    PyG is unavailable).
    """
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.longlong)
    n = X.shape[0]

    edge_index = knn_adjacency(X, k=k)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    test_mask[perm[:n_test]] = True
    val_mask[perm[n_test:n_test + n_val]] = True
    train_mask[perm[n_test + n_val:]] = True

    x = torch.tensor(X)
    edge_idx = edge_index.long()

    try:
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_idx, y=torch.tensor(y))
        data.train_mask = torch.from_numpy(train_mask)
        data.val_mask = torch.from_numpy(val_mask)
        data.test_mask = torch.from_numpy(test_mask)
        return data
    except Exception:  # pragma: no cover - PyG absent
        return {
            "x": x, "edge_index": edge_idx, "y": torch.tensor(y),
            "train_mask": torch.from_numpy(train_mask),
            "val_mask": torch.from_numpy(val_mask),
            "test_mask": torch.from_numpy(test_mask),
        }


def build_from_pt(data, k=5, seed=42):
    """Build a real graph directly from a preprocessed .pt dict."""
    return build_graph(data["features"].numpy(),
                       data["labels"].numpy(), k=k, seed=seed)