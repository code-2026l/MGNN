"""
Heterogeneous traffic graph construction (paper, Sec. 4.2).

Two node types -- flow nodes (one per flow, initialized with the 23-dim
statistical features) and IP nodes (initialized with a learnable embedding) --
are connected by three edge types, all built with data-driven thresholds:

  - temporal edges      : flows whose arrival-time difference falls below a
                          data-driven threshold; a typical flow connects to
                          3-5 temporal neighbours (median threshold ~3.2 s on
                          CIC-IDS2017)
  - volume edges        : flows whose statistical cosine similarity exceeds
                          the 95th-percentile threshold (0.78 on CIC-IDS2017)
  - communication edges : each flow to its source and destination IP

The graph is represented homogeneously for PyG's NeighborLoader: nodes 0..N-1
are flows and nodes N..N+U-1 are IPs. Node attributes carried through sampling
are:

  - x_all      : (N+U, 23) flow statistical features (zeros for IP rows)
  - node_type  : (N+U,) 0 = flow, 1 = IP
  - ip_index   : (N+U,) embedding id for IP nodes (-1 for flows)
  - n_ips      : number of IP nodes
"""

import numpy as np
import torch

try:
    from sklearn.neighbors import NearestNeighbors
    _HAS_SKL = True
except Exception:  # pragma: no cover
    _HAS_SKL = False


# ---------------------------------------------------------------------------
# Temporal edges
# ---------------------------------------------------------------------------

def temporal_threshold(timestamps, n_pairs=2_000_000, percentile=5.0,
                       window=128, seed=42):
    """Data-driven threshold for temporal edges (paper, Sec. 4.2).

    Two flows are connected when their arrival-time difference falls below
    the p-th percentile of pairwise differences (p = 5 by default), which
    captures coordinated attacks such as port scans. The percentile is
    estimated on a bounded sample of pairwise differences within a sliding
    window of ``window`` consecutive flows so the cost stays O(N); on
    CIC-IDS2017 a typical flow then connects to 3-5 temporal neighbours.
    """
    ts = np.sort(np.asarray(timestamps, dtype=np.float64))
    n = len(ts)
    if n < 2:
        return 1.0
    w = min(window, n - 1)
    if w < 1:
        return 1.0
    rng = np.random.RandomState(seed)
    n_pairs = min(n_pairs, n * w)
    i = rng.randint(0, n - w, n_pairs)
    j = i + rng.randint(1, w + 1, n_pairs)
    diffs = ts[j] - ts[i]
    diffs = diffs[diffs > 0]
    if diffs.size < 100:
        return float(np.median(ts[1:] - ts[:-1]) + 1e-6)
    return float(np.percentile(diffs, percentile))


def temporal_edges(timestamps, tau, max_window=16):
    """Forward (causal) temporal edges within ``tau`` seconds.

    Each flow connects to the flows arriving within ``tau`` after it, capped
    at ``max_window`` to bound the burst degree. Vectorised. Returns a
    (2, E) int64 array.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    n = len(ts)
    if n < 2 or not np.isfinite(tau) or tau <= 0:
        return np.zeros((2, 0), dtype=np.int64)

    order = np.argsort(ts, kind="stable")
    ts_s = ts[order]
    pos = np.arange(n)
    nbrs = np.searchsorted(ts_s, ts_s + tau, side="right") - 1
    nbrs = np.minimum(nbrs, pos + max_window)
    nbrs = np.minimum(nbrs, n - 1)
    counts = np.maximum(nbrs - pos, 0)
    total = int(counts.sum())
    if total == 0:
        return np.zeros((2, 0), dtype=np.int64)

    end_pos = np.cumsum(counts)
    offs = np.arange(total) - np.repeat(end_pos - counts, counts)
    src = np.repeat(order, counts)
    dst = order[np.repeat(pos, counts) + offs + 1]
    return np.stack([src, dst])


# ---------------------------------------------------------------------------
# Volume edges
# ---------------------------------------------------------------------------

def volume_threshold(features, n_pairs=2_000_000, seed=42):
    """Data-driven threshold: 95th percentile of pairwise cosine similarity.

    Estimated on a random sample of flow pairs (bounded to 2M pairs), giving
    the 0.78 threshold reported for CIC-IDS2017.
    """
    X = np.asarray(features, dtype=np.float32)
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    i = rng.randint(0, n, n_pairs)
    j = rng.randint(0, n, n_pairs)
    keep = i != j
    i, j = i[keep][:n_pairs], j[keep][:n_pairs]
    if len(i) < 100:
        return 0.9
    nrm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = (nrm[i] * nrm[j]).sum(axis=1)
    return float(np.percentile(sim, 95))


def volume_edges(features, threshold, k=8, batch=2048):
    """Volume edges: top-k cosine neighbours above the threshold.

    Returns forward edges (2, E) int64; both directions are added and the
    full set deduplicated by ``build_hetero_graph``.
    """
    X = np.asarray(features, dtype=np.float32)
    n = X.shape[0]
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64)

    nrm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    src_all, dst_all = [], []
    try:
        import torch as _t
        if _t.cuda.is_available():
            Xn = _t.from_numpy(nrm).cuda()
            k_eff = min(k + 1, n)
            for s in range(0, n, batch):
                sim = Xn[s:s + batch] @ Xn.t()          # (B, n)
                topv, topi = sim.topk(k_eff, dim=-1)
                rows = _t.arange(s, s + sim.size(0), device=Xn.device)
                self_mask = topi == rows.unsqueeze(1)
                topi[self_mask] = -1
                topv[self_mask] = -1.0
                keep = (topv > threshold) & (topi >= 0)
                r, c = keep.nonzero(as_tuple=True)
                if r.numel():
                    src_all.append((rows[r]).cpu())
                    dst_all.append(topi[r, c].cpu())
            if src_all:
                src = torch.cat(src_all).numpy()
                dst = torch.cat(dst_all).numpy()
                return np.stack([src, dst])
    except Exception:
        pass

    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine",
                          algorithm="brute", n_jobs=-1)
    nn.fit(X)
    dist, idx = nn.kneighbors(X)
    sim = 1.0 - dist
    keep = sim[:, 1:] > threshold
    src = np.repeat(np.arange(n)[:, None], k, axis=1)
    dst = idx[:, 1:]
    ss, dd = src[keep], dst[keep]
    return np.stack([ss, dd])


# ---------------------------------------------------------------------------
# Communication edges
# ---------------------------------------------------------------------------

def communication_edges(src_ip, dst_ip, n_ips):
    """Flow <-> IP edges (both directions).

    Returns a (2, E) int64 array with node ids in [0, N+U).
    """
    n = len(src_ip)
    parts = []
    for ip_ids in (src_ip, dst_ip):
        valid = ip_ids >= 0
        if not valid.any():
            continue
        flows = np.nonzero(valid)[0].astype(np.int64)
        ips = ip_ids[valid].astype(np.int64) + n         # IP nodes start at N
        parts.append(np.stack([flows, ips]))
        parts.append(np.stack([ips, flows]))
    if not parts:
        return np.zeros((2, 0), dtype=np.int64)
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Full heterogeneous graph
# ---------------------------------------------------------------------------

def _dedup_undirected(edge_index, n):
    """Remove self-loops and duplicate undirected pairs (keep both directions)."""
    e = edge_index
    if e.shape[1] == 0:
        return e
    keep = e[0] != e[1]
    e = e[:, keep]
    if e.shape[1] == 0:
        return e
    lo = np.minimum(e[0], e[1])
    hi = np.maximum(e[0], e[1])
    key = lo.astype(np.int64) * n + hi.astype(np.int64)
    _, uniq = np.unique(key, return_index=True)
    e = e[:, np.sort(uniq)]
    # re-add the reverse direction
    return np.concatenate([e, e[::-1]], axis=1)


def build_hetero_graph(stat, labels, src_ip, dst_ip, timestamps,
                       volume_thr=None, seed=42):
    """Build the heterogeneous traffic graph from a preprocessed .pt dict.

    Two node types are represented homogeneously for PyG's NeighborLoader:
    nodes 0..N-1 are flow nodes (initialized with the statistical features)
    and nodes N..N+U-1 are IP nodes. ``ip_index`` carries the *global* IP
    embedding id for every IP node (used by the type-specific IP projection),
    so IP embeddings stay shared across train/val/test subgraphs.

    Returns a dict of torch tensors (PyG-free) ready for NeighborLoader:

      x_all, node_type, ip_index, edge_index_all, n_ips, n_flows,
      src_ip, dst_ip, timestamp, labels, volume_threshold,
      temporal_threshold, n_edges
    """
    stat = np.asarray(stat, dtype=np.float32)
    y = np.asarray(labels).astype(np.int64)
    n = stat.shape[0]
    src_ip = np.asarray(src_ip, dtype=np.int64)
    dst_ip = np.asarray(dst_ip, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    # --- IP node ids (global, for the learnable IP embedding) ---
    valid = (src_ip >= 0) | (dst_ip >= 0)
    ips = np.unique(np.concatenate([src_ip[valid], dst_ip[valid]])) \
        if valid.any() else np.array([], dtype=np.int64)
    ip2local = {int(ip): i for i, ip in enumerate(ips)}
    src_local = np.array([ip2local.get(int(v), -1) for v in src_ip],
                         dtype=np.int64)
    dst_local = np.array([ip2local.get(int(v), -1) for v in dst_ip],
                         dtype=np.int64)
    n_ips = len(ips)

    # --- temporal edges ---
    tau = temporal_threshold(timestamps)
    et = temporal_edges(timestamps, tau)

    # --- volume edges ---
    if volume_thr is None:
        volume_thr = volume_threshold(stat)
    ev = volume_edges(stat, volume_thr)

    # --- communication edges ---
    ec = communication_edges(src_local, dst_local, n_ips)

    # --- merge flow-flow edges (dedup, self-loops removed) ---
    if et.shape[1] and ev.shape[1]:
        ff = np.concatenate([et, ev], axis=1)
    else:
        ff = et if et.shape[1] else ev
    ff = _dedup_undirected(ff, n) if ff.shape[1] else ff

    if ec.shape[1] and ff.shape[1]:
        edge_all = np.concatenate([ff, ec], axis=1)
    else:
        edge_all = ff if ff.shape[1] else ec

    x_all = np.zeros((n + n_ips, stat.shape[1]), dtype=np.float32)
    x_all[:n] = stat
    node_type = np.zeros(n + n_ips, dtype=np.int64)
    node_type[n:] = 1
    ip_index = np.full(n + n_ips, -1, dtype=np.int64)
    ip_index[n:] = ips

    return {
        "x_all": torch.from_numpy(x_all),
        "node_type": torch.from_numpy(node_type),
        "ip_index": torch.from_numpy(ip_index),
        "edge_index_all": torch.from_numpy(edge_all) if edge_all.shape[1] \
            else torch.zeros((2, 0), dtype=torch.long),
        "n_ips": n_ips,
        "n_flows": n,
        "src_ip": torch.from_numpy(src_local),
        "dst_ip": torch.from_numpy(dst_local),
        "timestamp": torch.from_numpy(timestamps),
        "labels": torch.from_numpy(y),
        "volume_threshold": float(volume_thr),
        "temporal_threshold": float(tau),
        "n_edges": int(edge_all.shape[1]),
    }


def build_from_pt(data, seed=42):
    """Build the heterogeneous graph directly from a preprocessed .pt dict."""
    n = len(data["labels"])
    src = data.get("src_ip", torch.full((n,), -1, dtype=torch.long))
    dst = data.get("dst_ip", torch.full((n,), -1, dtype=torch.long))
    ts = data.get("timestamp", torch.arange(n, dtype=torch.double))
    return build_hetero_graph(
        data["stat"].numpy(), data["labels"].numpy(),
        src.numpy(), dst.numpy(), ts.numpy(), seed=seed)


# ---------------------------------------------------------------------------
# PyG wrapping (NeighborLoader)
# ---------------------------------------------------------------------------

def make_pyg_data(graph, seq=None, stat=None):
    """Wrap a heterogeneous graph dict into a torch_geometric Data object.

    ``seq`` and ``stat`` are flow-only tensors; they are zero-padded to the
    full node count so NeighborLoader slices them along the node dimension.
    IP rows are masked out by ``node_type`` during forward passes.
    """
    from torch_geometric.data import Data
    n_total = graph["x_all"].size(0)
    y_all = torch.zeros(n_total, dtype=graph["labels"].dtype)
    y_all[:graph["labels"].size(0)] = graph["labels"]
    data = Data(x=graph["x_all"], edge_index=graph["edge_index_all"], y=y_all)
    data.node_type = graph["node_type"]
    data.ip_index = graph["ip_index"]
    if seq is not None:
        s = torch.zeros(n_total, *seq.shape[1:], dtype=seq.dtype)
        s[:seq.size(0)] = seq
        data.x_seq = s
    if stat is not None:
        st = torch.zeros(n_total, stat.shape[1], dtype=stat.dtype)
        st[:stat.size(0)] = stat
        data.x_stat = st
    return data


def make_neighbor_loader(data, seed_idx, batch_size=2048,
                         num_neighbors=(10, 10), shuffle=False,
                         num_workers=8, pin_memory=True):
    """Build a NeighborLoader over the heterogeneous graph.

    ``seed_idx`` selects the flow nodes used as training/evaluation seeds
    (all seeds are flow nodes; IP nodes enter only as sampled neighbors).
    """
    from torch_geometric.loader import NeighborLoader
    if not isinstance(seed_idx, torch.Tensor):
        seed_idx = torch.tensor(seed_idx, dtype=torch.long)
    return NeighborLoader(
        data, num_neighbors=list(num_neighbors), batch_size=batch_size,
        input_nodes=seed_idx, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory and num_workers > 0,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


# ---------------------------------------------------------------------------
# k-NN flow graph (used by GDN and as fallback)
# ---------------------------------------------------------------------------

def knn_adjacency(X, k, metric="cosine", batch=2048):
    """Build a k-NN adjacency edge_index on the full feature matrix.

    Uses the GPU when a CUDA device is available (blocked cosine similarity),
    falling back to scikit-learn / numpy otherwise.
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
        for s in range(0, n, batch):
            sim = Xn[s:s + batch] @ Xn.t()
            _, idx = sim.topk(k_eff, dim=-1)
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
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1: k + 1].reshape(-1)
    edge_index = np.stack([src, dst])
    return torch.tensor(edge_index, dtype=torch.long)


def flow_flow_graph(graph):
    """Homogeneous flow-flow graph (temporal + volume edges) for baselines.

    Builds a torch_geometric Data (or dict) with train/val/test masks from a
    heterogeneous graph dict.
    """
    n = graph["n_flows"]
    ff = graph["edge_index_all"].numpy()
    if ff.shape[1]:
        ff = ff[:, (ff[0] < n) & (ff[1] < n)]
    edge_index = torch.from_numpy(ff) if ff.shape[1] else \
        torch.zeros((2, 0), dtype=torch.long)
    y = graph["labels"]

    data = {
        "x": graph["x_all"][:n],
        "edge_index": edge_index,
        "y": y,
    }
    try:
        from torch_geometric.data import Data
        return Data(**data)
    except Exception:  # pragma: no cover - PyG absent
        return data
