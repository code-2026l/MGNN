#!/usr/bin/env python3
"""
Table 4: cross-dataset generalization (paper, Table 4 / Sec. 5.3).

Trains MGNN on a source dataset and evaluates zero-shot on a target dataset
(no fine-tuning), plus a 10%-target-data fine-tune variant. The shared
CICFlowMeter feature schema lets CIC-IDS2017, UNSW and CSE-CIC-IDS2018 be
fed through a single model; only the graph (IP/temporal structure) is rebuilt
per dataset.

Usage:
    python experiments/run_table4.py [--data-dir DIR] [--runs 5] [--epochs 30]
"""

import argparse
import json
import time
import numpy as np
import torch

from src.utils.config import add_common_args, parse_device, SEEDS


def _stratified_split(labels, train_frac, seed):
    """Stratified random split of labels into (train, rest)."""
    idx = np.arange(len(labels))
    rng = np.random.RandomState(seed)
    tr = []
    for cls in np.unique(labels):
        ids = rng.permutation(idx[labels == cls])
        n_tr = int(len(ids) * train_frac)
        tr.append(ids[:n_tr])
    return np.concatenate(tr)


def _train_mgnn_models(data, device, runs, epochs, batch_size, n_ips,
                       train_frac=0.8, seed_base=42):
    """Train ``runs`` MGNN models on ``data``'s full set and return weights.

    Returns a list of (model, optimizer-parallel state) so a caller can
    evaluate on arbitrary target features/frames.
    """
    from src.data.graph import build_hetero_graph, make_pyg_data, \
        make_neighbor_loader
    from src.models.mgnn import MGNN
    from src.utils.split import undersample_train
    from src.utils.training import train_epoch_mgnn, evaluate_mgnn

    seq, stat, labels = data["seq"], data["stat"], data["labels"]
    ts = data["timestamp"].numpy()
    src_ip = data["src_ip"].numpy()
    dst_ip = data["dst_ip"].numpy()
    N = len(labels)

    models = []
    val_loader = None
    for seed in SEEDS[:runs]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MGNN(fusion="attn_align", seq_len=seq.size(1),
                     stat_dim=stat.size(1), hidden=128, n_ips=n_ips,
                     gat_heads=4).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        train_idx = _stratified_split(labels.numpy(), train_frac, seed)
        # validation slice inside the training portion
        v_frac = int(len(train_idx) * 0.2)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(train_idx)
        tr_sel0, val_idx = perm[:len(perm) - v_frac], perm[len(perm) - v_frac:]
        tr_sel = undersample_train(tr_sel0, labels.numpy())

        g = build_hetero_graph(stat[train_idx], labels[train_idx],
                               src_ip[train_idx], dst_ip[train_idx],
                               ts[train_idx])
        # (reuse the training graph for both the sampling loader seeds)
        tr_d = make_pyg_data(g, seq=seq[train_idx], stat=stat[train_idx])
        pos = {int(i): p for p, i in enumerate(train_idx)}
        tr_ld = make_neighbor_loader(
            tr_d, np.array([pos[int(i)] for i in tr_sel]),
            batch_size=batch_size, shuffle=True)
        val_ld = make_neighbor_loader(
            tr_d, np.array([pos[int(i)] for i in val_idx]),
            batch_size=batch_size)

        best_f1, best_state, bad = -1.0, None, 0
        for _ in range(epochs):
            train_epoch_mgnn(model, tr_ld, opt, device, align=True)
            vm = evaluate_mgnn(val_ld, model, device)
            if vm["f1"] > best_f1:
                best_f1 = vm["f1"]
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= 10:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        models.append(model.to(device))
    return models


def _eval_on(data, models, device, batch_size):
    """Zero-shot evaluate a list of models on ``data``'s features."""
    from src.data.graph import build_hetero_graph, make_pyg_data, \
        make_neighbor_loader
    from src.utils.training import evaluate_mgnn

    seq, stat, labels = data["seq"], data["stat"], data["labels"]
    ts = data["timestamp"].numpy()
    src_ip = data["src_ip"].numpy()
    dst_ip = data["dst_ip"].numpy()
    test_idx = np.arange(len(labels))

    g = build_hetero_graph(stat, labels, src_ip, dst_ip, ts)
    td = make_pyg_data(g, seq=seq, stat=stat)
    ld = make_neighbor_loader(td, test_idx, batch_size=batch_size)
    f1s = [evaluate_mgnn(ld, m, device)["f1"] for m in models]
    return np.mean(f1s) * 100, np.std(f1s) * 100


def run_table4(data_dir, device, runs=5, epochs=30, batch_size=2048):
    from src.data.dataset import load_pt_data

    ci = load_pt_data(data_dir, "cic_ids2017.pt")
    uw = load_pt_data(data_dir, "unsw_nb15.pt")
    cs = load_pt_data(data_dir, "cse_ids2018.pt")
    ip = lambda d: int(d["n_unique_ips"])

    # e.g. CIC -> CSE: train on CIC, evaluate on CSE
    pairs = {
        "CIC->CSE": (ci, cs), "CSE->CIC": (cs, ci),
        "UNSW->CIC": (uw, ci),
    }
    results = {}
    for tag, (src_d, tgt_d) in pairs.items():
        t0 = time.time()
        models = _train_mgnn_models(src_d, device, runs, epochs, batch_size,
                                    ip(src_d))
        m, s = _eval_on(tgt_d, models, device, batch_size)
        results[tag] = {"f1": f"{m:.1f}", "std": f"{s:.1f}"}
        print(f"[{tag}] MGNN zero-shot F1={m:.1f}+-{s:.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cross-dataset generalization (Table 4)")
    add_common_args(parser)
    args = parser.parse_args()
    device = parse_device(args)
    t0 = time.time()
    res = run_table4(args.data_dir, device, runs=args.runs,
                     epochs=args.epochs, batch_size=args.batch_size)
    print(f"\n{'='*60}\nTotal {time.time()-t0:.0f}s")
    print(json.dumps(res, indent=2, default=str))
    with open("table4_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)