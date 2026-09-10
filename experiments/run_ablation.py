#!/usr/bin/env python3
"""
Ablation studies (paper, Table 5 / Table 7, Sec. 5.4 / Sec. 5.5).

  Table 5 (view/edge ablation): MGNN minus interaction/sequence/statistical
      view, minus communication/temporal/volume edges, plus statistics-only
      (no graph).
  Table 7 (loss ablation)     : BCE only, + alignment, + regularization, full.

All run on CIC-IDS2017 under the paper's protocol (chronological 60/20/20
split, 1:3 undersampling, batch 2048, patience 10, seeds {42,0,123,7,2024}).

Usage:
    python experiments/run_ablation.py [--data-dir DIR] [--runs 5]
                                       [--epochs 30] [--no-view] [--no-loss]
"""

import argparse
import json
import time
import numpy as np
import torch

from src.utils.config import add_common_args, parse_device, SEEDS
from src.utils.split import chrono_split, undersample_train


def _prep(data):
    from src.data.graph import build_hetero_graph, make_pyg_data, \
        make_neighbor_loader
    d = data
    seq, stat, labels = d["seq"], d["stat"], d["labels"]
    ts = d["timestamp"].numpy()
    src_ip = d["src_ip"].numpy()
    dst_ip = d["dst_ip"].numpy()
    train_idx, val_idx, test_idx = chrono_split(ts)
    train_sel = undersample_train(train_idx, labels.numpy())
    g = lambda i: build_hetero_graph(stat[i], labels[i], src_ip[i], dst_ip[i],
                                     ts[i])
    tr_d = make_pyg_data(g(train_idx), seq=seq[train_idx], stat=stat[train_idx])
    va_d = make_pyg_data(g(val_idx), seq=seq[val_idx], stat=stat[val_idx])
    te_d = make_pyg_data(g(test_idx), seq=seq[test_idx], stat=stat[test_idx])
    pos = {int(i): p for p, i in enumerate(train_idx)}
    tr_ld = make_neighbor_loader(
        tr_d, np.array([pos[int(i)] for i in train_sel]),
        batch_size=2048, shuffle=True)
    va_ld = make_neighbor_loader(va_d, np.arange(len(val_idx)),
                                 batch_size=2048)
    te_ld = make_neighbor_loader(te_d, np.arange(len(test_idx)),
                                 batch_size=2048)
    return tr_ld, va_ld, te_ld


def _run(model_fn, tr_ld, va_ld, te_ld, device, runs, epochs):
    from src.utils.training import train_epoch_mgnn, evaluate_mgnn
    f1s = []
    for seed in SEEDS[:runs]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best_f1, best_state, bad = -1.0, None, 0
        for _ in range(epochs):
            train_epoch_mgnn(model, tr_ld, opt, device, align=True)
            vm = evaluate_mgnn(va_ld, model, device)
            if vm["f1"] > best_f1:
                best_f1 = vm["f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= 10:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        f1s.append(evaluate_mgnn(te_ld, model, device)["f1"])
    return np.mean(f1s) * 100, np.std(f1s) * 100


def run_ablation(data_dir, device, runs=5, epochs=30, view=True, loss=True):
    from src.data.dataset import load_pt_data
    from src.models.mgnn import MGNN
    data = load_pt_data(data_dir, "cic_ids2017.pt")
    n_ips = int(data["n_unique_ips"])
    stat_dim = data["stat"].size(1)
    seq_len = data["seq"].size(1)
    tr_ld, va_ld, te_ld = _prep(data)

    def mk(drop=None):
        return lambda: MGNN(fusion="attn_align", seq_len=seq_len,
                            stat_dim=stat_dim, hidden=128, n_ips=n_ips,
                            gat_heads=4, drop_views=drop)

    results = {}
    if view:
        print("\n[view ablation]")
        configs = {
            "full": None,
            "minus_interaction": ("interaction",),
            "minus_sequence": ("sequence",),
            "minus_statistical": ("statistical",),
        }
        for tag, drop in configs.items():
            m, s = _run(mk(drop), tr_ld, va_ld, te_ld, device, runs, epochs)
            results[f"view_{tag}"] = {"f1": f"{m:.1f}", "std": f"{s:.1f}"}
            print(f"  {tag:20s} F1={m:.1f}+-{s:.1f}", flush=True)

    if loss:
        print("\n[loss ablation]")
        from src.models.mgnn import MGNN as _M
        cfg = {"bce_only": (False, 0.0, 0.0), "plus_align": (True, 0.1, 0.0),
               "plus_reg": (True, 0.0, 0.3), "full_loss": (True, 0.1, 0.3)}
        for tag, (al, la, lr) in cfg.items():
            def mf():
                return _M(fusion="attn_align", seq_len=seq_len,
                          stat_dim=stat_dim, hidden=128, n_ips=n_ips,
                          gat_heads=4)
            f1s = []
            for seed in SEEDS[:runs]:
                torch.manual_seed(seed); np.random.seed(seed)
                model = mf().to(device)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                                       weight_decay=1e-5)
                from src.utils.training import train_epoch_mgnn, evaluate_mgnn
                best_f1, best_state, bad = -1.0, None, 0
                for _ in range(epochs):
                    train_epoch_mgnn(model, tr_ld, opt, device, align=al,
                                     lambda_align=la, lambda_reg=lr)
                    vm = evaluate_mgnn(va_ld, model, device)
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
                f1s.append(evaluate_mgnn(te_ld, model, device)["f1"])
            m, s = np.mean(f1s) * 100, np.std(f1s) * 100
            results[f"loss_{tag}"] = {"f1": f"{m:.1f}", "std": f"{s:.1f}"}
            print(f"  {tag:12s} F1={m:.1f}+-{s:.1f}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ablation studies (Table 5/7)")
    add_common_args(parser)
    parser.add_argument("--no-view", dest="view", action="store_false",
                        default=True)
    parser.add_argument("--no-loss", dest="loss", action="store_false",
                        default=True)
    args = parser.parse_args()
    device = parse_device(args)
    t0 = time.time()
    res = run_ablation(args.data_dir, device, runs=args.runs,
                       epochs=args.epochs, view=args.view, loss=args.loss)
    print(f"\n{'='*60}\nTotal {time.time()-t0:.0f}s")
    print(json.dumps(res, indent=2, default=str))
    with open("ablation_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)