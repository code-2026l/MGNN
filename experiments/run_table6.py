#!/usr/bin/env python3
"""
Table 6: MGNN fusion-strategy comparison (paper, Table 6 / Sec. 5.4).

Compares the four fusion strategies -- concat, avg, attention (attn),
attention with alignment (attn_align) -- on the real benchmark datasets,
following the paper's protocol exactly:

  - chronological 60/20/20 split (first 60% of flows by time for training,
    20% validation, 20% testing; no cross-split graph edges);
  - benign flows undersampled to a 1:3 anomaly:benign ratio in training;
  - Adam (lr=1e-3, weight_decay=1e-5), batch size 2048, early stopping
    with patience 10, 5 seeds {42,0,123,7,2024};
  - heterogeneous flow/IP graph with temporal / volume / communication
    edges, all thresholds data-driven.

Usage:
    python experiments/run_table6.py [--data-dir DIR] [--runs 5]
                                     [--epochs 30] [--datasets ...] [--no-amp]
"""

import argparse
import json
import time
import numpy as np
import torch

from src.utils.config import add_common_args, parse_device, SEEDS
from src.utils.split import chrono_split, undersample_train


def run_table6(data_dir, device, runs=5, epochs=30, batch_size=2048,
               hidden=128, datasets=("cic_ids2017", "unsw_nb15",
                                     "cse_ids2018"),
               amp=True):
    """Table-6 fusion comparison on real data with the paper's protocol."""
    torch.backends.cudnn.benchmark = True
    from src.data.dataset import load_pt_data
    from src.data.graph import (build_hetero_graph, make_pyg_data,
                                make_neighbor_loader)
    from src.models.mgnn import MGNN
    from src.utils.training import train_epoch_mgnn, evaluate_mgnn
    from src.utils.metrics import format_metrics

    results = {}
    for ds_name in datasets:
        data = load_pt_data(data_dir, f"{ds_name}.pt")
        seq, stat, labels = data["seq"], data["stat"], data["labels"]
        ts = data["timestamp"].numpy()
        src_ip = data["src_ip"].numpy()
        dst_ip = data["dst_ip"].numpy()
        n_ips_global = int(data["n_unique_ips"])
        n = len(labels)

        train_idx, val_idx, test_idx = chrono_split(ts)
        train_sel = undersample_train(train_idx, labels.numpy(), ratio=3)
        print(f"\n[{ds_name}] N={n} ips={n_ips_global} "
              f"train={len(train_idx)} (u/samp={len(train_sel)}) "
              f"val={len(val_idx)} test={len(test_idx)} "
              f"anom(train)={labels[train_sel].mean().item()*100:.1f}% "
              f"anom(test)={labels[test_idx].mean().item()*100:.1f}%")

        # Per-split heterogeneous graphs (edges stay within each split).
        t0 = time.time()
        train_graph = build_hetero_graph(stat[train_idx], labels[train_idx],
                                         src_ip[train_idx], dst_ip[train_idx],
                                         ts[train_idx])
        val_graph = build_hetero_graph(stat[val_idx], labels[val_idx],
                                       src_ip[val_idx], dst_ip[val_idx],
                                       ts[val_idx])
        test_graph = build_hetero_graph(stat[test_idx], labels[test_idx],
                                        src_ip[test_idx], dst_ip[test_idx],
                                        ts[test_idx])
        print(f"  graphs: train {train_graph['n_edges']} edges, "
              f"val {val_graph['n_edges']}, test {test_graph['n_edges']} "
              f"({time.time()-t0:.0f}s)")

        train_data = make_pyg_data(train_graph, seq=seq[train_idx],
                                   stat=stat[train_idx])
        val_data = make_pyg_data(val_graph, seq=seq[val_idx],
                                 stat=stat[val_idx])
        test_data = make_pyg_data(test_graph, seq=seq[test_idx],
                                  stat=stat[test_idx])

        # Map the undersampled flows to positions inside the train subgraph.
        pos = {int(i): p for p, i in enumerate(train_idx)}
        train_seeds = np.array([pos[int(i)] for i in train_sel])
        val_seeds = np.arange(len(val_idx))
        test_seeds = np.arange(len(test_idx))

        train_loader = make_neighbor_loader(train_data, train_seeds,
                                            batch_size=batch_size,
                                            shuffle=True)
        val_loader = make_neighbor_loader(val_data, val_seeds,
                                          batch_size=batch_size)
        test_loader = make_neighbor_loader(test_data, test_seeds,
                                           batch_size=batch_size)

        ds_results = {}
        for fusion in ["concat", "avg", "attn", "attn_align"]:
            t0 = time.time()
            metrics_list = []
            for seed in SEEDS[:runs]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = MGNN(fusion=fusion, seq_len=seq.size(1),
                             stat_dim=stat.size(1), hidden=hidden,
                             n_ips=n_ips_global, gat_heads=4).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                                       weight_decay=1e-5)
                best_f1, best_state, bad = -1.0, None, 0
                for ep in range(epochs):
                    _ = train_epoch_mgnn(
                        model, train_loader, opt, device,
                        align=(fusion == "attn_align"), amp=amp)
                    vm = evaluate_mgnn(val_loader, model, device)
                    if vm["f1"] > best_f1:
                        best_f1 = vm["f1"]
                        best_state = {k: v.clone()
                                      for k, v in model.state_dict().items()}
                        bad = 0
                    else:
                        bad += 1
                    print(f"    [r{seed}] ep{ep+1}/{epochs} "
                          f"bestF1={best_f1:.4f}", flush=True)
                    if bad >= 10:  # early stopping, patience 10
                        break
                if best_state is not None:
                    model.load_state_dict(best_state)
                metrics_list.append(evaluate_mgnn(test_loader, model, device))
            fmt = format_metrics(metrics_list)
            ds_results[fusion] = fmt
            print(f"  {fusion:10s} F1={fmt['f1']:>6} FPR={fmt['fpr']:>5} "
                  f"P={fmt['precision']:>5} R={fmt['recall']:>5} "
                  f"AUC={fmt.get('auc', '-'):>5} ({time.time()-t0:.0f}s)")
        results[ds_name] = ds_results
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fusion strategy comparison (real data)")
    add_common_args(parser)
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names to run "
                             "(default: cic_ids2017,unsw_nb15,cse_ids2018).")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        default=True,
                        help="Disable bf16 automatic mixed precision.")
    args = parser.parse_args()
    device = parse_device(args)
    datasets = None if args.datasets is None else \
        tuple(d.strip() for d in args.datasets.split(","))
    t0 = time.time()
    res = run_table6(args.data_dir, device, runs=args.runs,
                     epochs=args.epochs, batch_size=args.batch_size,
                     hidden=args.hidden, datasets=datasets, amp=args.amp)
    print(f"\n{'='*60}\nTotal {time.time()-t0:.0f}s")
    print(json.dumps(res, indent=2, default=str))
    with open("table6_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
