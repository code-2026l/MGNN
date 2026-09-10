#!/usr/bin/env python3
"""
Table 2: main results on three benchmarks (paper, Table 2 / Sec. 5.2).

Trains the ten baselines (RF, XGBoost, CNN, LSTM, KitNET, GCN, GraphSAGE,
GAT, E-GraphSAGE, GDN) plus MGNN on each dataset under the paper's protocol
and reports Precision / Recall / F1 / FPR / MCC / AUC (mean +- std over 5
runs). MGNN numbers are produced by run_table6.py (attn_align) and read from
its JSON output when present; the baselines are reproduced here.

Usage:
    python experiments/run_table2.py [--data-dir DIR] [--dataset all]
                                     [--runs 5] [--epochs 100] [--models ...]
"""

import argparse
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import add_common_args, parse_device, SEEDS
from src.utils.split import chrono_split, undersample_train
from src.utils.metrics import compute_metrics, format_metrics

ML_MODELS = ["RF", "XGBoost"]
DL_MODELS = ["CNN", "LSTM", "KitNET"]
GNN_MODELS = ["GCN", "GraphSAGE", "GAT", "E-GraphSAGE", "GDN"]
ALL_MODELS = ML_MODELS + DL_MODELS + GNN_MODELS


def run_table2(data_dir, device, runs=5, epochs=100, hidden=128,
               dataset="all", models=None):
    from src.data.dataset import load_pt_data
    from src.data.graph import (build_hetero_graph, flow_flow_graph)
    from src.models import baselines as bl
    from src.utils.training import train_graph_model

    datasets = ["cic_ids2017", "unsw_nb15", "cse_ids2018"] \
        if dataset == "all" else [dataset]
    if models is None:
        models = ALL_MODELS

    gnn_classes = {
        "GCN": bl.GCN, "GraphSAGE": bl.GraphSAGE, "GAT": bl.GATNet,
        "E-GraphSAGE": bl.EGraphSAGE, "GDN": bl.GDN,
    }
    dl_classes = {
        "CNN": bl.CNNNet, "LSTM": bl.LSTMNet, "KitNET": bl.KitNET,
    }
    ml_factories = {
        "RF": lambda random_state: __import__(
            "sklearn.ensemble", fromlist=["RandomForestClassifier"]
        ).RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                 random_state=random_state),
        "XGBoost": lambda random_state: __import__(
            "xgboost", fromlist=["XGBClassifier"]
        ).XGBClassifier(n_estimators=200, learning_rate=0.1, n_jobs=-1,
                        random_state=random_state),
    }

    all_results = {}
    for dname in datasets:
        data = load_pt_data(data_dir, f"{dname}.pt")
        seq, stat, labels = data["seq"], data["stat"], data["labels"]
        ts = data["timestamp"].numpy()
        src_ip = data["src_ip"].numpy()
        dst_ip = data["dst_ip"].numpy()

        train_idx, val_idx, test_idx = chrono_split(ts)
        train_sel = undersample_train(train_idx, labels.numpy(), ratio=3)
        print(f"\n[{dname}] N={len(labels)} train={len(train_sel)} "
              f"val={len(val_idx)} test={len(test_idx)}")

        X = stat.numpy()
        y = labels.numpy()
        X_tr, y_tr = X[train_sel], y[train_sel]
        X_te, y_te = X[test_idx], y[test_idx]

        # Flow-level graph (temporal + volume edges) for the GNN baselines.
        graph = build_hetero_graph(stat.numpy(), labels.numpy(),
                                   src_ip, dst_ip, ts)
        ff = flow_flow_graph(graph)
        n = ff.num_nodes
        pos = {int(i): p for p, i in enumerate(train_idx)}
        ff.train_mask = torch.zeros(n, dtype=torch.bool)
        ff.train_mask[[pos[int(i)] for i in train_sel]] = True
        ff.val_mask = torch.zeros(n, dtype=torch.bool)
        ff.val_mask[val_idx] = True
        ff.test_mask = torch.zeros(n, dtype=torch.bool)
        ff.test_mask[test_idx] = True

        def _loader(idx, use_seq, shuffle):
            if use_seq:
                ds = TensorDataset(seq[torch.as_tensor(idx)],
                                   stat[torch.as_tensor(idx)],
                                   labels[torch.as_tensor(idx)])
            else:
                ds = TensorDataset(stat[torch.as_tensor(idx)],
                                   labels[torch.as_tensor(idx)])
            return DataLoader(ds, batch_size=2048, shuffle=shuffle,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)

        def _run_dl(model_cls, use_seq, use_stat):
            tr = _loader(train_sel, use_seq, True)
            va = _loader(val_idx, use_seq, False)
            te = _loader(test_idx, use_seq, False)
            metrics_l = []
            for seed in SEEDS[:runs]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = model_cls(in_dim=stat.size(-1)).to(device) \
                    if use_stat else model_cls(in_dim=seq.size(-1)).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                                       weight_decay=1e-5)
                best_f1, best_state, bad = -1.0, None, 0
                for _ in range(epochs):
                    model.train()
                    for batch in tr:
                        if use_seq:
                            s, st, lbl = batch
                            s = s.to(device)
                        else:
                            st, lbl = batch
                            s = st.to(device)
                        lbl = lbl.to(device).float()
                        opt.zero_grad()
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            model(s), lbl)
                        loss.backward()
                        opt.step()
                    model.eval()
                    preds, scores, ys = [], [], []
                    with torch.no_grad():
                        for batch in va:
                            s, lbl = batch if not use_seq else (batch[0],
                                                                batch[-1])
                            s = s.to(device)
                            sc = torch.sigmoid(model(s))
                            preds.extend((sc > 0.5).cpu().numpy())
                            scores.extend(sc.float().cpu().numpy())
                            ys.extend(lbl.numpy())
                    vm = compute_metrics(np.array(ys), np.array(preds),
                                         np.array(scores))
                    if vm["f1"] > best_f1:
                        best_f1 = vm["f1"]
                        best_state = {k: v.clone()
                                      for k, v in model.state_dict().items()}
                        bad = 0
                    else:
                        bad += 1
                    if bad >= 10:
                        break
                model.load_state_dict(best_state)
                preds, scores, ys = [], [], []
                model.eval()
                with torch.no_grad():
                    for batch in te:
                        s, lbl = batch if not use_seq else (batch[0],
                                                            batch[-1])
                        s = s.to(device)
                        sc = torch.sigmoid(model(s))
                        preds.extend((sc > 0.5).cpu().numpy())
                        scores.extend(sc.float().cpu().numpy())
                        ys.extend(lbl.numpy())
                metrics_l.append(compute_metrics(
                    np.array(ys), np.array(preds), np.array(scores)))
            return metrics_l

        def _run_ml(clf_factory):
            metrics_l = []
            for seed in SEEDS[:runs]:
                clf = clf_factory(random_state=seed)
                clf.fit(X_tr, y_tr)
                scores = clf.predict_proba(X_te)[:, 1]
                preds = (scores > 0.5).astype(int)
                metrics_l.append(compute_metrics(y_te, preds, scores))
            return metrics_l

        ds_results = {}
        for m in models:
            t0 = time.time()
            try:
                if m in ml_factories:
                    ml = _run_ml(ml_factories[m])
                elif m in dl_classes:
                    ml = _run_dl(dl_classes[m], use_seq=(m != "KitNET"),
                                 use_stat=(m == "KitNET"))
                elif m in gnn_classes:
                    ml = train_graph_model(gnn_classes[m], ff, device,
                                           runs=runs, epochs=epochs,
                                           lr=1e-3, hidden=hidden)
                else:
                    continue
                fmt = format_metrics(ml)
                ds_results[m] = fmt
                print(f"  {m:10s} F1={fmt['f1']:>6} FPR={fmt['fpr']:>5} "
                      f"MCC={fmt['mcc']:>5} ({time.time()-t0:.0f}s)",
                      flush=True)
            except Exception as e:
                print(f"  {m} FAILED: {e}")
                ds_results[m] = {"error": str(e)}
        all_results[dname] = ds_results
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main results (paper, Table 2)")
    add_common_args(parser)
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "cic_ids2017", "unsw_nb15",
                                 "cse_ids2018"])
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated models (default: all ten).")
    args = parser.parse_args()
    device = parse_device(args)
    models = None if args.models is None else \
        [m.strip() for m in args.models.split(",")]
    t0 = time.time()
    res = run_table2(args.data_dir, device, runs=args.runs,
                     epochs=args.epochs, hidden=args.hidden,
                     dataset=args.dataset, models=models)
    print(f"\n{'='*60}\nTotal {time.time()-t0:.0f}s")
    print(json.dumps(res, indent=2, default=str))
    with open(f"table2_{args.dataset}_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)