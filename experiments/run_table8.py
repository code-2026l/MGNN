#!/usr/bin/env python3
"""
Table 8: comparison with state-of-the-art methods (paper, Table 8 / Sec. 5.7).

Reproduces the paper's SOTA comparison on CIC-IDS2017 with a chronological
70/30 split: traditional ML (Random Forest, XGBoost, LightGBM), deep
learning (LSTM, CNN-LSTM, DNN, BiLSTM) and graph methods (GAT, GCN,
E-GraphSAGE, FN-GNN, GDN). MGNN itself is produced by run_table6.py.

Usage:
    python experiments/run_table8.py [--data-dir DIR] [--dataset cic_ids2017]
                                     [--runs 5] [--epochs 30]
                                     [--models GCN,GDN,...]
"""

import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import add_common_args, parse_device, SEEDS
from src.utils.split import chrono_split, undersample_train

ML_MODELS = ["RF", "XGBoost", "LightGBM"]
DL_MODELS = ["LSTM", "CNN-LSTM", "DNN", "BiLSTM"]
GNN_MODELS = ["GAT", "GCN", "E-GraphSAGE", "FN-GNN", "GDN"]
ALL_MODELS = ML_MODELS + DL_MODELS + GNN_MODELS


def run_table8(data_dir, dataset, device, runs=5, epochs=30, hidden=128,
               models=None):
    from src.data.dataset import load_pt_data
    from src.data.graph import (build_hetero_graph, flow_flow_graph)
    from src.models import baselines as bl
    from src.utils.training import train_graph_model
    from src.utils.metrics import compute_metrics, format_metrics

    data = load_pt_data(data_dir, f"{dataset}.pt")
    seq, stat, labels = data["seq"], data["stat"], data["labels"]
    ts = data["timestamp"].numpy()
    src_ip = data["src_ip"].numpy()
    dst_ip = data["dst_ip"].numpy()

    train_idx, val_idx, test_idx = chrono_split(ts, fracs=(0.56, 0.14, 0.30))
    train_sel = undersample_train(train_idx, labels.numpy())
    print(f"\n[{dataset}] N={len(labels)} train={len(train_sel)} "
          f"val={len(val_idx)} test={len(test_idx)} "
          f"anom(test)={labels[test_idx].mean().item()*100:.1f}%")

    X = stat.numpy()
    y = labels.numpy()
    X_tr, y_tr = X[train_sel], y[train_sel]
    X_va, y_va = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    # Flow-level graph (temporal + volume edges) for the GNN baselines.
    graph = build_hetero_graph(stat.numpy(), labels.numpy(), src_ip, dst_ip, ts)
    ff = flow_flow_graph(graph)
    n = ff.num_nodes
    pos = {int(i): p for p, i in enumerate(train_idx)}
    ff.train_mask = torch.zeros(n, dtype=torch.bool)
    ff.train_mask[[pos[int(i)] for i in train_sel]] = True
    ff.val_mask = torch.zeros(n, dtype=torch.bool)
    ff.val_mask[val_idx] = True
    ff.test_mask = torch.zeros(n, dtype=torch.bool)
    ff.test_mask[test_idx] = True
    print(f"  flow graph: {n} nodes, {ff.edge_index.size(1)} edges")

    if models is None:
        models = ALL_MODELS

    def _loader(idx, use_seq, shuffle):
        if use_seq:
            ds = TensorDataset(seq[torch.as_tensor(idx)], stat[torch.as_tensor(idx)],
                               labels[torch.as_tensor(idx)])
        else:
            ds = TensorDataset(stat[torch.as_tensor(idx)],
                               labels[torch.as_tensor(idx)])
        return DataLoader(ds, batch_size=2048, shuffle=shuffle, num_workers=4,
                          pin_memory=True, persistent_workers=True)

    def _run_dl(model_cls, use_seq):
        tr_loader = _loader(train_sel, use_seq, True)
        va_loader = _loader(val_idx, use_seq, False)
        te_loader = _loader(test_idx, use_seq, False)
        metrics_list = []
        for seed in SEEDS[:runs]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = model_cls(in_dim=seq.size(-1), hidden=hidden).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                                   weight_decay=1e-5)
            best_f1, best_state, bad = -1.0, None, 0
            for _ in range(epochs):
                model.train()
                for batch in tr_loader:
                    if use_seq:
                        s, st, lbl = batch
                        s, lbl = s.to(device), lbl.to(device).float()
                    else:
                        st, lbl = batch
                        s, lbl = st.to(device), lbl.to(device).float()
                    opt.zero_grad()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        model(s if use_seq else st), lbl)
                    loss.backward()
                    opt.step()
                model.eval()
                preds, scores, ys = [], [], []
                with torch.no_grad():
                    for batch in va_loader:
                        if use_seq:
                            s, st, lbl = batch
                            s = s.to(device)
                        else:
                            st, lbl = batch
                            s = st.to(device)
                        lg = model(s)
                        sc = torch.sigmoid(lg)
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
                for batch in te_loader:
                    if use_seq:
                        s, st, lbl = batch
                        s = s.to(device)
                    else:
                        st, lbl = batch
                        s = st.to(device)
                    sc = torch.sigmoid(model(s))
                    preds.extend((sc > 0.5).cpu().numpy())
                    scores.extend(sc.float().cpu().numpy())
                    ys.extend(lbl.numpy())
            metrics_list.append(compute_metrics(
                np.array(ys), np.array(preds), np.array(scores)))
        return metrics_list

    def _run_ml(clf_factory):
        metrics_list = []
        for seed in SEEDS[:runs]:
            rng = np.random.RandomState(seed)
            clf = clf_factory(random_state=seed)
            clf.fit(X_tr, y_tr)
            scores = clf.predict_proba(X_te)[:, 1]
            preds = (scores > 0.5).astype(int)
            metrics_list.append(compute_metrics(y_te, preds, scores))
        return metrics_list

    gnn_classes = {
        "GCN": bl.GCN, "GAT": bl.GATNet, "E-GraphSAGE": bl.EGraphSAGE,
        "FN-GNN": bl.FNGNN, "GDN": bl.GDN,
    }
    dl_classes = {
        "LSTM": bl.LSTMNet, "CNN-LSTM": bl.CNNLSTMNet,
        "DNN": bl.DNN, "BiLSTM": bl.BiLSTMNet,
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
        "LightGBM": lambda random_state: __import__(
            "lightgbm", fromlist=["LGBMClassifier"]
        ).LGBMClassifier(n_estimators=200, learning_rate=0.1, n_jobs=-1,
                         random_state=random_state, verbose=-1),
    }

    results = {}
    for mname in models:
        t0 = time.time()
        try:
            if mname in ml_factories:
                metrics_list = _run_ml(ml_factories[mname])
            elif mname in dl_classes:
                metrics_list = _run_dl(dl_classes[mname],
                                       use_seq=mname != "DNN")
            elif mname in gnn_classes:
                metrics_list = train_graph_model(
                    gnn_classes[mname], ff, device, runs=runs, epochs=epochs,
                    lr=1e-3, hidden=hidden)
            else:
                print(f"  unknown model {mname}, skipped")
                continue
            fmt = format_metrics(metrics_list)
            results[mname] = fmt
            print(f"  {mname:10s} F1={fmt['f1']:>6} FPR={fmt['fpr']:>5} "
                  f"P={fmt['precision']:>5} R={fmt['recall']:>5} "
                  f"MCC={fmt['mcc']:>5} AUC={fmt.get('auc', '-'):>5} "
                  f"({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  {mname} FAILED: {e}")
            results[mname] = {"error": str(e)}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SOTA comparison (real data)")
    add_common_args(parser)
    parser.add_argument("--dataset", type=str, default="cic_ids2017",
                        choices=["cic_ids2017", "unsw_nb15", "cse_ids2018"])
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated models to run "
                             "(default: all of "
                             + ",".join(ALL_MODELS) + ").")
    args = parser.parse_args()
    device = parse_device(args)
    models = None if args.models is None else \
        [m.strip() for m in args.models.split(",")]
    t0 = time.time()
    res = run_table8(args.data_dir, args.dataset, device, runs=args.runs,
                     epochs=args.epochs, hidden=args.hidden, models=models)
    print(f"\n{'='*60}\nTotal {time.time()-t0:.0f}s")
    print(json.dumps(res, indent=2, default=str))
    with open(f"table8_{args.dataset}_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
