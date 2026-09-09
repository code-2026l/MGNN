#!/usr/bin/env python3
"""
Table 3: graph-model comparison on REAL data.

Builds a real k-NN flow graph from the preprocessed features and compares
GCN / GraphSAGE / E-GraphSAGE. Requires a preprocessed .pt file; the graph
is built from actual feature geometry, never from a synthetic generator.

Usage:
    python experiments/run_table3.py [--data-dir DIR] [--dataset cic_ids2017]
                                     [--runs 5] [--epochs 200] [--k 5]
"""

import argparse
import json
import time
import numpy as np
import torch

from src.utils.config import add_common_args, parse_device

GRAPH_MODELS = {"GCN", "GraphSAGE", "E-GraphSAGE"}


def run_table3(data_dir, dataset, device, runs=5, epochs=200, hidden=128,
               k=5):
    from src.data.dataset import load_pt_data
    from src.data.graph import build_graph

    data = load_pt_data(data_dir, f"{dataset}.pt")
    print(f"\nBuilding real {k}-NN graph on {dataset} "
          f"(N={data['features'].size(0)})...")
    graph = build_graph(data["features"].numpy(), data["labels"].numpy(),
                        k=k, seed=42)
    if not isinstance(graph, dict) and hasattr(graph, "edge_index"):
        n_edges = graph.edge_index.size(1)
    else:
        n_edges = graph["edge_index"].size(1)
    print(f"  graph: {data['features'].size(0)} nodes, {n_edges} edges")
    print(f"  train={graph['train_mask'].sum()}, "
          f"val={graph['val_mask'].sum()}, "
          f"test={graph['test_mask'].sum()}")

    from src.models.baselines import GCN, GraphSAGE, EGraphSAGE
    from src.utils.training import train_graph_model
    from src.utils.metrics import format_metrics

    classes = {"GCN": GCN, "GraphSAGE": GraphSAGE, "E-GraphSAGE": EGraphSAGE}
    results = {}
    for mname in sorted(GRAPH_MODELS):
        t0 = time.time()
        try:
            metrics = train_graph_model(classes[mname], graph, device,
                                        runs=runs, epochs=epochs,
                                        lr=1e-3, hidden=hidden)
            fmt = format_metrics(metrics)
            results[mname] = fmt
            print(f"  {mname:10s} F1={fmt['f1']:>6} FPR={fmt['fpr']:>5} "
                  f"P={fmt['precision']:>5} R={fmt['recall']:>5} "
                  f"({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  {mname} FAILED: {e}")
            results[mname] = {"error": str(e)}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Graph model comparison (real data)")
    add_common_args(parser)
    parser.add_argument("--dataset", type=str, default="cic_ids2017",
                        choices=["cic_ids2017", "unsw_nb15", "cse_ids2018"])
    parser.add_argument("--k", type=int, default=5,
                        help="k-NN graph degree")
    args = parser.parse_args()
    device = parse_device(args)
    t0 = time.time()
    res = run_table3(args.data_dir, args.dataset, device, runs=args.runs,
                     epochs=args.epochs, hidden=args.hidden, k=args.k)
    print(f"\n{'='*60}\nTotal {time.time()-t0:.0f}s")
    print(json.dumps(res, indent=2, default=str))
    with open(f"table3_{args.dataset}_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)