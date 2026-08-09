#!/usr/bin/env python3
"""
Graph-based model comparison.

Compares GCN, GraphSAGE, and EGraphSAGE on synthetic graph data
(or real graph data if available).

Usage:
    python experiments/run_table3.py [--runs 5] [--epochs 100]
"""

import argparse
import json
import time
import numpy as np
import torch

from src.models.baselines import GCN, GraphSAGE, EGraphSAGE
from src.utils.config import add_common_args, parse_device
from src.utils.training import train_graph_model
from src.utils.metrics import format_metrics


def run_table3(device, runs=5, epochs=100):
    """Run graph model comparison on synthetic data.

    Real graph data requires edge_index construction from features,
    which is dataset-specific. This script uses synthetic graphs
    (k-NN approximation) for model validation.
    """
    from src.data.synthetic import make_synthetic_graph_data

    print('\nNOTE: Using synthetic graph data (5000 nodes, random edges).')
    print('For publication-quality results, replace with real graph data')
    print('constructed via k-NN or domain-specific heuristics.\n')

    dataset_names = ['CIC-IDS2017', 'UNSW-NB15', 'CSE-IDS2018']
    model_classes = {
        'GCN': GCN,
        'GraphSAGE': GraphSAGE,
        'E-GraphSAGE': EGraphSAGE,
    }

    results = {}
    for ds_name in dataset_names:
        data = make_synthetic_graph_data(seed=42)
        results[ds_name] = {}

        for mname, mcls in model_classes.items():
            print(f'  {ds_name} / {mname}...', flush=True)
            metrics_list = train_graph_model(
                mcls, data, device, runs=runs, epochs=epochs)
            fmt = format_metrics(metrics_list)
            results[ds_name][mname] = fmt
            print(f'    F1={fmt["f1"]}  FPR={fmt["fpr"]}  '
                  f'P={fmt["precision"]}  R={fmt["recall"]}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph model comparison')
    add_common_args(parser)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs per run.')
    args = parser.parse_args()
    device = parse_device(args)
    print(f'Device: {device}')
    print(f'Runs: {args.runs}')

    t0 = time.time()
    results = run_table3(device, runs=args.runs, epochs=args.epochs)
    elapsed = time.time() - t0

    print(f'\n{"=" * 60}')
    print(f'Total time: {elapsed:.0f}s')
    print(json.dumps(results, indent=2))
