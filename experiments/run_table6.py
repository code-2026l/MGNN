#!/usr/bin/env python3
"""
Table 6: MGNN fusion strategy comparison.

Compares four fusion strategies (concat, avg, attn, attn_align)
on sequence+stat view data. Reports F1 and FPR.

Usage:
    python experiments/run_table6.py [--data-dir PATH] [--runs 5] [--epochs 30]
"""

import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.models.mgnn import MGNN
from src.utils.config import add_common_args, parse_device, SEEDS
from src.utils.training import train_epoch_mgnn, evaluate_mgnn
from src.utils.metrics import format_metrics


def run_table6(data_dir, device, runs=5, epochs=30, batch_size=256, hidden=128):
    """Run fusion strategy comparison.

    Uses preprocessed .pt data if available, otherwise falls back
    to synthetic data.
    """
    from src.data.download import load_pt_data
    from src.data.synthetic import make_synthetic_seqstat_data

    results = {}
    dataset_names = ['cic_ids2017', 'unsw_nb15', 'cse_ids2018']

    for ds_name in dataset_names:
        pt_name = f'{ds_name}.pt'
        data = load_pt_data(data_dir, pt_name)
        if data is None:
            print(f'\n  No preprocessed data for {ds_name}, using synthetic fallback.')
            ds = make_synthetic_seqstat_data(n_samples=10000, seed=42)
        else:
            seq, stat, labels = data['seq'], data['stat'], data['labels']
            ds = torch.utils.data.TensorDataset(seq, stat, labels)

        n = len(ds)
        n_train = int(0.6 * n)
        n_test = n - n_train
        train_ds, test_ds = random_split(ds, [n_train, n_test])
        seq_len = ds[0][0].size(0)
        stat_dim = ds[0][1].size(0)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        ds_results = {}
        for fusion in ['concat', 'avg', 'attn', 'attn_align']:
            print(f'\n  [{ds_name}] Fusion: {fusion}', flush=True)
            metrics_list = []
            t0 = time.time()
            for seed in SEEDS[:runs]:
                torch.manual_seed(seed)
                model = MGNN(fusion=fusion, seq_len=seq_len, stat_dim=stat_dim,
                             hidden=hidden, use_gat=True).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                for _ in range(epochs):
                    train_epoch_mgnn(model, train_loader, opt, device,
                                     align=(fusion == 'attn_align'))
                metrics_list.append(evaluate_mgnn(test_loader, model, device))

            fmt = format_metrics(metrics_list)
            ds_results[fusion] = fmt
            print(f'    F1={fmt["f1"]}  FPR={fmt["fpr"]}  '
                  f'P={fmt["precision"]}  R={fmt["recall"]}  '
                  f'({time.time()-t0:.0f}s)')

        results[ds_name] = ds_results

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Table 6: Fusion strategy comparison')
    add_common_args(parser)
    args = parser.parse_args()
    device = parse_device(args)
    print(f'Device: {device}')
    print(f'Data dir: {args.data_dir}')
    print(f'Runs: {args.runs}, Epochs: {args.epochs}, Batch: {args.batch_size}')

    t0 = time.time()
    results = run_table6(args.data_dir, device, runs=args.runs,
                         epochs=args.epochs, batch_size=args.batch_size,
                         hidden=args.hidden)
    elapsed = time.time() - t0

    print(f'\n{"=" * 60}')
    print(f'Total time: {elapsed:.0f}s')
    print(json.dumps(results, indent=2))
