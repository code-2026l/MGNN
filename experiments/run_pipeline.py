#!/usr/bin/env python3
"""
MGNN Full Experiment Pipeline.

End-to-end pipeline:
  1. Download datasets (CIC-IDS2017, UNSW-NB15, CSE-CIC-IDS2018)
  2. Preprocess CSV -> .pt files
  3. Run Table 3 (graph model comparison) and Table 6 (fusion strategy)

Usage:
    python experiments/run_pipeline.py [--data-dir PATH] [--skip-download]
    python experiments/run_pipeline.py --table3 --table6
"""

import argparse
import json
import os
import time
import numpy as np
import torch

from src.utils.config import add_common_args, parse_device


def main():
    parser = argparse.ArgumentParser(description='MGNN Full Experiment Pipeline')
    add_common_args(parser)
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download (use existing files).')
    parser.add_argument('--table3', action='store_true',
                        help='Run Table 3 (graph model comparison).')
    parser.add_argument('--table6', action='store_true',
                        help='Run Table 6 (fusion strategy comparison).')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments.')
    args = parser.parse_args()
    device = parse_device(args)

    print('=' * 60)
    print('MGNN Experiment Pipeline')
    print(f'Device: {device}')
    print(f'Data dir: {args.data_dir}')
    print(f'Seeds: {args.seeds}')
    print('=' * 60)

    os.makedirs(args.data_dir, exist_ok=True)
    all_results = {}
    t_start = time.time()

    # Step 1: Download datasets
    if not args.skip_download:
        from src.data.download import download_cic_ids2017, preprocess_cic_csv
        cic_files = download_cic_ids2017(args.data_dir)
        if cic_files:
            data = preprocess_cic_csv(cic_files, args.data_dir)
            if data is None:
                print('CIC-IDS2017 preprocessing failed, using synthetic fallback.')

    # Step 2: Run Table 6 (fusion strategies)
    if args.table6 or args.all:
        from experiments.run_table6 import run_table6
        print('\n' + '=' * 60)
        print('TABLE 6: Fusion Strategy Comparison')
        print('=' * 60)
        all_results['table6'] = run_table6(
            args.data_dir, device, runs=args.runs,
            epochs=args.epochs, batch_size=args.batch_size,
            hidden=args.hidden)

    # Step 3: Run Table 3 (graph model comparison)
    if args.table3 or args.all:
        from experiments.run_table3 import run_table3
        print('\n' + '=' * 60)
        print('TABLE 3: Graph Model Comparison')
        print('=' * 60)
        all_results['table3'] = run_table3(
            device, runs=args.runs, epochs=args.epochs)

    elapsed = time.time() - t_start
    print(f'\n{"=" * 60}')
    print(f'Pipeline complete. Total time: {elapsed:.0f}s')

    if all_results:
        results_path = os.path.join(args.data_dir, 'experiment_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'Results saved to {results_path}')
        print(json.dumps(all_results, indent=2))


if __name__ == '__main__':
    main()
