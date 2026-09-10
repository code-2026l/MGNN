#!/usr/bin/env python3
"""
End-to-end real-data pipeline.

  1) Download + preprocess the three benchmark datasets
  2) Run Table 6 (fusion comparisons) and Table 8 (SOTA comparison)

Usage:
    python experiments/run_pipeline.py [--data-dir data]
    python experiments/run_pipeline.py --smoke     # tiny cap, fast check
Requires --prepare first (download/preprocess) unless data/*.pt already exist.
"""

import argparse
import json
import os
import time

from src.utils.config import add_common_args, parse_device


def main():
    parser = argparse.ArgumentParser("MGNN real-data pipeline")
    add_common_args(parser)
    parser.add_argument("--prepare", action="store_true",
                        help="download + preprocess all datasets first")
    parser.add_argument("--smoke", action="store_true",
                        help="use a small n_samples for a fast smoke test")
    parser.add_argument("--table3", action="store_true", default=True,
                        help="run SOTA comparison (default on)")
    parser.add_argument("--table6", action="store_true", default=True,
                        help="run fusion comparison (default on)")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True,
                        help="use bf16 automatic mixed precision (default on)")
    args = parser.parse_args()
    device = parse_device(args)
    os.makedirs(args.data_dir, exist_ok=True)
    cap = 50000 if args.smoke else None

    if args.prepare or not os.path.exists(
            os.path.join(args.data_dir, "cic_ids2017.pt")):
        from src.data.dataset import (prepare_cic_ids2017, prepare_unsw_nb15,
                                      prepare_cse_ids2018)
        prepare_cic_ids2017(args.data_dir, n_samples=cap, download=True)
        prepare_unsw_nb15(args.data_dir, n_samples=cap, download=True)
        prepare_cse_ids2018(args.data_dir, n_samples=cap, download=True)

    results = {}
    t_start = time.time()

    if args.table6:
        from experiments.run_table6 import run_table6
        results["table6"] = run_table6(
            args.data_dir, device, runs=args.runs, epochs=args.epochs,
            batch_size=args.batch_size, hidden=args.hidden, amp=args.amp)
    if args.table3:
        from experiments.run_table3 import run_table3
        # Paper, Table 8: SOTA comparison on CIC-IDS2017 (70/30 split).
        results["table3"] = run_table3(
            args.data_dir, "cic_ids2017", device, runs=args.runs,
            epochs=args.epochs, hidden=args.hidden)

    print(f"\nPipeline done in {time.time()-t_start:.0f}s")
    print(json.dumps(results, indent=2, default=str))
    with open(os.path.join(args.data_dir, "pipeline_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
