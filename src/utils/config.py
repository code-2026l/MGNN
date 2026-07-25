"""
Configuration and argument parsing for MGNN experiments.
"""

import argparse
import os
import torch


def get_default_data_dir():
    """Return default data directory."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'data')


def get_device():
    """Detect and return available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


SEEDS = [42, 0, 123, 7, 2024]
BENIGN_RATIO = 0.803
ANOMALY_RATIO = 0.197


def add_common_args(parser):
    """Add common CLI arguments shared across experiment scripts."""
    parser.add_argument('--data-dir', type=str, default=get_default_data_dir(),
                        help='Directory containing dataset files.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device override (e.g. "cuda:0" or "cpu").')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help='Random seeds for repeated runs.')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of repeated runs (uses first N seeds).')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs.')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden dimension.')
    return parser


def parse_device(args):
    """Resolve device from args or auto-detect."""
    if args.device is not None:
        return torch.device(args.device)
    return get_device()
