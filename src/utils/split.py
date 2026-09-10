"""
Data split helpers shared by all experiment scripts (paper, Sec. 5.1).
"""

import numpy as np


def chrono_split(timestamps, fracs=(0.6, 0.2, 0.2)):
    """Chronological split (paper, Sec. 5.1).

    The first ``fracs[0]`` of flows (by arrival time) form the training set,
    the next ``fracs[1]`` the validation set and the last ``fracs[2]`` the
    test set, preserving temporal ordering so future flows never influence
    predictions on past ones. Returns index arrays (order-preserving).
    """
    order = np.argsort(timestamps, kind="stable")
    n = len(order)
    n_tr = int(n * fracs[0])
    n_va = int(n * fracs[1])
    return (order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:])


def undersample_train(train_idx, labels, ratio=3):
    """Undersample benign flows to a 1:ratio anomaly:benign ratio.

    Applied to the training split only; the validation/test sets retain the
    original class distribution (paper, Sec. 5.1). Deterministic.
    """
    y = labels[train_idx]
    anom = y > 0.5
    n_anom = int(anom.sum())
    n_benign_keep = min(int(n_anom * ratio), int((~anom).sum()))
    rng = np.random.RandomState(0)
    benign_ids = np.where(~anom)[0]
    keep = rng.choice(benign_ids, n_benign_keep, replace=False)
    return train_idx[np.concatenate([np.where(anom)[0], np.sort(keep)])]
