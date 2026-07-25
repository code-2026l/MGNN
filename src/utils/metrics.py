"""
Evaluation metrics for binary anomaly detection.

Computes: Precision, Recall, F1, FPR, MCC, AUC.
"""

import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef)


def compute_metrics(y_true, y_pred, y_scores=None):
    """Compute all classification metrics.

    Args:
        y_true: Ground-truth labels (0 = benign, 1 = anomaly).
        y_pred: Binary predictions.
        y_scores: Raw prediction scores (for AUC). Optional.

    Returns:
        dict with precision, recall, f1, fpr, mcc, auc keys.
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    results = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'fpr': fp / max(tn + fp, 1),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }

    if y_scores is not None and len(np.unique(y_true)) > 1:
        results['auc'] = roc_auc_score(y_true, y_scores)
    else:
        results['auc'] = 0.0

    return results


def summarize_metrics(metrics_list):
    """Average metrics across multiple runs with std.

    Args:
        metrics_list: List of dicts from compute_metrics().

    Returns:
        dict with 'mean' and 'std' sub-dicts.
    """
    keys = ['precision', 'recall', 'f1', 'fpr', 'mcc', 'auc']
    means = {}
    stds = {}
    for k in keys:
        vals = [m[k] * 100 for m in metrics_list]
        means[k] = np.mean(vals)
        stds[k] = np.std(vals)
    return {'mean': means, 'std': stds}


def format_metrics(metrics_list):
    """Format metrics for table insertion (mean +/- std as string)."""
    summary = summarize_metrics(metrics_list)
    formatted = {}
    for k in summary['mean']:
        formatted[k] = (f"{summary['mean'][k]:.1f}$\\pm${summary['std'][k]:.1f}")
    return formatted
