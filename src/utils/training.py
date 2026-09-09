"""
Training and evaluation utilities for MGNN and baseline models.
"""

import contextlib
import torch
import torch.nn.functional as F
import numpy as np
from src.utils.metrics import compute_metrics


def train_epoch_mgnn(model, loader, optimizer, device, align=False,
                     lambda_align=0.1, lambda_reg=0.3, amp=True):
    """Train MGNN for one epoch.

    Implements the paper's total loss (Eq. (8)):
        L = L_BCE + lambda_align * L_align + lambda_reg * L_reg
    where L_align encourages inter-view consistency (only for the 'attn_align'
    fusion) and L_reg penalizes decision-boundary curvature (smoothness).

    A single encoder pass produces the classifier logits, the fused
    pre-classifier representation (for L_reg), and the three view embeddings
    (for L_align) together, so the encoders are not recomputed.

    Args:
        model: MGNN model instance.
        loader: DataLoader yielding (seq, stat, labels).
        optimizer: PyTorch optimizer.
        device: torch device.
        align: Whether to apply the alignment loss L_align.
        lambda_align: Weight of the alignment loss (paper: 0.1).
        lambda_reg: Weight of the decision-boundary regularization (paper: 0.3).
        amp: Use bf16 automatic mixed precision (near-lossless on A100 and
            roughly 2x faster through the tensor cores). Pass False for an
            exact IEEE-fp32 run.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if amp else contextlib.nullcontext())
    for seq, stat, lbl in loader:
        seq = seq.to(device, non_blocking=True)
        stat = stat.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True).float()
        optimizer.zero_grad()
        with autocast:
            logits, h, views = model.forward_all(seq, stat)
            loss = F.binary_cross_entropy_with_logits(logits, lbl)

            if lambda_reg > 0:
                # L_reg: penalize the local curvature of the decision function.
                # Measured as the squared gradient norm of the (squared) logit
                # w.r.t. the fused pre-classifier representation h. This flattens
                # the decision boundary and widens the margin
                # (paper: margin 0.42 vs 0.31).
                hg = h.detach().requires_grad_(True)
                cur_logit = model.classifier(hg)
                g = torch.autograd.grad(
                    cur_logit.pow(2).mean(), hg, retain_graph=True)[0]
                loss = loss + lambda_reg * g.pow(2).sum(-1).mean()

            if align:
                vn = F.normalize(views, dim=-1)
                align_loss = 0.0
                for i in range(3):
                    for j in range(i + 1, 3):
                        align_loss += (vn[:, i] - vn[:, j]).pow(2).sum(-1).mean()
                loss = loss + lambda_align * align_loss / 3

        loss.backward()
        optimizer.step()
        total_loss += loss.detach().float().item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_mgnn(loader, model, device):
    """Evaluate MGNN on a dataloader.

    Returns:
        dict with precision, recall, f1, fpr, mcc, auc.
    """
    model.eval()
    all_preds, all_labels, all_scores = [], [], []
    use_bf16 = (device.type == "cuda")
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if use_bf16 else contextlib.nullcontext())
    for seq, stat, lbl in loader:
        seq, stat = seq.to(device, non_blocking=True), \
            stat.to(device, non_blocking=True)
        with autocast:
            logits = model(seq, stat)
            scores = torch.sigmoid(logits)
        preds = (scores > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbl.numpy())
        all_scores.extend(scores.float().cpu().numpy())
    return compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_scores))


def train_graph_model(model_cls, data, device, runs=5, epochs=100, lr=1e-3,
                      hidden=128):
    """Train a graph model (GCN/SAGE/etc.) over multiple runs.

    Args:
        model_cls: Model class with constructor (in_dim, hidden, out_dim).
        data: torch_geometric Data object with train/val/test masks
              (or the equivalent dict produced by src.data.graph).
        device: torch device.
        runs: Number of repeated runs with different seeds.
        epochs: Training epochs per run.
        lr: Learning rate.
        hidden: Hidden dimension for the model.

    Returns:
        List of metric dicts, one per run.
    """
    from src.utils.config import SEEDS
    results = []
    y = data["y"] if isinstance(data, dict) else data.y
    x = data["x"] if isinstance(data, dict) else data.x
    for seed in SEEDS[:runs]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_cls(x.size(-1), hidden=hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        data_dev = data.to(device)

        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            out = model(data_dev)
            loss = F.binary_cross_entropy_with_logits(
                out[data_dev.train_mask], data_dev.y[data_dev.train_mask].float())
            loss.backward()
            opt.step()

            if epoch % 20 == 19:
                model.eval()
                with torch.no_grad():
                    val_out = torch.sigmoid(model(data_dev)[data_dev.val_mask])
                    val_pred = (val_out > 0.5).float()
                    from sklearn.metrics import f1_score as sk_f1
                    val_f1 = sk_f1(
                        data_dev.y[data_dev.val_mask].cpu(), val_pred.cpu(),
                        zero_division=0)

        # Final test evaluation
        model.eval()
        with torch.no_grad():
            test_logits = model(data_dev)[data_dev.test_mask]
            test_scores = torch.sigmoid(test_logits)
            test_pred = (test_scores > 0.5).float().cpu().numpy()
            test_true = data_dev.y[data_dev.test_mask].cpu().numpy()

        metrics = compute_metrics(test_true, test_pred, test_scores.cpu().numpy())
        results.append(metrics)
    return results
