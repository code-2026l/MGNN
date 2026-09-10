#!/usr/bin/env python3
"""A100 kernel-config benchmark for the MGNN training step.

Times a single fused fwd+bwd on a synthetic heterogeneous-graph batch shaped
exactly like the real CIC/UNSW data, across a matrix of CUDA configurations,
and reports the fastest. Determines whether torch.compile (static/dynamic)
and TF32 help on this Ampere box, so the paper run can pick the right lever
set offline. bf16 automatic mixed precision is always on (tensor cores).
"""

import argparse
import time
import torch
import torch.nn.functional as F

from src.models.mgnn import MGNN


def _timed_cfg(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main(batch=2048, seq_len=100, stat_dim=23, hidden=128, epochs=30,
         n_ips=50000):
    device = torch.device("cuda")
    torch.manual_seed(0)

    n_flow = batch
    n_total = n_flow + n_ips
    seq = torch.randn(n_total, seq_len, 3, device=device)
    stat = torch.randn(n_total, stat_dim, device=device)
    x_all = torch.randn(n_total, stat_dim, device=device)
    node_type = torch.zeros(n_total, dtype=torch.long, device=device)
    node_type[n_flow:] = 1
    ip_index = torch.full((n_total,), -1, dtype=torch.long, device=device)
    ip_index[n_flow:] = torch.arange(n_ips, device=device)
    flow_mask = node_type == 0
    edge_index = torch.randint(0, n_total, (2, 5 * n_total),
                               device=device)
    edge_index[0] = torch.randint(0, n_flow, edge_index[0].size(),
                                  device=device)
    lbl = (torch.rand(n_flow, device=device) > 0.5).float()

    cfgs = {
        "bf16 only (baseline)":        dict(tf32=False, mode=None),
        "bf16 + TF32":                 dict(tf32=True,  mode=None),
        "bf16 + TF32 + compile(static)": dict(tf32=True, mode="static"),
        "bf16 + TF32 + compile(dynamic)": dict(tf32=True, mode="dynamic"),
    }
    results = {}
    for name, c in cfgs.items():
        torch.backends.cuda.matmul.allow_tf32 = c.get("tf32", False)
        torch.backends.cudnn.allow_tf32 = c.get("tf32", False)
        model = MGNN(fusion="attn_align", seq_len=seq_len, stat_dim=stat_dim,
                     hidden=hidden, n_ips=n_ips, gat_heads=4).to(device)
        if c.get("mode"):
            try:
                model = torch.compile(model, dynamic=(c["mode"] == "dynamic"),
                                      fullgraph=(c["mode"] == "static"),
                                      mode="max-autotune-no-cudagraphs")
            except Exception as e:
                print(f"[{name}] compile failed: {e}")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        def step():
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, h, views = model.forward_all(
                    seq[flow_mask], stat[flow_mask], x_all, edge_index,
                    node_type, ip_index, flow_mask)
                logits = logits[:n_flow]
                loss = F.binary_cross_entropy_with_logits(logits, lbl)
                # L_reg + L_align as in training.py
                hg = h[:n_flow].detach().requires_grad_(True)
                cl = model.classifier(hg)
                g = torch.autograd.grad(cl.pow(2).mean(), hg,
                                        retain_graph=True)[0]
                loss = loss + 0.3 * g.pow(2).sum(-1).mean()
                vn = F.normalize(views[:n_flow], dim=-1)
                al = (vn[:, 0] - vn[:, 1]).pow(2).sum(-1).mean() + \
                     (vn[:, 0] - vn[:, 2]).pow(2).sum(-1).mean() + \
                     (vn[:, 1] - vn[:, 2]).pow(2).sum(-1).mean()
                loss = loss + 0.1 * al
            loss.backward()
            opt.step()

        try:
            dt = _timed_cfg(step)
            results[name] = dt
            print(f"{name:38s} {dt*1000:7.1f} ms/step "
                  f"({(batch / dt) / 1e6:.2f} M-flows/s, "
                  f"~{dt*epochs/60:.1f} min/epoch")
        except Exception as e:
            print(f"{name:38s} FAILED: {e}")

    print("\n=== ranked ===")
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"{k:38s} {v*1000:7.1f} ms/step")
    best = min(results, key=results.get)
    print(f"\nBEST: {best} ({results[best]*1000:.1f} ms/step)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-ips", type=int, default=50000)
    args = p.parse_args()
    main(batch=args.batch, epochs=args.epochs, n_ips=args.n_ips)
