#!/usr/bin/env python3
"""A100 kernel-config benchmark for the MGNN training step.

Times a single fused fwd+bwd on a synthetic batch shaped exactly like the real
CIC/UNSW data, across a matrix of CUDA configurations, and reports the fastest.
Determines whether torch.compile (dynamic/static/max-autotune) and TF32 help
on this Ampere box, so the paper run can pick the right lever set offline.
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


def main(batch=2048, seq_len=100, stat_dim=23, hidden=128, epochs=30):
    device = torch.device("cuda")
    torch.manual_seed(0)
    seq = torch.randn(batch, seq_len, 1, device=device)
    stat = torch.randn(batch, stat_dim, device=device)
    lbl = (torch.rand(batch, device=device) > 0.5).float()

    base = kwargs = dict(fusion="attn_align", seq_len=seq_len, stat_dim=stat_dim,
                         hidden=hidden, use_gat=True)

    def make():
        return MGNN(**base).to(device)

    # --- config matrix -----------------------------------------------------
    cfgs = {
        "bf16 only (baseline)":       dict(amp=True, tf32=False, mode=None),
        "bf16 + TF32":                dict(amp=True, tf32=True,  mode=None),
        "bf16 + TF32 + compile(static)": dict(amp=True, tf32=True, mode="static"),
        "bf16 + TF32 + compile(dynamic)": dict(amp=True, tf32=True, mode="dynamic"),
    }
    results = {}
    for name, c in cfgs.items():
        torch.backends.cuda.matmul.allow_tf32 = c.get("tf32", False)
        torch.backends.cudnn.allow_tf32 = c.get("tf32", False)
        model = make()
        if c.get("mode"):
            fullgraph = (c["mode"] == "static")
            try:
                model = torch.compile(model, dynamic=(c["mode"] == "dynamic"),
                                      fullgraph=fullgraph,
                                      mode="max-autotune-no-cudagraphs")
            except Exception as e:
                print(f"[{name}] compile failed: {e}")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        def step():
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, h, views = model.forward_all(seq, stat)
                loss = F.binary_cross_entropy_with_logits(logits, lbl)
                # L_reg + L_align as in training.py
                hg = h.detach().requires_grad_(True)
                cl = model.classifier(hg)
                g = torch.autograd.grad(cl.pow(2).mean(), hg,
                                        retain_graph=True)[0]
                loss = loss + 0.3 * g.pow(2).sum(-1).mean()
                vn = F.normalize(views, dim=-1)
                al = (vn[:, 0] - vn[:, 1]).pow(2).sum(-1).mean() + \
                     (vn[:, 0] - vn[:, 2]).pow(2).sum(-1).mean() + \
                     (vn[:, 1] - vn[:, 2]).pow(2).sum(-1).mean()
                loss = loss + 0.1 * al / 3
            loss.backward()
            opt.step()

        try:
            dt = _timed_cfg(step)
            results[name] = dt
            print(f"{name:38s} {dt*1000:7.1f} ms/step "
                  f"({epochs * (batch / dt) / 1e6:.2f} M-samples/s, "
                  f"~{dt*1e3/1000*epochs/60:.1f} min/epoch")
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
    args = p.parse_args()
    main(batch=args.batch, epochs=args.epochs)