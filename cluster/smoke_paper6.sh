#!/bin/bash
# Paper-condition smoke on CIC-IDS2017: 1 seed, 3 epochs, check split+metrics.
set -euo pipefail
cd ~/mgnn
source .venv/bin/activate
export PYTHONPATH="$PWD"
python - <<'PY' 2>&1 | tail -35
import time, torch, numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from src.data.dataset import load_pt_data
from src.models.mgnn import MGNN
from src.utils.training import train_epoch_mgnn, evaluate_mgnn
from src.utils.metrics import format_metrics
import experiments.run_table6 as R

device = torch.device('cuda')
data = load_pt_data('data', 'cic_ids2017.pt')
seq, stat, labels = data['seq'], data['stat'], data['labels']
n = len(labels)
train_idx, val_idx, test_idx = R._chrono_split(labels)
train_us = R._undersample_train(train_idx, labels, ratio=3)
print(f"CIC N={n} train={len(train_idx)} us={len(train_us)} val={len(val_idx)} test={len(test_idx)}")
print(f"  anom: all={labels.mean().item()*100:.1f}% us_train={labels[train_us].mean().item()*100:.1f}% test={labels[test_idx].mean().item()*100:.1f}%")
ds = TensorDataset(seq, stat, labels)
tr = DataLoader(Subset(ds, train_us.tolist()), batch_size=1024, shuffle=True)
va = DataLoader(Subset(ds, val_idx.tolist()), batch_size=1024)
te = DataLoader(Subset(ds, test_idx.tolist()), batch_size=1024)
model = MGNN(fusion='attn_align', seq_len=seq.size(1), stat_dim=stat.size(1), hidden=128, use_gat=True).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best=(-1,None); t0=time.time()
for e in range(3):
    train_epoch_mgnn(model, tr, opt, device, align=True)
    vm = evaluate_mgnn(va, model, device)
    if vm['f1']>best[0]: best=(vm['f1'], {k:v.clone() for k,v in model.state_dict().items()})
    print(f"  ep{e+1} val_F1={vm['f1']} ({time.time()-t0:.0f}s)")
model.load_state_dict(best[1])
print("test meta:", format_metrics([evaluate_mgnn(te, model, device)]))
print("PAPER SMOKE OK")
PY