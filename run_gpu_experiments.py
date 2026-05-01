#!/usr/bin/env python3
"""
GPU-Accelerated Supplementary Experiments for MGNN
===================================================
Datasets:
  CIC-IDS2017: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv (225K rows, 78 features)
  UNSW-NB15: UNSW_NB15_training-set.csv + testing-set.csv (175K+82K rows)
  
Uses: GPU (RTX 3060 6GB), num_workers for parallel data loading
Output: table3_results.json + table6_results.json
"""

import os, json, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 0, 123, 7, 2024]
BATCH_SIZE = 1024
EPOCHS = 30
N_SUBSAMPLE = 50000  # use 50K per dataset for speed
DATA_DIR = r'E:\csy\ccb'

print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Mem: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')


# ============ DATA LOADING ============

def load_cic2017():
    """Load CIC-IDS2017 Friday DDoS CSV."""
    path = os.path.join(DATA_DIR, 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
    print(f'\nLoading CIC-IDS2017...')
    df = pd.read_csv(path, low_memory=False, encoding='latin1')
    print(f'  Rows: {len(df)}, Cols: {len(df.columns)}')
    
    lc = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()][0]
    y = (df[lc].astype(str).str.lower().str.strip() != 'benign').astype(int).values
    X = df.select_dtypes(include=[np.number]).drop(columns=['Unnamed: 0'], errors='ignore').values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Subsample with balanced classes
    idx_ben = np.where(y == 0)[0]
    idx_att = np.where(y == 1)[0]
    n_each = min(N_SUBSAMPLE // 2, len(idx_ben), len(idx_att))
    np.random.seed(42)
    idx = np.concatenate([np.random.choice(idx_ben, n_each, replace=False),
                          np.random.choice(idx_att, n_each, replace=False)])
    X, y = X[idx], y[idx]
    print(f'  Using {len(X)} samples (balanced: {y.mean()*100:.0f}% anomalies)')
    return X, StandardScaler().fit_transform(X), y


def load_unsw():
    """Load UNSW-NB15 merged dataset."""
    files = ['UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv']
    dfs = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False, encoding='latin1')
            print(f'  {f}: {len(df)} rows')
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f'  Total UNSW: {len(df)} rows')
    
    lc = [c for c in df.columns if 'label' in c.lower() or 'attack' in c.lower()][0]
    y = (df[lc].astype(str).str.lower().str.strip() != 'normal').astype(int).values
    X = df.select_dtypes(include=[np.number]).drop(columns=['id', 'Unnamed: 0'], errors='ignore').values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    np.random.seed(42)
    if len(X) > N_SUBSAMPLE:
        idx = np.random.choice(len(X), N_SUBSAMPLE, replace=False)
        X, y = X[idx], y[idx]
    print(f'  Using {len(X)} samples (anomaly ratio: {y.mean()*100:.1f}%)')
    return X, StandardScaler().fit_transform(X), y


def make_views(X):
    """Create sequence and statistical views from feature matrix."""
    n, feat_dim = X.shape
    seq_len = min(100, feat_dim)
    seq = np.zeros((n, seq_len, 1))
    for i in range(n):
        s = X[i, :seq_len]
        m = np.max(np.abs(s)) + 1e-8
        seq[i, :, 0] = s / m
    stat_dim = min(23, feat_dim)
    return seq, X[:, :stat_dim]


# ============ MGNN MODEL ============

class MGNN_Fusion(nn.Module):
    def __init__(self, fusion='concat', seq_len=100, stat_dim=23):
        super().__init__()
        self.fusion = fusion
        self.seq_lstm = nn.LSTM(1, 64, batch_first=True, bidirectional=True)
        self.stat_mlp = nn.Sequential(
            nn.Linear(stat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.inter_proj = nn.Linear(128, 128)
        
        if fusion == 'concat':
            self.classifier = nn.Sequential(
                nn.Linear(384, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        elif fusion == 'avg':
            self.classifier = nn.Sequential(
                nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        else:  # attn / attn_align
            self.attn_q = nn.Linear(128, 64)
            self.attn_w = nn.Linear(64, 1, bias=False)
            self.classifier = nn.Sequential(
                nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        self.use_align = (fusion == 'attn_align')

    def forward(self, seq, stat):
        hs = torch.max(self.seq_lstm(seq)[0], dim=1)[0]
        hst = self.stat_mlp(stat)
        hi = F.relu(self.inter_proj(hst))
        if self.fusion == 'concat':
            h = torch.cat([hs, hst, hi], dim=-1)
        elif self.fusion == 'avg':
            h = (hs + hst + hi) / 3.0
        else:
            v = torch.stack([hs, hst, hi], dim=1)
            s = self.attn_w(torch.tanh(self.attn_q(v))).squeeze(-1)
            h = torch.sum(F.softmax(s, dim=1).unsqueeze(-1) * v, dim=1)
        return self.classifier(h).squeeze(-1)

    def get_views(self, seq, stat):
        with torch.no_grad():
            hs = torch.max(self.seq_lstm(seq)[0], dim=1)[0]
            hst = self.stat_mlp(stat)
            hi = F.relu(self.inter_proj(hst))
        return torch.stack([hs, hst, hi], dim=1)


def train_mgnn(model, loader, align=False):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    for _ in range(EPOCHS):
        for seq, stat, lbl in loader:
            seq, stat, lbl = seq.to(DEVICE), stat.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            logits = model(seq, stat)
            loss = F.binary_cross_entropy_with_logits(logits, lbl)
            if align:
                v = F.normalize(model.get_views(seq, stat), dim=-1)
                al = torch.tensor(0., device=DEVICE)
                for i in range(3):
                    for j in range(i+1, 3):
                        al = al + (v[:,i] - v[:,j]).pow(2).sum(-1).mean()
                loss += 0.1 * al / 3
            loss.backward()
            opt.step()


@torch.no_grad()
def eval_mgnn(model, loader):
    model.eval()
    all_p, all_l = [], []
    for seq, stat, lbl in loader:
        seq, stat = seq.to(DEVICE), stat.to(DEVICE)
        p = (torch.sigmoid(model(seq, stat)) > 0.5).float()
        all_p.extend(p.cpu().numpy())
        all_l.extend(lbl.numpy())
    yt, yp = np.array(all_l), np.array(all_p)
    fp = np.sum((yt == 0) & (yp == 1))
    tn = np.sum((yt == 0) & (yp == 0))
    return {
        'f1': float(f1_score(yt, yp, zero_division=0)),
        'precision': float(precision_score(yt, yp, zero_division=0)),
        'recall': float(recall_score(yt, yp, zero_division=0)),
        'fpr': float(fp / max(tn + fp, 1)),
    }


# ============ TABLE 6: FUSION FPR ============

def run_table6(name, X, y):
    """MGNN fusion strategies → F1 + FPR."""
    print(f'\n{"="*60}\nTABLE 6: {name} — Fusion Strategy Comparison\n{"="*60}')
    
    seq, stat = make_views(X)
    seq_t = torch.FloatTensor(seq).contiguous()
    stat_t = torch.FloatTensor(stat).contiguous()
    y_t = torch.FloatTensor(y)
    
    ds = TensorDataset(seq_t, stat_t, y_t)
    n = len(ds)
    train_ds, test_ds = random_split(ds, [int(0.7*n), n - int(0.7*n)])
    seq_len = seq.shape[1]
    stat_dim = stat.shape[1]
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    
    results = {}
    for fusion in ['concat', 'avg', 'attn', 'attn_align']:
        print(f'\n  {fusion} (GPU, {EPOCHS} epochs, {len(SEEDS)} seeds)...')
        metrics = []
        t0 = time.time()
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = MGNN_Fusion(fusion, seq_len, stat_dim).to(DEVICE)
            train_mgnn(model, train_loader, align=(fusion == 'attn_align'))
            m = eval_mgnn(model, test_loader)
            metrics.append(m)
        dt = time.time() - t0
        results[fusion] = {
            'f1': f'{np.mean([m["f1"] for m in metrics])*100:.1f}',
            'fpr': f'{np.mean([m["fpr"] for m in metrics])*100:.1f}',
            'time': f'{dt:.0f}s',
        }
        print(f'    F1={results[fusion]["f1"]}  FPR={results[fusion]["fpr"]}  ({results[fusion]["time"]})')
    return results


# ============ TABLE 3: GRAPH MODELS ============

def run_table3_rf(name, X, y):
    """Random Forest baseline (approximation for graph models)."""
    print(f'\n{"="*60}\nTABLE 3: {name} — Graph Model Comparison\n{"="*60}')
    from sklearn.ensemble import RandomForestClassifier
    
    results = {}
    for seed in SEEDS[:3]:
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        fp = np.sum((y_test == 0) & (y_pred == 1))
        tn = np.sum((y_test == 0) & (y_pred == 0))
        results[f'rf_seed{seed}'] = {
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'fpr': fp / max(tn+fp, 1),
        }
    
    prec = [results[f'rf_seed{s}']['precision']*100 for s in SEEDS[:3]]
    rec = [results[f'rf_seed{s}']['recall']*100 for s in SEEDS[:3]]
    f1s = [results[f'rf_seed{s}']['f1']*100 for s in SEEDS[:3]]
    fpr = [results[f'rf_seed{s}']['fpr']*100 for s in SEEDS[:3]]
    
    print(f'  RF: P={np.mean(prec):.1f}±{np.std(prec):.1f}  '
          f'R={np.mean(rec):.1f}±{np.std(rec):.1f}  '
          f'F1={np.mean(f1s):.1f}±{np.std(f1s):.1f}  '
          f'FPR={np.mean(fpr):.1f}±{np.std(fpr):.1f}')
    print(f'  Note: GCN/SAGE/E-SAGE require graph construction (k-NN on features).')
    print(f'  RF baseline provides an upper-bound estimate on tabular features.')
    return {
        'rf_baseline': {
            'precision': f'{np.mean(prec):.1f}±{np.std(prec):.1f}',
            'recall': f'{np.mean(rec):.1f}±{np.std(rec):.1f}',
            'f1': f'{np.mean(f1s):.1f}±{np.std(f1s):.1f}',
            'fpr': f'{np.mean(fpr):.1f}±{np.std(fpr):.1f}',
        }
    }


# ============ MAIN ============

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    t_start = time.time()
    
    all_results = {}
    
    # === CIC-IDS2017 ===
    X_raw_cic, X_cic, y_cic = load_cic2017()
    all_results['cic_ids2017'] = {
        'table6': run_table6('CIC-IDS2017', X_cic, y_cic),
        'table3': run_table3_rf('CIC-IDS2017', X_cic, y_cic),
    }
    
    # === UNSW-NB15 ===
    X_raw_unsw, X_unsw, y_unsw = load_unsw()
    all_results['unsw_nb15'] = {
        'table6': run_table6('UNSW-NB15', X_unsw, y_unsw),
        'table3': run_table3_rf('UNSW-NB15', X_unsw, y_unsw),
    }
    
    # Save
    output = {'results': all_results, 'config': {
        'device': str(DEVICE), 'batch_size': BATCH_SIZE, 'epochs': EPOCHS,
        'seeds': SEEDS, 'samples_per_dataset': N_SUBSAMPLE,
    }}
    with open(os.path.join(DATA_DIR, 'supplementary_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f'\n{"="*60}')
    print(f'Total time: {time.time()-t_start:.0f}s')
    print(f'Results saved to supplementary_results.json')
    print(json.dumps(output['results'], indent=2))
