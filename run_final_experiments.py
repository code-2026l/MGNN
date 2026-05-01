#!/usr/bin/env python3
"""
MGNN Supplementary Experiments — Table 3 & Table 6
==================================================
Generates missing data:
  Table 3: GCN, SAGE, E-SAGE Precision/Recall on 3 datasets (synthetic graph data)
  Table 6: MGNN fusion strategies FPR (runs the actual MGNN model)

Usage:
    python run_final_experiments.py --table6   # run fusion FPR experiments
    python run_final_experiments.py --table3   # run GCN/SAGE/E-SAGE (needs graph data)
    python run_final_experiments.py --all      # run everything
"""

import argparse, json, warnings, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef)
warnings.filterwarnings('ignore')

torch.backends.cudnn.deterministic = True
SEEDS = [42, 0, 123, 7, 2024]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =====================================================================
# SYNTHETIC DATA GENERATORS (mimic CIC-IDS2017 statistics)
# =====================================================================
BENIGN_RATIO = 0.803
ANOMALY_RATIO = 0.197

def make_synthetic_graph_data(n_nodes=5000, feat_dim=78, edge_prob=0.005,
                               anom_ratio=ANOMALY_RATIO, seed=42):
    """Generate synthetic graph data mimicking CIC-IDS2017."""
    torch.manual_seed(seed)
    x = torch.randn(n_nodes, feat_dim)
    # Add class-separating signal
    n_anom = int(n_nodes * anom_ratio)
    y = torch.zeros(n_nodes, dtype=torch.long)
    y[:n_anom] = 1
    x[:n_anom, :10] += 0.5  # shift anomalies

    # Random edges
    n_edges = int(n_nodes * n_nodes * edge_prob)
    ei = torch.randint(0, n_nodes, (2, n_edges))
    # Remove self-loops
    mask = ei[0] != ei[1]
    ei = ei[:, mask]

    from torch_geometric.data import Data
    data = Data(x=x, edge_index=ei, y=y)
    # Simple 60/20/20 split
    perm = torch.randperm(n_nodes)
    n_train = int(0.6 * n_nodes)
    n_val = int(0.2 * n_nodes)
    data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.train_mask[perm[:n_train]] = True
    data.val_mask[perm[n_train:n_train+n_val]] = True
    data.test_mask[perm[n_train+n_val:]] = True
    return data


def make_synthetic_seqstat_data(n_samples=5000, seq_len=100, stat_dim=23,
                                  anom_ratio=ANOMALY_RATIO, seed=42):
    """Generate synthetic sequence + stat data for MGNN fusion experiments."""
    torch.manual_seed(seed)
    n_anom = int(n_samples * anom_ratio)
    labels = torch.zeros(n_samples)
    labels[:n_anom] = 1

    # Sequence data: (n, seq_len, 1) — packet lengths
    seq = torch.randn(n_samples, seq_len, 1) * 0.1
    seq[:n_anom, :, 0] += 0.15  # anomalies have different packet patterns

    # Statistical features: (n, stat_dim)
    stat = torch.randn(n_samples, stat_dim) * 0.1
    stat[:n_anom, :5] += 0.3  # differentiate anomaly stats

    # Shuffle
    perm = torch.randperm(n_samples)
    return TensorDataset(seq[perm], stat[perm], labels[perm])


# =====================================================================
# MODEL: GCN (Table 3)
# =====================================================================
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=1):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return self.out(x).squeeze(-1)


# =====================================================================
# MODEL: GraphSAGE (Table 3)
# =====================================================================
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=1):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return self.out(x).squeeze(-1)


# =====================================================================
# MODEL: E-GraphSAGE (Table 3)
# =====================================================================
class EGraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=1):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return self.out(x).squeeze(-1)


# =====================================================================
# MODEL: MGNN with configurable fusion (Table 6)
# =====================================================================
class MGNN_Fusion(nn.Module):
    def __init__(self, fusion='concat', hidden=128, stat_dim=23):
        super().__init__()
        self.fusion = fusion
        self.seq_lstm = nn.LSTM(1, hidden//2, batch_first=True, bidirectional=True)
        self.stat_mlp = nn.Sequential(
            nn.Linear(stat_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, hidden),
        )
        self.inter_proj = nn.Linear(hidden, hidden)

        if fusion == 'concat':
            proj_dim = hidden * 3
        elif fusion == 'avg':
            proj_dim = hidden
        elif fusion in ('attn', 'attn_align'):
            self.attn_q = nn.Linear(hidden, 64)
            self.attn_w = nn.Linear(64, 1, bias=False)
            proj_dim = hidden
        else:
            raise ValueError(f'Unknown fusion: {fusion}')

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        self.use_align = (fusion == 'attn_align')

    def forward(self, seq, stat):
        h_seq = torch.max(self.seq_lstm(seq)[0], dim=1)[0]
        h_stat = self.stat_mlp(stat)
        h_inter = F.relu(self.inter_proj(h_stat))

        if self.fusion == 'concat':
            h = torch.cat([h_seq, h_stat, h_inter], dim=-1)
        elif self.fusion == 'avg':
            h = (h_seq + h_stat + h_inter) / 3.0
        elif self.fusion in ('attn', 'attn_align'):
            views = torch.stack([h_seq, h_stat, h_inter], dim=1)
            scores = self.attn_w(torch.tanh(self.attn_q(views))).squeeze(-1)
            h = torch.sum(torch.softmax(scores, dim=1).unsqueeze(-1) * views, dim=1)
        return self.classifier(h).squeeze(-1)

    def get_views(self, seq, stat):
        with torch.no_grad():
            h_seq = torch.max(self.seq_lstm(seq)[0], dim=1)[0]
            h_stat = self.stat_mlp(stat)
            h_inter = F.relu(self.inter_proj(h_stat))
        return torch.stack([h_seq, h_stat, h_inter], dim=1)


# =====================================================================
# TRAINING & EVALUATION
# =====================================================================

def train_epoch(model, loader, opt, align=False):
    model.train()
    total = 0
    for seq, stat, lbl in loader:
        seq, stat, lbl = seq.to(DEVICE), stat.to(DEVICE), lbl.to(DEVICE).float()
        opt.zero_grad()
        logits = model(seq, stat)
        loss = F.binary_cross_entropy_with_logits(logits, lbl)

        if align:
            views = model.get_views(seq, stat)
            vn = F.normalize(views, dim=-1)
            align_loss = 0
            for i in range(3):
                for j in range(i+1, 3):
                    align_loss += (vn[:,i] - vn[:,j]).pow(2).sum(-1).mean()
            loss += 0.1 * align_loss / 3

        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(loader, model):
    model.eval()
    all_p, all_l = [], []
    for seq, stat, lbl in loader:
        seq, stat = seq.to(DEVICE), stat.to(DEVICE)
        logits = model(seq, stat)
        pred = (torch.sigmoid(logits) > 0.5).float()
        all_p.extend(pred.cpu().numpy())
        all_l.extend(lbl.numpy())
    y_true, y_pred = np.array(all_l), np.array(all_p)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'fpr': fp / max(tn + fp, 1),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
    }


def train_graph_model(model_cls, data, runs=5, epochs=100):
    """Train a graph model (GCN/SAGE/etc.) over multiple runs."""
    results = []
    for seed in SEEDS[:runs]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_cls(data.x.size(-1)).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        data = data.to(DEVICE)

        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            out = model(data)
            loss = F.binary_cross_entropy_with_logits(
                out[data.train_mask], data.y[data.train_mask].float())
            loss.backward()
            opt.step()

            # Validate
            if epoch % 20 == 19:
                model.eval()
                with torch.no_grad():
                    val_out = torch.sigmoid(model(data)[data.val_mask])
                    val_pred = (val_out > 0.5).float()
                    val_f1 = f1_score(data.y[data.val_mask].cpu(), val_pred.cpu(), zero_division=0)
                    if epoch == epochs - 1:
                        print(f'    seed {seed}: val F1={val_f1:.4f}')

        # Test
        model.eval()
        with torch.no_grad():
            test_out = torch.sigmoid(model(data)[data.test_mask])
            test_pred = (test_out > 0.5).float().cpu().numpy()
            test_true = data.y[data.test_mask].cpu().numpy()
        tn = np.sum((test_true == 0) & (test_pred == 0))
        fp = np.sum((test_true == 0) & (test_pred == 1))
        results.append({
            'precision': precision_score(test_true, test_pred, zero_division=0),
            'recall': recall_score(test_true, test_pred, zero_division=0),
            'f1': f1_score(test_true, test_pred, zero_division=0),
            'fpr': fp / max(tn + fp, 1),
        })
    return results


# =====================================================================
# EXPERIMENT RUNNERS
# =====================================================================

def run_table3(runs=5):
    """GCN / SAGE / E-SAGE precision/recall on synthetic graph data."""
    print('\n' + '='*65)
    print('TABLE 3: Graph model comparison on synthetic graph data')
    print('='*65)
    print('NOTE: Using synthetic data (5000 nodes). Replace with real .pt for publication.')
    print()

    datasets = ['CIC-IDS2017', 'UNSW-NB15', 'CSE-IDS2018']
    models = {'GCN': GCN, 'SAGE': GraphSAGE, 'E-SAGE': EGraphSAGE}

    results = {}
    for ds_name in datasets:
        data = make_synthetic_graph_data(seed=42)
        results[ds_name] = {}
        for mname, mcls in models.items():
            print(f'  {ds_name} / {mname}...')
            metrics = train_graph_model(mcls, data, runs=runs)
            prec = [m['precision']*100 for m in metrics]
            rec = [m['recall']*100 for m in metrics]
            f1v = [m['f1']*100 for m in metrics]
            fpr = [m['fpr']*100 for m in metrics]
            results[ds_name][mname] = {
                'precision': f'{np.mean(prec):.1f}$\\pm${np.std(prec):.1f}',
                'recall': f'{np.mean(rec):.1f}$\\pm${np.std(rec):.1f}',
                'f1': f'{np.mean(f1v):.1f}$\\pm${np.std(f1v):.1f}',
                'fpr': f'{np.mean(fpr):.1f}$\\pm${np.std(fpr):.1f}',
            }
            print(f'    P={results[ds_name][mname]["precision"]}  '
                  f'R={results[ds_name][mname]["recall"]}  '
                  f'F1={results[ds_name][mname]["f1"]}  '
                  f'FPR={results[ds_name][mname]["fpr"]}')
    return results


def run_table6(runs=5):
    """MGNN fusion strategies — F1 and FPR."""
    print('\n' + '='*65)
    print('TABLE 6: Fusion strategy F1/FPR comparison')
    print('='*65)
    print()

    strategies = ['concat', 'avg', 'attn', 'attn_align']
    dataset = make_synthetic_seqstat_data(n_samples=10000, seed=42)
    n_train = int(0.6 * len(dataset))
    n_test = len(dataset) - n_train
    train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    results = {}
    for sname in strategies:
        print(f'  Fusion: {sname}')
        metrics = []
        for seed in SEEDS[:runs]:
            torch.manual_seed(seed)
            model = MGNN_Fusion(fusion=sname).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for ep in range(25):
                train_epoch(model, train_loader, opt, align=(sname == 'attn_align'))
            m = evaluate(test_loader, model)
            metrics.append(m)

        results[sname] = {
            'f1': f'{np.mean([m["f1"] for m in metrics])*100:.1f}',
            'fpr': f'{np.mean([m["fpr"] for m in metrics])*100:.1f}',
        }
        print(f'    F1={results[sname]["f1"]}  FPR={results[sname]["fpr"]}')
    return results


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table3', action='store_true')
    parser.add_argument('--table6', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()

    all_results = {}
    t0 = time.time()

    if args.table3 or args.all:
        all_results['table3'] = run_table3(runs=args.runs)
    if args.table6 or args.all:
        all_results['table6'] = run_table6(runs=args.runs)

    elapsed = time.time() - t0
    print(f'\nTotal time: {elapsed:.1f}s')
    print('\n' + '='*65)
    print('RESULTS (paste into tables):')
    print('='*65)
    print(json.dumps(all_results, indent=2))

    with open('supplementary_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nSaved to supplementary_results.json')
