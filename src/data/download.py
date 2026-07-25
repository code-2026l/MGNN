"""
Data download and preprocessing utilities for benchmark datasets.

Supports:
  - CIC-IDS2017: UNB official CSV files
  - UNSW-NB15: UNSW Canberra CSV files
  - CSE-CIC-IDS2018: UNB official CSV files
"""

import os
import sys
import ssl
import urllib.request
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# SSL context for dataset download (research servers often have expired certs)
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE


def download_file(url, dest, desc="Downloading"):
    """Download a file with progress display."""
    if os.path.exists(dest):
        print(f'  {dest} exists, skipping download')
        return True
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=120) as resp:
            total = int(resp.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192
            with open(dest, 'wb') as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        sys.stdout.write(
                            f'\r  {desc}: {pct:.0f}% ({downloaded/1024/1024:.1f}MB)')
                        sys.stdout.flush()
            print(f'\r  {desc}: 100% ({downloaded/1024/1024:.1f}MB)')
            return True
    except Exception as e:
        print(f'  Download failed: {e}')
        return False


def download_cic_ids2017(data_dir):
    """Download CIC-IDS2017 CSV files from UNB."""
    print('\nDownloading CIC-IDS2017...')
    cic_files = {
        'Monday-WorkingHours.pcap_ISCX.csv':
            'https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSV/'
            'MachineLearningCSV/Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv':
            'https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSV/'
            'MachineLearningCSV/Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv':
            'https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSV/'
            'MachineLearningCSV/Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv':
            'https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSV/'
            'MachineLearningCSV/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Friday-WorkingHours.pcap_ISCX.csv':
            'https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSV/'
            'MachineLearningCSV/Friday-WorkingHours.pcap_ISCX.csv',
    }
    downloaded = []
    for fname, url in cic_files.items():
        dest = os.path.join(data_dir, fname)
        if download_file(url, dest, f'CIC {fname[:10]}'):
            downloaded.append(dest)
    print(f'Downloaded {len(downloaded)} / {len(cic_files)} CIC-IDS2017 files.')
    return downloaded


def extract_flow_features(df):
    """Extract feature matrix and labels from CIC-IDS2017/2018 CSV."""
    df.columns = df.columns.str.strip()
    drop_cols = ['Label', 'Fwd Label', 'Flow ID', 'Src IP', 'Dst IP',
                 'Timestamp', 'SimillarHTTP', 'Fwd URG Flags']
    drop_cols = [c for c in drop_cols if c in df.columns]
    if 'Label' in df.columns:
        y = (df['Label'].astype(str).str.lower().str.strip() != 'benign').astype(int).values
    else:
        print('  WARNING: No Label column found, using zeros.')
        y = np.zeros(len(df), dtype=int)
    X_df = df.drop(
        columns=[c for c in drop_cols if c in df.columns]
                + (['Unnamed: 0'] if 'Unnamed: 0' in df.columns else []),
        errors='ignore')
    X = X_df.apply(pd.to_numeric, errors='coerce').values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    X = np.clip(X, -1e10, 1e10)
    return X, y


def make_views(X, seq_len=100, stat_dim=23):
    """Create sequence and statistical views from feature matrix."""
    n, feat_dim = X.shape
    seq_len = min(seq_len, feat_dim)
    seq = np.zeros((n, seq_len, 1))
    for i in range(n):
        s = X[i, :seq_len]
        m = np.max(np.abs(s)) + 1e-8
        seq[i, :, 0] = s / m
    stat_dim = min(stat_dim, feat_dim)
    return seq, X[:, :stat_dim]


def preprocess_cic_csv(file_paths, data_dir, output_name='cic_ids2017.pt',
                       n_samples=50000, seq_len=100, stat_dim=23):
    """Load CIC CSVs, preprocess, and save to .pt file."""
    print(f'\nPreprocessing CIC-IDS2017 -> {output_name}')
    all_X, all_y = [], []
    for fp in file_paths:
        if not os.path.exists(fp):
            print(f'  Skipping {fp} (not found)')
            continue
        print(f'  Loading {os.path.basename(fp)}...')
        try:
            df = pd.read_csv(fp, low_memory=False, encoding='latin1')
            print(f'    Rows: {len(df)}, Cols: {len(df.columns)}')
            X, y = extract_flow_features(df)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f'    Error: {e}')
    if not all_X:
        print('  No data loaded.')
        return None
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f'  Total: {len(X)} samples, {X.shape[1]} features')
    if len(X) > n_samples:
        idx = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]
        print(f'  Subsampled to {n_samples}')
    X = StandardScaler().fit_transform(X)
    seq, stat = make_views(X, seq_len=seq_len, stat_dim=stat_dim)
    data = {
        'seq': torch.FloatTensor(seq),
        'stat': torch.FloatTensor(stat),
        'labels': torch.FloatTensor(y),
        'features': torch.FloatTensor(X),
    }
    save_path = os.path.join(data_dir, output_name)
    torch.save(data, save_path)
    print(f'  Saved to {save_path}')
    return data


def load_pt_data(data_dir, name):
    """Load preprocessed .pt file from data directory."""
    path = os.path.join(data_dir, name)
    if os.path.exists(path):
        print(f'  Loading {path}')
        return torch.load(path, weights_only=False)
    print(f'  {path} not found.')
    return None
