"""
Data download and preprocessing for the three MGNN benchmark datasets.

Supported datasets (real, publicly available):
  - CIC-IDS2017      : per-day CSV, label column "Label" (benign vs attack)
  - UNSW-NB15        : labeled CSV, labels 0/1, categorical proto/service/state
  - CSE-CIC-IDS2018  : per-day CSV, label column "Label" (benign vs attack)

All pipelines produce a single .pt file with the schema:
    seq      : float32 (N, seq_len, 1)     sequence view
    stat     : float32 (N, stat_dim)       statistical view
    labels   : float32 (N,)                0/1 anomaly labels
    features : float32 (N, D)              full numeric feature matrix
    raw_rows  (UNSW only): per-class counts used to sanity-check labels
"""

import os
import ssl
import urllib.request
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

SEQ_LEN = 100
STAT_DIM = 23

# ---------------------------------------------------------------------------
# Download sources
# ---------------------------------------------------------------------------

# CSE-CIC-IDS2018: official public S3 bucket (verified reachable, no signing).
# Official names use "Weekday-Day-Month-Year" with the misspelled "Thuesday".
CSE_IDS2018_BASE = (
    "https://cse-cic-ids2018.s3.amazonaws.com/"
    "Processed%20Traffic%20Data%20for%20ML%20Algorithms/"
)
CSE_IDS2018_FILES = [
    "WednesDAY-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]

# CIC-IDS2017 per-day files (names as released by UNB).
CIC_IDS2017_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]
# Try each source in order; first live mirror wins for that file.
# NOTE: the official cicresearch.ca/UNB hosts now 302-redirect every file to a
# landing page (no direct CSV), so we use a verified GitHub mirror that ships
# the raw per-day CSVs. `python -m src.data...` uses this for auto download.
# As of 2026-09, yashpotdar-py/cicids-dataset (default branch `main`) hosts all
# 8 files, each reachable via raw.githubusercontent (HEAD 200 verified).
CIC_IDS2017_SOURCES = [
    "https://raw.githubusercontent.com/yashpotdar-py/cicids-dataset/main/",
]

# UNSW-NB15 official labeled files.
UNSW_LABELED_FILES = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]
UNSW_SOURCES = [
    "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
]


def _url_join(base, name):
    return base.rstrip("/") + "/" + name


def download_file(url, dest, desc="Downloading"):
    """Download a single file with progress. Returns True on success."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  {os.path.basename(dest)} exists, skipping download")
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=120) as resp:
            total = int(resp.headers.get("content-length", 0))
            got = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
            print(f"  {desc} {os.path.basename(dest)}: {got/1024/1024:.1f}MB")
            return True
    except Exception as e:
        print(f"  download failed for {url}: {e}")
        return False


def download_dataset_files(name, data_dir):
    """Best-effort download of the raw CSVs for a dataset.

    Returns the list of local file paths that exist. Missing files can be
    supplied manually under ``data_dir`` — preprocessing accepts whatever is
    present, so a partially downloaded set still works.
    """
    os.makedirs(data_dir, exist_ok=True)
    if name == "cse_ids2018":
        files = []
        for fn in CSE_IDS2018_FILES:
            dest = os.path.join(data_dir, fn)
            if download_file(_url_join(CSE_IDS2018_BASE, fn), dest, "IDS2018"):
                files.append(dest)
        return files
    if name == "cic_ids2017":
        files = []
        for fn in CIC_IDS2017_FILES:
            dest = os.path.join(data_dir, fn)
            ok = False
            for base in CIC_IDS2017_SOURCES:
                if base.endswith(".html"):
                    continue
                if download_file(_url_join(base, fn), dest, "CIC"):
                    ok = True
                    break
            if ok:
                files.append(dest)
        return files
    if name == "unsw_nb15":
        files = []
        for fn in UNSW_LABELED_FILES:
            dest = os.path.join(data_dir, fn)
            if os.path.exists(dest):
                files.append(dest)
        print("  UNSW-NB15: place training/testing CSV files in data_dir, "
              "or the script/src.data.download_dataset_files.sources may be edited.")
        return files
    return []


# ---------------------------------------------------------------------------
# Schema-aware CSV -> (X, y) extraction
# ---------------------------------------------------------------------------

_CATEGORICAL_COLS = {"proto", "service", "state"}


def _extract_cic_like(df):
    """Features + labels for CIC-IDS2017 / CSE-CIC-IDS2018 style CSVs."""
    df.columns = [str(c).strip() for c in df.columns]
    label_col = next((c for c in df.columns if c.lower() in ("label", " class")), None)
    if label_col is None or label_col not in df.columns:
        raise ValueError(f"No label column found; got {list(df.columns)[:12]}...")

    drop = {"flow id", "src ip", "dst ip", "src port", "dst port",
            "timestamp", "simillarhttp", "fwd urg flags", "unnamed: 0"}
    X_df = df.drop(columns=[c for c in df.columns
                            if c.lower() in drop or "label" in c.lower()])
    X = X_df.apply(pd.to_numeric, errors="coerce").values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    y = (df[label_col].astype(str).str.lower().str.strip() != "benign").astype(int)
    return X, y.to_numpy()


def _extract_unsw(df):
    """Features + labels for UNSW-NB15 labeled CSVs (0/1 label column)."""
    df.columns = [str(c).strip() for c in df.columns]
    cat = [c for c in df.columns if c in _CATEGORICAL_COLS]
    feat_cols = [c for c in df.columns if c not in cat
                 and c not in ("id", "attack_cat", "label", "bin class")]
    X_df = df[feat_cols].copy()
    for c in cat:
        X_df[c] = df[c].astype("category").cat.codes
    X = X_df.apply(pd.to_numeric, errors="coerce").values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    if "label" in df.columns:
        y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        y = (df.get("attack_cat", pd.Series("", index=df.index)).astype(str) != "Normal").astype(int).to_numpy()
    return X, y


def preprocess_csvs(file_paths, name, data_dir, output_name, n_samples=None,
                    seq_len=SEQ_LEN, stat_dim=STAT_DIM):
    """Load raw CSVs, build the three views, and save to ``data_dir/output_name``.

    Args:
        file_paths: list of existing CSV paths (schema per ``name``).
        name: 'cic_ids2017' | 'unsw_nb15' | 'cse_ids2018'.
        n_samples: optional cap (None keeps everything). Use a small value only
            for fast smoke tests; the paper uses the full labeled sets.
    """
    extractor = _extract_unsw if name == "unsw_nb15" else _extract_cic_like
    all_X, all_y = [], []
    for fp in file_paths:
        if not os.path.exists(fp):
            print(f"  skipping {os.path.basename(fp)} (not found)")
            continue
        print(f"  loading {os.path.basename(fp)}...", flush=True)
        df = pd.read_csv(fp, low_memory=False, encoding="latin1")
        X, y = extractor(df)
        print(f"    rows={len(df)} feats={X.shape[1]} "
              f"anomaly={y.mean()*100:.2f}%")
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise FileNotFoundError(
            f"No usable {name} CSV files provided. Supply the raw CSVs in "
            f"{data_dir} or run the downloader.")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    if n_samples is not None and len(X) > n_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]
        print(f"  NOTE: capped to {n_samples} rows (smoke-test mode)")

    X = StandardScaler().fit_transform(X)
    seq, stat = _make_views(X, seq_len=seq_len, stat_dim=stat_dim)

    data = {
        "seq": torch.FloatTensor(seq),
        "stat": torch.FloatTensor(stat),
        "labels": torch.FloatTensor(y),
        "features": torch.FloatTensor(X),
        "anomaly_rate": float(y.mean()),
        "n_total": int(len(y)),
    }
    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(data_dir, output_name)
    torch.save(data, out)
    print(f"  saved {out}  (N={len(y)}, feat={X.shape[1]}, "
          f"anom={y.mean()*100:.2f}%)")
    return data


def _make_views(X, seq_len=SEQ_LEN, stat_dim=STAT_DIM):
    """Build the sequence and statistical views used by MGNN."""
    n, feat_dim = X.shape
    sl = min(seq_len, feat_dim)
    seq = np.zeros((n, sl, 1))
    for i in range(n):
        s = X[i, :sl]
        m = float(np.max(np.abs(s))) + 1e-8
        seq[i, :, 0] = s / m
    return seq, X[:, :min(stat_dim, feat_dim)]


# ---------------------------------------------------------------------------
# Convenience entry points per dataset
# ---------------------------------------------------------------------------

def prepare_cic_ids2017(data_dir, n_samples=None, download=True):
    """Download (optional) and preprocess CIC-IDS2017 -> cic_ids2017.pt"""
    paths = []
    if download:
        paths = download_dataset_files("cic_ids2017", data_dir)
    if not paths:
        paths = [os.path.join(data_dir, f) for f in CIC_IDS2017_FILES
                 if os.path.exists(os.path.join(data_dir, f))]
    return preprocess_csvs(paths, "cic_ids2017", data_dir,
                           "cic_ids2017.pt", n_samples=n_samples)


def prepare_unsw_nb15(data_dir, n_samples=None, download=True):
    """Preprocess UNSW-NB15 labeled CSVs -> unsw_nb15.pt"""
    paths = []
    if download:
        paths = download_dataset_files("unsw_nb15", data_dir)
    if not paths:
        candidates = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]
        paths = [os.path.join(data_dir, f)
                 for f in candidates if os.path.exists(os.path.join(data_dir, f))]
    if not paths:  # fall back to the full split files if present
        candidates = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
        paths = [os.path.join(data_dir, f) for f in candidates
                 if os.path.exists(os.path.join(data_dir, f))]
    return preprocess_csvs(paths, "unsw_nb15", data_dir,
                           "unsw_nb15.pt", n_samples=n_samples)


def prepare_cse_ids2018(data_dir, n_samples=None, download=True):
    """Download (optional) and preprocess CSE-CIC-IDS2018 -> cse_ids2018.pt"""
    paths = []
    if download:
        paths = download_dataset_files("cse_ids2018", data_dir)
    if not paths:
        paths = [os.path.join(data_dir, f) for f in CSE_IDS2018_FILES
                 if os.path.exists(os.path.join(data_dir, f))]
    return preprocess_csvs(paths, "cse_ids2018", data_dir,
                           "cse_ids2018.pt", n_samples=n_samples)


def load_pt_data(data_dir, name):
    """Load a preprocessed .pt file (raises if missing)."""
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the prepare_* function for this dataset first.")
    return torch.load(path, weights_only=False)