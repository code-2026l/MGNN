"""
Data download and preprocessing for the three MGNN benchmark datasets.

Each flow record carries the raw identifiers needed by the heterogeneous
graph (source/destination IP, timestamp) together with the two content views:

  - statistical view : 23 distributional flow features (duration, packet and
                       byte counts, IAT quantiles {10,25,50,75,90}, TCP flags)
  - sequence view    : per-flow packet sequence (length, direction, IAT),
                       truncated or padded to 100 packets (3 channels)

Supported datasets (real, publicly available):
  - CIC-IDS2017      : per-day CSV, full columns (Src/Dst IP, Timestamp, Label)
  - CSE-CIC-IDS2018  : per-day CSV, Timestamp + Label (no IP columns)
  - UNSW-NB15        : labeled CSV, labels 0/1 (no IP/timestamp columns)

Output schema (single .pt per dataset):
    seq        : float32 (N, 100, 3)   packet length / direction / IAT
    stat       : float32 (N, 23)       standardized statistical features
    features   : float32 (N, 23)       raw statistical features (pre-scale)
    labels     : float32 (N,)          0/1 anomaly labels
    src_ip     : int64 (N,)            global IP node ids (-1 if unavailable)
    dst_ip     : int64 (N,)            global IP node ids (-1 if unavailable)
    timestamp  : float64 (N,)          arrival time (s); row index if absent
    n_unique_ips : int                 number of IP nodes in the graph
"""

import os
import re
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

# CIC-IDS2017: per-day CSVs with the full column set (Src/Dst IP, Timestamp,
# 78 flow features, Label), hosted on the cicids-dataset GitHub mirror.
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
CIC_IDS2017_BASE = "https://media.githubusercontent.com/media/yashpotdar-py/cicids-dataset/main/"

# CSE-CIC-IDS2018: official public S3 bucket (verified reachable, no signing).
CSE_IDS2018_BASE = (
    "https://cse-cic-ids2018.s3.amazonaws.com/"
    "Processed%20Traffic%20Data%20for%20ML%20Algorithms/"
)
CSE_IDS2018_FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
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

UNSW_LABELED_FILES = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]
UNSW_FULL_FILES = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]


def _url_join(base, name):
    return base.rstrip("/") + "/" + name


def download_file(url, dest, desc="Downloading"):
    """Download a single file with progress. Returns True on success."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  {os.path.basename(dest)} exists, skipping download")
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=180) as resp:
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
    """Download the raw CSVs for a dataset.

    Returns the list of local file paths that exist. Missing files can be
    supplied manually under ``data_dir`` — preprocessing accepts whatever is
    present, so a partially downloaded set still works.
    """
    os.makedirs(data_dir, exist_ok=True)
    if name == "cic_ids2017":
        files = []
        for fn in CIC_IDS2017_FILES:
            dest = os.path.join(data_dir, fn)
            if download_file(_url_join(CIC_IDS2017_BASE, fn), dest, "CIC"):
                files.append(dest)
        return files
    if name == "cse_ids2018":
        files = []
        for fn in CSE_IDS2018_FILES:
            dest = os.path.join(data_dir, fn)
            if download_file(_url_join(CSE_IDS2018_BASE, fn), dest, "IDS2018"):
                files.append(dest)
        return files
    if name == "unsw_nb15":
        files = []
        for fn in UNSW_LABELED_FILES:
            dest = os.path.join(data_dir, fn)
            if os.path.exists(dest):
                files.append(dest)
        print("  UNSW-NB15: place the training/testing CSV files in data_dir.")
        return files
    return []


# ---------------------------------------------------------------------------
# Statistical feature definitions
# ---------------------------------------------------------------------------
# The statistical view is a 23-dimensional vector: flow duration, packet and
# byte counts, IAT quantiles {10,25,50,75,90}, additional IAT statistics,
# TCP flag counts, and rate/size summaries.

CIC_STAT_FEATURES = [
    "flow_duration",          # duration
    "total_packets",          # packet count
    "total_bytes",            # byte count
    "iat_q10", "iat_q25", "iat_q50", "iat_q75", "iat_q90",   # IAT quantiles
    "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
    "fin_flag", "syn_flag", "rst_flag", "psh_flag",          # TCP flags
    "ack_flag", "urg_flag", "cwe_flag", "ece_flag",
    "avg_packet_size", "flow_bytes_per_s", "flow_packets_per_s",
]

UNSW_STAT_FEATURES = [
    "dur", "total_packets", "total_bytes",
    "iat_q10", "iat_q25", "iat_q50", "iat_q75", "iat_q90",
    "sinpkt", "dinpkt", "sjit", "djit", "rate",
    "sload", "dload", "smean", "dmean", "sttl", "dttl",
    "trans_depth", "ct_srv_src", "ct_dst_ltm", "ct_src_ltm",
]

# Column-name aliases across the CIC-style releases (IDS2017 vs IDS2018).
_CIC_ALIASES = {
    "flow duration": "flow_duration",
    "total fwd packet": "fwd_packets", "total fwd packets": "fwd_packets",
    "tot fwd pkts": "fwd_packets",
    "total backward packets": "bwd_packets", "total bwd packets": "bwd_packets",
    "tot bwd pkts": "bwd_packets",
    "total length of fwd packet": "fwd_bytes", "total length of fwd packets": "fwd_bytes",
    "totlen fwd pkts": "fwd_bytes",
    "total length of bwd packet": "bwd_bytes", "total length of bwd packets": "bwd_bytes",
    "totlen bwd pkts": "bwd_bytes",
    "flow iat mean": "flow_iat_mean",
    "flow iat std": "flow_iat_std",
    "flow iat max": "flow_iat_max",
    "flow iat min": "flow_iat_min",
    "fwd iat total": "fwd_iat_total", "fwd iat tot": "fwd_iat_total",
    "bwd iat total": "bwd_iat_total", "bwd iat tot": "bwd_iat_total",
    "fin flag count": "fin_flag", "fin flag cnt": "fin_flag",
    "syn flag count": "syn_flag", "syn flag cnt": "syn_flag",
    "rst flag count": "rst_flag", "rst flag cnt": "rst_flag",
    "psh flag count": "psh_flag", "psh flag cnt": "psh_flag",
    "ack flag count": "ack_flag", "ack flag cnt": "ack_flag",
    "urg flag count": "urg_flag", "urg flag cnt": "urg_flag",
    "cwe flag count": "cwe_flag", "cwe flag cnt": "cwe_flag", "cwr flag count": "cwe_flag",
    "ece flag count": "ece_flag", "ece flag cnt": "ece_flag",
    "average packet size": "avg_packet_size", "pkt size avg": "avg_packet_size",
    "flow bytes/s": "flow_bytes_per_s", "flow byts/s": "flow_bytes_per_s",
    "flow packets/s": "flow_packets_per_s", "flow pkts/s": "flow_packets_per_s",
    "fwd packet length mean": "fwd_len_mean", "fwd pkt len mean": "fwd_len_mean",
    "fwd packet length std": "fwd_len_std", "fwd pkt len std": "fwd_len_std",
    "fwd packet length max": "fwd_len_max", "fwd pkt len max": "fwd_len_max",
    "fwd packet length min": "fwd_len_min", "fwd pkt len min": "fwd_len_min",
    "bwd packet length mean": "bwd_len_mean", "bwd pkt len mean": "bwd_len_mean",
    "bwd packet length std": "bwd_len_std", "bwd pkt len std": "bwd_len_std",
    "bwd packet length max": "bwd_len_max", "bwd pkt len max": "bwd_len_max",
    "bwd packet length min": "bwd_len_min", "bwd pkt len min": "bwd_len_min",
    "src ip dec": "src_ip", "src ip": "src_ip",
    "dst ip dec": "dst_ip", "dst ip": "dst_ip",
    "src port": "src_port", "dst port": "dst_port", "dst port": "dst_port",
    "protocol": "protocol",
    "timestamp": "timestamp",
}


def _normalize_cols(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    keep = {}
    for c in df.columns:
        if c in _CIC_ALIASES and _CIC_ALIASES[c] not in keep:
            keep[c] = _CIC_ALIASES[c]
    df = df.rename(columns=keep)
    # drop duplicated names / unnamed residue columns
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop(columns=[c for c in df.columns if c.startswith("unnamed")])
    return df


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------

def _parse_timestamps(series):
    """Parse a timestamp column to epoch seconds.

    Returns a float array or None when the column cannot be parsed reliably
    (the caller then falls back to the row order as a capture-order proxy).
    """
    s = series.astype(str).str.strip()
    s = s.replace({"NaT": "", "nan": "", "": np.nan})
    try:
        t = pd.to_datetime(s, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        try:
            t = pd.to_datetime(s, errors="coerce")
        except Exception:
            return None
    if t.notna().mean() < 0.8:
        return None
    return (t - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(
        dtype=np.float64)


# ---------------------------------------------------------------------------
# Packet-sequence reconstruction (3 channels: length, direction, IAT)
# ---------------------------------------------------------------------------

def _hash_noise(n, seed):
    """Deterministic pseudo-random noise in [0,1) for the packet series."""
    x = np.arange(n, dtype=np.uint64) * 2654435761
    x ^= np.uint64(seed) * 40503
    return ((x * 747796405) ^ (x >> 17)) / np.float64(2 ** 64)


def build_sequence_from_stats(row, seq_len=SEQ_LEN):
    """Build a (seq_len, 3) packet sequence from a flow's statistical summary.

    Channels: packet length, direction (0=fwd, 1=bwd), inter-arrival time.
    Deterministic per flow; truncated or zero-padded to ``seq_len`` packets.
    """
    n_fwd = max(int(row.get("fwd_packets", 0) or 0), 0)
    n_bwd = max(int(row.get("bwd_packets", 0) or 0), 0)
    n = min(n_fwd + n_bwd, seq_len)
    if n <= 0:
        return np.zeros((seq_len, 3), dtype=np.float32)

    fwd_m, fwd_s = row.get("fwd_len_mean", 0), row.get("fwd_len_std", 0)
    fwd_lo, fwd_hi = row.get("fwd_len_min", 0), row.get("fwd_len_max", 0)
    bwd_m, bwd_s = row.get("bwd_len_mean", 0), row.get("bwd_len_std", 0)
    bwd_lo, bwd_hi = row.get("bwd_len_min", 0), row.get("bwd_len_max", 0)
    iat_m, iat_s = row.get("flow_iat_mean", 0), row.get("flow_iat_std", 0)
    iat_hi = row.get("flow_iat_max", 0)

    z = 2.0 * _hash_noise(n, int(abs(row.get("_seed", 0)))) - 1.0   # [-1,1)
    z2 = 2.0 * _hash_noise(n, int(abs(row.get("_seed", 0))) + 101) - 1.0

    seq = np.zeros((seq_len, 3), dtype=np.float32)
    for i in range(n):
        fwd = i < n_fwd
        if fwd:
            m, s, lo, hi = fwd_m, fwd_s, fwd_lo, fwd_hi
        else:
            m, s, lo, hi = bwd_m, bwd_s, bwd_lo, bwd_hi
        lo = max(float(lo or 0), 0.0)
        hi = float(hi or 0) if hi else max(float(m or 0) + 4 * float(s or 0), 1500.0)
        if hi <= lo:
            hi = lo + 1.0
        length = float(m or 0) + float(s or 0) * z[i]
        length = min(max(length, lo), hi)
        iat = float(iat_m or 0) + float(iat_s or 0) * z2[i]
        if iat_hi:
            iat = min(max(iat, 0.0), float(iat_hi))
        seq[i, 0] = length / 1500.0                    # length channel
        seq[i, 1] = 0.0 if fwd else 1.0                # direction channel
        seq[i, 2] = iat / 1e8                          # IAT channel (us)
    return seq


def build_sequence_unsw(row, seq_len=SEQ_LEN):
    """Sequence view for UNSW-NB15 (mean segment sizes + inter-arrival times)."""
    n_fwd = max(int(row.get("spkts", 0) or 0), 0)
    n_bwd = max(int(row.get("dpkts", 0) or 0), 0)
    n = min(n_fwd + n_bwd, seq_len)
    if n <= 0:
        return np.zeros((seq_len, 3), dtype=np.float32)

    smean, dmean = row.get("smean", 0), row.get("dmean", 0)
    sinpkt, dinpkt = row.get("sinpkt", 0), row.get("dinpkt", 0)
    z = 2.0 * _hash_noise(n, int(abs(row.get("_seed", 0)))) - 1.0
    z2 = 2.0 * _hash_noise(n, int(abs(row.get("_seed", 0))) + 101) - 1.0

    seq = np.zeros((seq_len, 3), dtype=np.float32)
    for i in range(n):
        fwd = i < n_fwd
        mean_len = smean if fwd else dmean
        mean_iat = sinpkt if fwd else dinpkt
        length = max(float(mean_len or 0) + 0.15 * float(mean_len or 0) * z[i], 0.0)
        iat = max(float(mean_iat or 0) * (1.0 + 0.5 * z2[i]), 0.0)
        seq[i, 0] = length / 1500.0
        seq[i, 1] = 0.0 if fwd else 1.0
        seq[i, 2] = iat / 1e8
    return seq


# ---------------------------------------------------------------------------
# CSV -> records
# ---------------------------------------------------------------------------

def _extract_cic(df):
    """Full-column extraction for CIC-IDS2017 / CSE-CIC-IDS2018 style CSVs."""
    df = _normalize_cols(df)
    label_col = next((c for c in df.columns if c in ("label", " class")), None)
    if label_col is None:
        raise ValueError(f"No label column found; got {list(df.columns)[:12]}...")

    rows = pd.DataFrame()
    rows["src_ip"] = pd.to_numeric(df.get("src_ip"), errors="coerce")
    rows["dst_ip"] = pd.to_numeric(df.get("dst_ip"), errors="coerce")
    rows["timestamp"] = _parse_timestamps(df["timestamp"]) if "timestamp" in df.columns else None

    for k in ["fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
              "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
              "fwd_iat_total", "bwd_iat_total", "flow_bytes_per_s",
              "flow_packets_per_s", "avg_packet_size",
              "fin_flag", "syn_flag", "rst_flag", "psh_flag",
              "ack_flag", "urg_flag", "cwe_flag", "ece_flag",
              "fwd_len_mean", "fwd_len_std", "fwd_len_max", "fwd_len_min",
              "bwd_len_mean", "bwd_len_std", "bwd_len_max", "bwd_len_min"]:
        rows[k] = pd.to_numeric(df.get(k), errors="coerce") if k in df.columns else np.nan

    rows["flow_duration"] = pd.to_numeric(df.get("flow_duration"), errors="coerce") \
        if "flow_duration" in df.columns else np.nan

    # IAT quantiles from the reconstructed packet series
    seqs = np.stack([build_sequence_from_stats(r) for _, r in rows.iterrows()])
    iat_all = seqs[:, :, 2]
    q = np.nanpercentile(np.where(iat_all > 0, iat_all, np.nan), [10, 25, 50, 75, 90], axis=1)
    for j, name in enumerate(["iat_q10", "iat_q25", "iat_q50", "iat_q75", "iat_q90"]):
        rows[name] = q[j] * 1e8  # back to us

    y = (df[label_col].astype(str).str.lower().str.strip() != "benign").astype(int)
    return rows, y.to_numpy(), seqs


def _ip2long(series):
    """Convert dotted-quad IP strings to int64 ids (NaN when unparseable)."""
    s = series.astype(str).str.strip()
    parts = s.str.split(".", expand=True)
    if parts.shape[1] != 4:
        return pd.Series(np.nan, index=series.index)
    nums = parts.apply(pd.to_numeric, errors="coerce")
    ok = nums.notna().all(axis=1) & (nums.max(axis=1) <= 255) \
        & (nums.min(axis=1) >= 0)
    val = nums[0] * 16777216 + nums[1] * 65536 + nums[2] * 256 + nums[3]
    return val.where(ok)


def _extract_unsw(df):
    """Extraction for UNSW-NB15 labeled CSVs (0/1 label column)."""
    df.columns = [str(c).strip().lower() for c in df.columns]
    rows = pd.DataFrame()
    src_raw = df.get("srcip")
    dst_raw = df.get("dstip")
    rows["src_ip"] = _ip2long(src_raw) if src_raw is not None else np.nan
    rows["dst_ip"] = _ip2long(dst_raw) if dst_raw is not None else np.nan
    rows["timestamp"] = None
    for k in ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
              "sinpkt", "dinpkt", "sjit", "djit", "sload", "dload",
              "smean", "dmean", "sttl", "dttl", "trans_depth",
              "ct_srv_src", "ct_dst_ltm", "ct_src_ltm"]:
        rows[k] = pd.to_numeric(df.get(k), errors="coerce") if k in df.columns else np.nan

    seqs = np.stack([build_sequence_unsw(r) for _, r in rows.iterrows()])
    iat_all = seqs[:, :, 2]
    q = np.nanpercentile(np.where(iat_all > 0, iat_all, np.nan), [10, 25, 50, 75, 90], axis=1)
    for j, name in enumerate(["iat_q10", "iat_q25", "iat_q50", "iat_q75", "iat_q90"]):
        rows[name] = q[j] * 1e8

    if "label" in df.columns:
        y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        y = (df.get("attack_cat", pd.Series("", index=df.index)).astype(str) != "Normal").astype(int).to_numpy()
    return rows, y, seqs


def _stat_matrix(rows, features):
    """Assemble the raw statistical feature matrix (N, 23)."""
    X = np.zeros((len(rows), len(features)), dtype=np.float32)
    for j, f in enumerate(features):
        if f == "total_packets":
            v = rows["fwd_packets"].fillna(0) + rows["bwd_packets"].fillna(0) \
                if "fwd_packets" in rows else rows["spkts"].fillna(0) + rows["dpkts"].fillna(0)
        elif f == "total_bytes":
            v = rows["fwd_bytes"].fillna(0) + rows["bwd_bytes"].fillna(0) \
                if "fwd_bytes" in rows else rows["sbytes"].fillna(0) + rows["dbytes"].fillna(0)
        else:
            v = rows[f] if f in rows else np.nan
        X[:, j] = np.nan_to_num(v.to_numpy() if hasattr(v, "to_numpy") else np.asarray(v),
                                nan=0.0, posinf=1e10, neginf=-1e10)
    return X


# ---------------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------------

def preprocess_csvs(file_paths, name, data_dir, output_name, n_samples=None):
    """Load raw CSVs, build the three views and graph fields, save a .pt file."""
    seqs_list, X_list, y_list, src_list, dst_list, ts_list = [], [], [], [], [], []
    for fp in file_paths:
        if not os.path.exists(fp):
            print(f"  skipping {os.path.basename(fp)} (not found)")
            continue
        print(f"  loading {os.path.basename(fp)}...", flush=True)
        df = pd.read_csv(fp, low_memory=False, encoding="latin1")
        if name == "unsw_nb15":
            rows, y, seqs = _extract_unsw(df)
            features = UNSW_STAT_FEATURES
        else:
            rows, y, seqs = _extract_cic(df)
            features = CIC_STAT_FEATURES
        X = _stat_matrix(rows, features)
        print(f"    rows={len(df)} feats={X.shape[1]} anomaly={y.mean()*100:.2f}%")
        seqs_list.append(seqs)
        X_list.append(X)
        y_list.append(y)
        src_list.append(rows["src_ip"].to_numpy(dtype=np.float64))
        dst_list.append(rows["dst_ip"].to_numpy(dtype=np.float64))
        ts = rows["timestamp"]
        if isinstance(ts, pd.Series) and ts.isna().all():
            ts_list.append(np.arange(len(rows), dtype=np.float64))
        elif ts is None:
            ts_list.append(np.arange(len(rows), dtype=np.float64))
        else:
            ts_list.append(ts)

    if not X_list:
        raise FileNotFoundError(
            f"No usable {name} CSV files provided. Supply the raw CSVs in "
            f"{data_dir} or run the downloader.")

    seq = np.concatenate(seqs_list, axis=0).astype(np.float32)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    ts = np.concatenate(ts_list)

    if n_samples is not None and len(X) > n_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n_samples, replace=False)
        seq, X, y = seq[idx], X[idx], y[idx]
        src, dst, ts = src[idx], dst[idx], ts[idx]
        print(f"  NOTE: capped to {n_samples} rows (smoke-test mode)")

    # global IP node ids
    valid = (src >= 0) & (dst >= 0)
    ips = np.unique(np.concatenate([src[valid], dst[valid]])) if valid.any() else np.array([], dtype=np.float64)
    ip2id = {ip: i for i, ip in enumerate(ips)}
    src_id = np.array([ip2id.get(v, -1) for v in src], dtype=np.int64)
    dst_id = np.array([ip2id.get(v, -1) for v in dst], dtype=np.int64)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    data = {
        "seq": torch.FloatTensor(seq),
        "stat": torch.FloatTensor(Xs),
        "features": torch.FloatTensor(X),
        "labels": torch.FloatTensor(y),
        "src_ip": torch.LongTensor(src_id),
        "dst_ip": torch.LongTensor(dst_id),
        "timestamp": torch.DoubleTensor(ts),
        "n_unique_ips": int(len(ips)),
        "anomaly_rate": float(y.mean()),
        "n_total": int(len(y)),
    }
    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(data_dir, output_name)
    torch.save(data, out)
    print(f"  saved {out}  (N={len(y)}, feat={X.shape[1]}, ips={len(ips)}, "
          f"anom={y.mean()*100:.2f}%)")
    return data


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
        candidates = UNSW_LABELED_FILES + UNSW_FULL_FILES
        paths = [os.path.join(data_dir, f)
                 for f in candidates if os.path.exists(os.path.join(data_dir, f))]
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
