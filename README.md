# MGNN — Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection

Official implementation of MGNN, a multi-view graph neural network that detects
coordinated attacks in encrypted traffic by fusing three complementary signals:

- **Sequence view** — a bidirectional LSTM encoder over the packet sequence
  (length, direction, inter-arrival time), max-pooled over time, sequences
  truncated at 100 packets.
- **Statistical view** — a two-layer MLP with batch normalization over
  23-dimensional distributional flow features.
- **Interaction view** — a 2-layer, 4-head GAT with type-specific linear
  projections over a heterogeneous flow/IP graph. The graph has three edge
  types (temporal, volume, communication), all built with data-driven
  thresholds.

Cross-view attention scores each view with a single linear layer, activates the
scores with tanh and applies softmax to obtain per-flow fusion weights, enabling
post-hoc interpretability.

## Results

On three public benchmarks (21.6M flows, 38 attack categories) MGNN achieves
**F1 = 98.5% / 96.8% / 96.1%** on CIC-IDS2017 / UNSW-NB15 / CSE-CIC-IDS2018,
outperforming the strongest baseline (GDN) by 5.4–12.1 F1 points, with FPR 0.6%
on CIC-IDS2017 and 0.8–0.9% on the other two datasets, and processes **1.2M
flows/s** on a single NVIDIA A100 GPU.

## Training Objective

The total loss combines three terms (paper, Sec. 4.5):

```text
L = L_BCE + 0.1 * L_align + 0.3 * L_reg
```

- **`L_BCE`** — binary cross-entropy over the anomaly logit; benign flows are
  undersampled to a 1:3 anomaly:benign ratio in the training split.
- **`L_align`** — alignment loss (paper, Eq. 7): the sum over view pairs of the
  squared L2 distance between unit-normalized view embeddings, preventing
  representational collapse.
- **`L_reg`** — a regularization term promoting a smooth local decision
  boundary. The full loss achieves MCC = 0.93 and AUROC = 0.97 on CIC-IDS2017.
  Adam uses weight decay 1e-5.

## Experimental Protocol

- **Data splits** — chronological: first 60% of flows (by time) train, next 20%
  validate, last 20% test. Preserves temporal ordering and prevents leakage.
- **Undersampling** — training benign flows reduced to a 1:3 anomaly:benign
  ratio; test set retains the original distribution.
- **Optimizer** — Adam (lr=1e-3, weight_decay=1e-5), batch size 2048, early
  stopping with patience 10.
- **Runs** — 5 independent runs, seeds {42, 0, 123, 7, 2024}, reported as
  mean ± std.
- **GAT encoder** — 2 layers, 4 attention heads, hidden 128.
- **Hardware** — NVIDIA A100 80 GB, bf16 automatic mixed precision.

## Reproducing the Paper's Tables

| Paper table | Experiment script | Output |
|-------------|-------------------|--------|
| Table 2/3   | `experiments/run_table2.py` | Main results: 10 baselines + MGNN on 3 datasets |
| Table 4     | `experiments/run_table4.py` | Cross-dataset generalization (zero-shot) |
| Table 6     | `experiments/run_table6.py` | Fusion strategy (concat / avg / attn / attn_align) |
| Table 5     | `experiments/run_ablation.py --view` | View and edge ablation |
| Table 7     | `experiments/run_ablation.py --loss` | Loss-component ablation |
| Table 8     | `experiments/run_table8.py` | SOTA comparison on CIC-IDS2017 (12 baselines) |
| —           | `experiments/bench_a100.py` | A100 kernel-config / throughput benchmark |

## Datasets

| Dataset          | Flows   | Attack types | Anom.% | Source |
|------------------|---------|--------------|--------|--------|
| CIC-IDS2017      | 2.83M   | 14           | 19.7   | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) |
| UNSW-NB15        | 2.54M   | 9            | 44.9   | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| CSE-CIC-IDS2018  | 16.23M  | 15           | 16.3   | [UNB S3](https://cse-cic-ids2018.s3.amazonaws.com/) |

## Requirements

- Python 3.9+
- PyTorch >= 2.1, PyTorch Geometric >= 2.4
- numpy, pandas, scikit-learn, scipy, xgboost, lightgbm

```bash
pip install -r requirements.txt
```

## Reproducing the Experiments

### 1. Prepare real data

```python
from src.data.dataset import prepare_cic_ids2017, prepare_unsw_nb15, prepare_cse_ids2018
prepare_cic_ids2017("data")   # -> data/cic_ids2017.pt
prepare_unsw_nb15("data")     # -> data/unsw_nb15.pt
prepare_cse_ids2018("data")   # -> data/cse_ids2018.pt
```

Downloads and preprocesses each dataset into a single `.pt` per dataset
(statistical view, sequence view, labels, IP/timestamp graph fields).

### 2. Run the experiments

```bash
# Fusion comparison (paper, Table 6)
python experiments/run_table6.py --data-dir data --runs 5 --epochs 30 --amp

# Main results over 10 baselines + MGNN on all datasets (paper, Table 2/3)
python experiments/run_table2.py --data-dir data --runs 5 --epochs 100

# SOTA comparison on CIC-IDS2017, 12 baselines (paper, Table 8)
python experiments/run_table8.py --data-dir data --dataset cic_ids2017 --runs 5 --epochs 100

# Cross-dataset generalization (paper, Table 4)
python experiments/run_table4.py --data-dir data --runs 5 --epochs 30

# View and loss abalations on CIC-IDS2017 (paper, Table 5 / Table 7)
python experiments/run_ablation.py --data-dir data --runs 5 --epochs 30 --view
python experiments/run_ablation.py --data-dir data --runs 5 --epochs 30 --loss
```

### 3. Cluster (Slurm) runs

`cluster/` contains a single-node workflow for data preparation and the
experiments:

```bash
bash cluster/setup_env.sh                 # create venv, install pinned deps
bash cluster/download_prep.sh             # download + preprocess all datasets
sbatch cluster/slurm_run_table6.sub       # fusion comparisons on an A100 node
sbatch cluster/slurm_run_table8.sub       # SOTA comparison
```

## Directory Layout

```
MGNN/
├── src/
│   ├── data/    dataset.py (real preprocessing), graph.py (heterogeneous graph)
│   ├── models/  mgnn.py, baselines.py (GCN/SAGE/GAT/E-SAGE/GDN/FN-GNN/DNN/...)
│   └── utils/   config.py, metrics.py, split.py, training.py
├── experiments/ run_table2.py, run_table4.py, run_table6.py, run_table8.py,
│                run_ablation.py, run_pipeline.py, bench_a100.py
├── cluster/     Slurm + environment scripts
├── requirements.txt, README.md, LICENSE
```

## Citation

```text
@misc{mgnn2026,
  title={MGNN: Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection},
  author={Ding, Wei and Zhou, Run and Liao, Rong and Wang, Yuxiang and Fan, Rui
          and Yan, Ruiyang and Hao, Shuang and Yin, Feifei and Cao, Di},
  howpublished={GitHub repository},
  year={2026},
  url={https://github.com/code-2026l/MGNN}
}
```

## License

Released under the [MIT License](LICENSE).