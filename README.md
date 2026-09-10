# MGNN — Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection

Official implementation of MGNN, a multi-view graph neural network that
detects coordinated attacks in encrypted traffic by fusing three
complementary signals:

- **Sequence view** — a bidirectional LSTM encoder over the packet sequence
  (length, direction, inter-arrival time), with max-pooled hidden states and
  sequences truncated at 100 packets.
- **Statistical view** — a two-layer MLP with batch normalization over
  23-dimensional distributional flow features.
- **Interaction view** — a 2-layer, 4-head GAT with type-specific linear
  projections over a heterogeneous flow/IP graph whose three edge types
  (temporal, volume, communication) are constructed with data-driven
  thresholds.

A cross-view attention mechanism scores each view with a linear + tanh layer
and applies softmax to obtain per-flow fusion weights, enabling post-hoc
interpretability.

## Training Objective

The total loss combines three terms (paper, Sec. 4.5):

```text
L = L_BCE + lambda_align * L_align + lambda_reg * L_reg
          (lambda_align = 0.1)        (lambda_reg  = 0.3)
```

- **`L_BCE`** — binary cross-entropy over the anomaly logit. Benign flows
  are undersampled to a 1:3 anomaly:benign ratio in the training split.
- **`L_align`** — alignment loss: the sum over view pairs of the squared L2
  distance between unit-normalized view embeddings, preventing
  representational collapse while preserving discriminative differences.
- **`L_reg`** — a regularization term promoting a smooth local decision
  boundary, measured as the squared gradient norm of the squared logit with
  respect to the fused representation. Distinct from the Adam L2 weight
  decay (`1e-5`).

## Results

On three public benchmarks (21.6M flows, 38 attack categories) MGNN achieves
F1 = 98.5% / 96.8% / 96.1% on CIC-IDS2017 / UNSW-NB15 / CSE-CIC-IDS2018,
outperforming the strongest baseline by 5.4–12.1 F1 points, with FPR 4.2% on
CIC-IDS2017 and 0.8–0.9% on the other two datasets, and processes 1.2M
flows/s on a single A100 GPU.

## Requirements

- Python 3.9+
- PyTorch >= 2.1, PyTorch Geometric >= 2.4
- numpy, pandas, scikit-learn, scipy, xgboost, lightgbm

```bash
pip install -r requirements.txt
```

## Datasets

| Dataset        | Flows   | Attack types | Source |
|----------------|---------|--------------|--------|
| CIC-IDS2017    | 2.83M   | 14           | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) |
| UNSW-NB15      | 2.54M   | 9            | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| CSE-CIC-IDS2018| 16.23M  | 15           | [UNB S3](https://cse-cic-ids2018.s3.amazonaws.com/) |

## Reproducing the Experiments

### 1. Prepare real data

```python
from src.data.dataset import prepare_cic_ids2017, prepare_unsw_nb15, prepare_cse_ids2018
prepare_cic_ids2017("data", download=True)   # -> data/cic_ids2017.pt
prepare_unsw_nb15("data", download=True)     # -> data/unsw_nb15.pt
prepare_cse_ids2018("data", download=True)   # -> data/cse_ids2018.pt
```

### 2. Run the experiments

```bash
# Fusion strategy comparison (concat / avg / attn / attn_align) — paper protocol:
# chronological 60/20/20 split, 1:3 undersampling, batch 2048, early stopping
# patience 10, seeds {42,0,123,7,2024}, bf16 AMP on an A100
python experiments/run_table6.py --data-dir data --runs 5 --epochs 30 --amp

# SOTA comparison on CIC-IDS2017 (paper, Table 8, chronological 70/30 split):
# RF, XGBoost, LightGBM, LSTM, CNN-LSTM, DNN, BiLSTM,
# GAT, GCN, E-GraphSAGE, FN-GNN, GDN
python experiments/run_table3.py --data-dir data --dataset cic_ids2017 --runs 5 --epochs 30
```

### 3. Cluster (Slurm) runs

`cluster/` contains a single-node workflow for data preparation and the
experiments:

```bash
bash cluster/setup_env.sh        # create venv, install pinned deps
bash cluster/download_prep.sh    # download + preprocess all three datasets
sbatch cluster/slurm_run_table6.sub   # fusion comparisons on an A100 node
sbatch cluster/slurm_run_table3.sub   # SOTA comparison
```

## Directory Layout

```
MGNN/
├── src/
│   ├── data/    dataset.py (real preprocessing), graph.py (heterogeneous graph)
│   ├── models/  mgnn.py, baselines.py (GCN/SAGE/GAT/E-SAGE/GDN/FN-GNN/DNN/LSTM...)
│   └── utils/   config.py, metrics.py, training.py
├── experiments/ run_table6.py, run_table3.py, run_pipeline.py, bench_a100.py
├── cluster/     Slurm + environment scripts
├── requirements.txt, README.md, LICENSE
```

## Citation

```
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
