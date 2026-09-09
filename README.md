# MGNN — Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection

Official implementation accompanying the paper submitted to *Expert Systems*
(Wiley), Special Issue "Computer Applications Frontiers".

MGNN detects coordinated attacks in encrypted traffic by integrating three
complementary signals:

- **Sequence view** — a bidirectional LSTM encoder over packet-length sequences
  (max-pooled hidden states; sequences truncated at 100 packets).
- **Statistical view** — a two-layer MLP (ReLU + batch normalization) over
  23-dimensional distributional flow features.
- **Interaction view** — a 2-layer, 4-head GAT over a heterogeneous flow/IP graph
  with temporal, volume, and communication edges (all thresholds data-driven).

A cross-view attention mechanism scores each view with a linear + tanh layer and
applies softmax to obtain per-flow fusion weights, enabling post-hoc
interpretability.

## Training Objective

The total loss combines three terms:

```text
L = L_BCE + lambda_align * L_align + lambda_reg * L_reg
          (lambda_align = 0.1)        (lambda_reg  = 0.3)
```

- **`L_BCE`** — binary cross-entropy over the anomaly logit. Benign flows are
  undersampled to 3x the anomaly count in every training split (1:3
  anomaly-to-benign ratio; on CIC-IDS2017 this reduces the benign set from
  2.27M to 0.85M flows).
- **`L_align`** — the alignment loss, which pulls normalized view embeddings
  toward a common direction (inter-view cosine similarity rises from 0.34 to
  0.71), preventing representational collapse. It is an explicit sum over view
  pairs of the squared L2 distance between unit-normalized embeddings.
- **`L_reg`** — a regularization term that penalizes decision-boundary curvature
  (measured as decision margin: 0.42 with `L_reg` vs. 0.31 without), keeping the
  boundary smooth and the margin wide. It is distinct from the Adam L2 weight
  decay (`1e-5`).

`lambda_align` and `lambda_reg` are tuned by grid search on the validation set
(`lambda_align in {0.01, 0.05, 0.1, 0.5}`, `lambda_reg in {0.1, 0.2, 0.3, 0.5}`).

## Implementation Notes

- **Real data only.** `src/data/dataset.py` preprocesses the three real
  benchmark datasets (CIC-IDS2017, UNSW-NB15, CSE-CIC-IDS2018) from their
  official CSVs; there is no synthetic fallback.
- **A100-optimised.** Training uses bf16 automatic mixed precision (tensor-core
  speedup, near-lossless). `experiments/bench_a100.py` measures the kernel
  configs; `torch.compile` is opt-in (`--compile`) because on the dynamic k-NN
  interaction graph it is ~5% slower than eager. Single-pass forward returns
  logits, the fused representation, and the views together, so the encoders are
  computed once per step.
- The interaction encoder decouples from PyTorch Geometric only if PyG is
  unavailable; the graph baselines (`run_table3.py`) require PyG.

## Requirements

- Python 3.9+
- PyTorch >= 2.0, **PyTorch Geometric >= 2.4**
- numpy, pandas, scikit-learn, scipy

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
# Fusion strategy comparison (MGNN) — real data, fixed 60/20/20 split,
# seeds {42,0,123,7,2024}, bf16 AMP on an A100
python experiments/run_table6.py --data-dir data --runs 5 --epochs 30 --amp

# Graph baseline comparison on a real k-NN flow graph
python experiments/run_table3.py --data-dir data --dataset cic_ids2017 --runs 5 --epochs 200 --k 5
```

Settings follow the paper: a 60/20/20 stratified split, benign flows
undersampled to 1:3 during training with validation/test keeping the original
class distribution, and Adam (`lr=1e-3`, `weight_decay=1e-5`, `batch_size=2048`).

### 3. Cluster (Slurm) runs

`cluster/` contains a single-node workflow for the data-prep and the experiments:

```bash
bash cluster/setup_env.sh        # create venv, install pinned deps
bash cluster/download_prep.sh    # download + preprocess all three datasets
sbatch cluster/slurm_run_table6.sub   # fusion comparisons on an A100 node
sbatch cluster/slurm_run_table3.sub   # graph baseline comparisons
```

## Directory Layout

```
MGNN/
├── src/
│   ├── data/    dataset.py (real preprocessing), graph.py, synthetic.py
│   ├── models/  mgnn.py, baselines.py (GCN/GraphSAGE/E-GraphSAGE)
│   └── utils/   config.py, metrics.py, training.py
├── experiments/ run_table6.py, run_table3.py, bench_a100.py
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
  note={Under review at Expert Systems},
  url={https://github.com/code-2026l/MGNN}
}
```

## License

Released under the [MIT License](LICENSE).