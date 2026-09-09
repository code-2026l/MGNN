# MGNN — Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection

Official implementation accompanying the paper submitted to *Expert Systems*
(Wiley), Special Issue "Computer Applications Frontiers".

MGNN detects coordinated attacks in encrypted traffic by fusing three views:

- **Sequence view** — BiLSTM encoder over packet-length sequences
- **Statistical view** — MLP encoder over distributional flow features
- **Interaction view** — GAT encoder over a k-NN graph of flow embeddings

Four fusion strategies are supported: `concat`, `avg`, `attn`, `attn_align`.

The objective is the paper's total loss (Eq. 8 in Sec. 5.1):

```text
L = L_BCE + lambda_align * L_align  + lambda_reg * L_reg
          (lambda_align = 0.1)        (lambda_reg  = 0.3)
```

- `L_BCE` — binary cross-entropy on the anomaly logit.
- `L_align` — inter-view alignment loss (applied for the `attn_align` fusion),
  encouraging the three normalized view embeddings to agree.
- `L_reg` — decision-boundary regularization penalizing the local curvature of
  the decision function (squared gradient norm of the logit w.r.t. the fused
  pre-classifier representation), which flattens the boundary and widens the
  margin. This is distinct from the Adam L2 weight decay (`1e-5`) set on the
  optimizer.

Default optimizer is Adam (lr `1e-3`, weight decay `1e-5`), batch size `2048`,
GAT with 2 layers / 4 heads and hidden dim `128`. Experiments use the paper's
five seeds `{42, 0, 123, 7, 2024}` over `5` runs and a 60/20/20 split with the
benign flows undersampled to a 1:3 anomaly-to-benign ratio during training
while validation/test keep the original class distribution.

---

## Important: what this repository actually does

This is a **real-data** implementation. Unlike naively auto-generated code, it
does **not** silently fall back to synthetic data:

- `src/data/dataset.py` preprocesses the **three real benchmark datasets**
  (CIC-IDS2017, UNSW-NB15, CSE-CIC-IDS2018) from their official CSVs.
- `experiments/run_table3.py` trains GCN / GraphSAGE / E-GraphSAGE on a **real
  k-NN flow graph** built from the actual features (`src/data/graph.py`).
- `experiments/run_table6.py` compares the four MGNN fusion strategies on real
  data with a **deterministic split** and fixed seeds. It raises an error if a
  dataset is missing — it never fabricates numbers.

To reproduce the paper tables you must first prepare the real data
(see *Reproducing the experiments*).

## Requirements

- Python 3.9+
- PyTorch >= 2.0, **PyTorch Geometric >= 2.4** (required for the graph
  baselines; the MGNN interaction encoder works without it via a linear
  fallback)
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

## Reproducing the experiments

### 1. Prepare real data

```python
from src.data.dataset import prepare_cic_ids2017, prepare_unsw_nb15, prepare_cse_ids2018
prepare_cic_ids2017("data", download=True)   # -> data/cic_ids2017.pt
prepare_unsw_nb15("data", download=True)     # -> data/unsw_nb15.pt
prepare_cse_ids2018("data", download=True)   # -> data/cse_ids2018.pt
```

Each `prepare_*` accepts raw CSVs already present in `data/` (it keeps
whatever is available), and caps the size only when you pass `n_samples=...`
(explicitly, for smoke tests). By default the **full** labeled set is used.

### 2. Run the experiments

```bash
# Fusion strategy comparison (MGNN) — real data, fixed split, seeds {42,0,123,7,2024}
python experiments/run_table6.py --data-dir data --runs 5 --epochs 30

# Graph baseline comparison on a real k-NN graph
python experiments/run_table3.py --data-dir data --dataset cic_ids2017 --runs 5 --epochs 200 --k 5
```

Seeds, split fractions, `--hidden`, `--batch-size`, and `--test-frac` are all
tunable from the CLI so runs are reproducible.

### 3. Cluster (Slurm) runs

`cluster/` contains a full single-node + data-prep workflow:

```bash
bash cluster/setup_env.sh        # create venv, install pinned deps
bash cluster/download_prep.sh    # download + preprocess all three datasets
sbatch cluster/slurm_run_table6.sub   # fusion comparisons on an A100 node
sbatch cluster/slurm_run_table3.sub   # graph baseline comparisons
```

## Directory layout

```
MGNN/
├── src/
│   ├── data/    dataset.py (real preprocessing), graph.py, synthetic.py
│   ├── models/  mgnn.py, baselines.py (GCN/GraphSAGE/E-GraphSAGE)
│   └── utils/   config.py, metrics.py, training.py
├── experiments/ run_table6.py, run_table3.py
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