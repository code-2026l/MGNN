# MGNN: Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection

Official implementation of the paper:

> **MGNN: Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection**

MGNN detects coordinated attacks in encrypted traffic by fusing three views:
- **Sequence view**: BiLSTM encoder over packet-length sequences
- **Statistical view**: MLP encoder over distributional flow features
- **Interaction view**: GAT encoder over a k-NN graph of flow embeddings

## Repository Structure

```
MGNN/
├── src/                          # Core library
│   ├── models/
│   │   ├── mgnn.py              # MGNN: three-view fusion with GAT
│   │   └── baselines.py         # GCN, GraphSAGE, EGraphSAGE
│   ├── data/
│   │   ├── dataset.py           # Dataset download & CSV preprocessing
│   │   └── synthetic.py         # Synthetic data generators
│   └── utils/
│       ├── config.py            # Argument parsing & device setup
│       ├── metrics.py           # Evaluation: P/R/F1/FPR/MCC/AUC
│       └── training.py          # Training & evaluation loops
├── experiments/
│   ├── run_table3.py            # Table 3: Graph model comparison
│   ├── run_table6.py            # Table 6: Fusion strategy comparison
│   └── run_pipeline.py          # End-to-end: download + preprocess + both tables
├── figures/                     # Paper figures
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch >= 2.0
- NumPy, Pandas, scikit-learn, SciPy
- (Optional) PyTorch Geometric >= 2.4 for graph-based models (GCN, GraphSAGE, GAT)

Install dependencies:

```bash
pip install -r requirements.txt
```

For graph models, additionally install PyTorch Geometric following the
[official instructions](https://pytorch-geometric.readthedocs.io/).

## Datasets

Experiments use three publicly available benchmark datasets:

| Dataset | Size (flows) | Attack types | Source |
|---------|-------------|--------------|--------|
| CIC-IDS2017 | 2.83M | 15 | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) |
| UNSW-NB15 | 257K | 9 | [UNSW](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/) |
| CSE-CIC-IDS2018 | 16.23M | 14 | [UNB](https://www.unb.ca/cic/datasets/ids-2018.html) |

### Automatic download

```bash
python experiments/run_pipeline.py
```

This downloads CIC-IDS2017 CSV files, preprocesses them into `.pt` format,
and runs both Table 3 and Table 6 experiments.

### Manual data preparation

1. Download CSVs from the links above and place them in `data/raw/`
2. Run preprocessing:
   ```python
   from src.data.dataset import preprocess_cic_csv
   data = preprocess_cic_csv(file_paths, data_dir='data')
   ```

Alternatively, place preprocessed `.pt` files directly in `data/`.

## Usage

### Run all experiments

```bash
python experiments/run_pipeline.py --all
```

### Run specific tables

```bash
# Fusion strategy comparison (Table 6)
python experiments/run_table6.py --runs 5 --epochs 30

# Graph model comparison (Table 3, uses synthetic graph data)
python experiments/run_table3.py --runs 5 --epochs 100
```

### Command-line options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `./data` | Dataset directory |
| `--device` | auto | `cuda:0` or `cpu` |
| `--runs` | 5 | Number of repeated runs |
| `--epochs` | 30 | Training epochs |
| `--batch-size` | 256 | Training batch size |
| `--hidden` | 128 | Hidden dimension |
| `--skip-download` | False | Skip dataset download |

## Model Architecture

MGNN fuses three heterogeneous views:

1. **Sequence view**: BiLSTM processes packet-length sequences (max-over-time pooling)
2. **Statistical view**: MLP encodes 23-dimensional flow statistics
3. **Interaction view**: Multi-head GAT on k-NN graph of flow features

Four fusion strategies are supported:
- `concat`: Concatenate all three view representations
- `avg`: Average all three views
- `attn`: Learnable cross-view attention weights
- `attn_align`: Attention fusion + alignment loss (paper's primary config)

## Citation

```bibtex
@inproceedings{mgnn2025,
  title={MGNN: Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection},
  author={Anonymous},
  booktitle={IEEE Conference},
  year={2025}
}
```

## License

This project is released under the MIT License.
