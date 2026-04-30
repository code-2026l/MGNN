# MGNN: Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection

This repository contains the source code for the IEEE conference paper:

> **MGNN: Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection**

## Paper

The LaTeX source and all figures are provided for reproducibility.

### Compile

```bash
pdflatex paper.tex
pdflatex paper.tex
```

Requires `IEEEtran.cls` (included) and standard LaTeX packages.

## Repository Structure

```
MGNN/
|-- paper.tex                        # Main LaTeX source
|-- figures/                         # All figures used in the paper
|   |-- fig_architecture.pdf         # MGNN architecture overview
|   |-- fig_roc_curves.pdf           # ROC curves on three benchmarks
|   |-- fig_ablation_study.pdf       # Ablation study (component, fusion, loss, edge type)
|   |-- fig_attention_analysis.pdf   # Attention weight analysis
|   |-- fig_per_attack_comparison.pdf # Per-attack F1 comparison
|   |-- fig_sota_tradeoff.pdf        # SOTA F1-FPR trade-off comparison
|   |-- fig_hyperparameter_sensitivity.pdf # Hyperparameter sensitivity
|   |-- fig_scalability.pdf          # Efficiency and scalability
|   |-- fig_training_convergence.pdf # Training convergence
|   |-- fig_tsne_visualization.pdf   # t-SNE embedding visualization
|-- IEEEtran/                        # IEEE conference template files
```

## Datasets

Experiments use the following publicly available benchmarks:

- **CIC-IDS2017**: [Sharafaldin et al., 2018](https://www.unb.ca/cic/datasets/ids-2017.html)
- **UNSW-NB15**: [Moustafa & Slay, 2015](https://www.unsw.edu.au/unsw-canberra-cyber/cybersecurity-and-critical-infrastructure/research-areas/unsw-nb15-dataset)
- **CSE-CIC-IDS2018**: [Sharafaldin et al., 2018](https://www.unb.ca/cic/datasets/ids-2018.html)

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{anonymous2025mgnn,
  title={MGNN: Multi-View Graph Neural Network for Encrypted Traffic Anomaly Detection},
  author={Anonymous},
  booktitle={IEEE Conference},
  year={2025}
}
```
