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
MAGNN-Github/
|-- paper.tex              # Main LaTeX source
|-- figures/               # All figures used in the paper
|   |-- architecture.pdf   # MGNN architecture overview
|   |-- roc.pdf            # ROC curves on three benchmarks
|   |-- fig3_nature_final.pdf  # Case study: heterogeneous graph
|   |-- attention.pdf      # Attention weight analysis
|   |-- per_attack.pdf     # Per-attack F1 comparison
|   |-- sota_tradeoff.pdf  # SOTA trade-off comparison
|   |-- dimension.pdf      # Hyperparameter sensitivity
|   |-- scalability.pdf    # Efficiency and scalability
|   |-- convergence.pdf    # Training convergence
|   |-- tsne.pdf           # t-SNE visualization
|-- IEEEtran/              # IEEE conference template files
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
