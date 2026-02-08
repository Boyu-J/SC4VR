# SC4VR: Supervised Contrastive Learning for Video Representations

Official implementation of Enhancing vision representations for traffic safety-critical events via supervised contrastive learning.

SC4VR (Supervised Contrastive Learning for Video Representations) is a novel approach designed to enhance video representation learning, specifically for crash and near-crash events. This repository contains the codebase and documentation for implementing SC4VR.

supervised contrastive pretraining on safety-critical events → better video/clip representations → downstream gains

## Tutorial

Jupyter notebook



## Quickstart

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Boyu-J/SC4VR.git
   ```
   
2. Install dependencies:
   ```bash
   conda create -n sc4vr python=3.10
   conda activate sc4vr
   pip install -r requirements.txt
   ```

### Data preparation

Where to download datasets (or placeholders).

Expected folder structure.


### Train

   ```bash
   python train.py
   ```

### Inference

   ```bash
   python inference.py
   ```

## Citation

If you use this repository in your research, please cite the following:

```bibtex
@article{JIANG2025468,
title = {Enhancing vision representations for traffic safety-critical events via supervised contrastive learning},
journal = {Journal of Safety Research},
volume = {95},
pages = {468-475},
year = {2025},
issn = {0022-4375},
doi = {https://doi.org/10.1016/j.jsr.2025.10.002},
url = {https://www.sciencedirect.com/science/article/pii/S0022437525001471},
author = {Boyu Jiang and Liang Shi and Feng Guo}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

