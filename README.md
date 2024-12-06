# SC4VR: Supervised Contrastive Learning for Video Representations

SC4VR (Supervised Contrastive Learning for Video Representations) is a novel approach designed to enhance video representation learning, specifically for crash and near-crash events. This repository contains the codebase and documentation for implementing SC4VR.

## Features

- **Supervised Contrastive Learning**: Enhances intra-class similarity and inter-class separability in video representations.
- **Video Swin Transformer Backbone**: Utilized for robust feature extraction from video data.
- **Class Imbalance Handling**: Incorporates weighted loss functions to improve detection of minority classes (e.g., rare crash events).


## Methodology

SC4VR leverages supervised contrastive learning to align representations of similar events (e.g., crash videos) while ensuring distinct separation from other classes (e.g., near-crash and baseline events). Key steps include:

1. **Representation Learning**: Training a Video Swin Transformer backbone with a contrastive loss function.
2. **Clustering Analysis**: Using metrics like the Silhouette Score (SS) to assess the alignment of learned representations with ground-truth labels.
3. **Classification Performance**: Evaluating representations through a classification task with comprehensive metrics.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SC4VR.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Citation

If you use this repository in your research, please cite the following:

```bibtex
@phdthesis{shi2024enhanced,
  author       = {Jiang, Boyu; Shi, liang; Guo, Feng},
  title        = {SC4VR: A Supervised Contrastive Learning Framework for Vision Representation of Traffic Safety Critical Events},
  year         = {2025},
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

