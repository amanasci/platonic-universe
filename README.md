# The Platonic Universe: Testing The Platonic Representation Hypothesis With Astronomical Data

This repository contains the code for testing the **Platonic Representation Hypothesis (PRH)** on astronomical data, as described in our paper "The Platonic Universe: Do Foundation Models See the Same Sky?"

## Background & Motivation

The Platonic Representation Hypothesis suggests that neural networks trained with different objectives on different data modalities converge toward a shared statistical model of reality in their representation spaces. As models become larger and are trained on more diverse tasks, they should converge toward a "Platonic ideal" representation of underlying reality.

### Why Astronomy?

Astronomical observations provide an ideal testbed for the PRH because:
- **Shared Physical Origin**: Different astronomical observations (images, spectra, photometry) all emerge from the same underlying physics
- **Multiple Modalities**: We can compare representations across fundamentally different data types (like optical images, infrared images, and spectroscopy)
- **Scale**: Modern astronomical surveys provide the data volume necessary to test convergence across multiple model architectures

Our results show that **larger models exhibit more similar representations**, even when trained across different data modalities. This suggests that astronomical foundation models may be able to leverage pre-trained general-purpose architectures.

## Repository Structure

```
platonic-universe/
├── src/pu/                    # Core package
│   ├── metrics.py            # MKNN similarity metrics
│   ├── preprocess.py         # Data preprocessing for different models
│   └── zoom.py              # Galaxy image resizing utilities
├── scripts/                  # Experiment scripts
│   ├── get_embs.py          # Extract embeddings from HuggingFace models
│   ├── get_astropt_embs.py  # Extract embeddings from AstroPT models
│   ├── get_specformer_embs.py # Extract spectral embeddings
│   ├── run_model.bash       # Batch script for model experiments
│   └── run_astropt.bash     # Batch script for AstroPT experiments
└── pyproject.toml           # Project dependencies
```

## Installation and Usage

1. **Clone the repository:**
```bash
git clone https://github.com/UniverseTBD/platonic-universe.git
cd platonic-universe
```

2. **Install dependencies using uv:**
```bash
pip install uv
uv sync
```

### Quick Start: Running Experiments

The repository provides scripts to test representational alignment across different astronomical datasets and model architectures.

#### 1. Test Vision Models (ViT, DINOv2, ConvNeXtv2, IJEPA)

```bash
# Run all modalities for a specific model
./scripts/run_model.bash vit 0       # ViT on GPU 0
./scripts/run_model.bash dino 1      # DINOv2 on GPU 1
./scripts/run_model.bash convnext 2  # ConvNeXtv2 on GPU 2
./scripts/run_model.bash ijepa 3     # IJEPA on GPU 3
```

#### 2. Test AstroPT Models

```bash
# Run AstroPT across all modalities
./scripts/run_astropt.bash 0  # on GPU 0
```

#### 3. Individual Model Runs

```bash
# Test specific model-modality combinations
uv run scripts/get_embs.py --model vit --mode jwst --num-workers 32
uv run scripts/get_embs.py --model dino --mode legacysurvey --num-workers 32
```

### Supported Models & Datasets

**Models Tested:**
- **Vision Transformers (ViT)**: Base, Large, Huge
- **DINOv2**: Small, Base, Large, Giant
- **ConvNeXtv2**: Nano, Tiny, Base, Large
- **IJEPA**: Huge, Giant
- **AstroPT**: Astronomy-specific transformer (Small, Base, Large)
- **Specformer**: Spectroscopy-specific model

**Astronomical Datasets:**
- **HSC (Hyper Suprime-Cam)**: Ground-based optical imaging (reference baseline)
- **JWST**: Space-based infrared imaging
- **Legacy Survey**: Ground-based optical imaging
- **DESI**: Spectroscopy

### Understanding the Results

The code measures **representational alignment** using the Mutual k-Nearest Neighbour (MKNN) metric:

```python
from pu.metrics import mknn

# Calculate MKNN score between two embedding sets
score = mknn(embeddings_1, embeddings_2, k=10)
print(f"MKNN alignment score: {score:.4f}")
```

**Higher MKNN scores** indicate more similar representations between models or modalities.

## Key Scripts Explained

### `get_embs.py`
Extracts embeddings from HuggingFace vision models across different astronomical datasets:

```bash
uv run scripts/get_embs.py \
    --model vit \       # Model type: vit, dino, convnext, ijepa
    --mode jwst \       # Dataset: jwst, legacysurvey, desi, sdss
    --batch-size 128 \  # Processing batch size
    --num-workers 32 \  # DataLoader workers
    --knn-k 10          # K for MKNN calculation
```

### `get_astropt_embs.py`
Extracts embeddings from AstroPT models (astronomy-specific transformers):

```bash
uv run scripts/get_astropt_embs.py \
    --mode jwst \       # Dataset to compare with HSC
    --batch-size 128 \  # Processing batch size
    --num-workers 32    # DataLoader workers
```

## Contributing

This project is open source under the AGPLv3.

We welcome contributions! Please feel free to open a pull request to:
- Add support for new model architectures
- Include additional astronomical datasets
- Implement alternative similarity metrics
- Improve preprocessing pipelines

We also hang out on the UTBD Discord, [so feel free to reach out there!](https://discord.gg/VQvUSWxnu9)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{utbd2025,
  title={The Platonic Universe: Do Foundation Models See the Same Sky?},
  author={UniverseTBD: Duraphe, Kshitij and Smith, Michael J. and Sourav, Shashwat and Wu, John F.},
  journal={Machine Learning and the Physical Sciences Workshop at NeurIPS},
  year={2025}
}
```
