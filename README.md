# **Land-Cover Mapping with Geospatial Foundation Models**

Problem Statement

This project investigates whether geospatial foundation models (GFMs)—large
self-supervised transformers pretrained on global multispectral archives—can
significantly improve the accuracy, label efficiency, and generalization of
land-cover mapping from satellite imagery. Traditional land-cover classification
requires extensive labeled datasets and regional models, limiting update
frequency and global consistency. By fine-tuning GFMs on small labeled subsets
of Sentinel-2 data, the project aims to determine how much labeling effort can
be reduced while maintaining or improving classification performance. The
results are relevant for environmental monitoring, agriculture, climate
modeling, and urban-planning systems that depend on timely, high-resolution
land-cover maps.

Input and Output Data Format

**Input:**

- Multispectral Sentinel-2 Level-2A tiles (13 spectral bands, 10–20 m
  resolution).
- Preprocessed into georeferenced image patches (e.g., 256×256 pixels),
  cloud-masked and normalized.

**Output:**

- Pixel-wise land-cover segmentation maps aligned with the spatial grid of the
  input.
- Class labels follow the ESA WorldCover taxonomy (e.g., built-up, cropland,
  forest, shrubland, grassland, water, wetlands, bare soil).

The system processes batches of image tensors (C×H×W) and outputs segmentation
logits for each class.

Metrics

Primary metrics:

- **mIoU (mean Intersection over Union):** standard for segmentation; measures
  overlap per class.
- **Macro F1-score:** captures performance balance across classes, especially
  minority classes.
- **Overall Accuracy:** useful but secondary due to class imbalance.

Secondary metrics:

- **Label-efficiency curves:** performance vs. training label volume.
- **Cross-region generalization performance:** trained on region A, tested on
  region B.

Expected values depend on AOI; typical benchmarks for 10 m land-cover
classification:

- mIoU: 0.50–0.70
- F1: 0.60–0.80

GFMs are expected to outperform non-pretrained baselines by \+5–15 mIoU in
low-label regimes.

Validation

- **Train/val/test split** based on geographically disjoint Areas of Interest
  (AOIs) to prevent spatial leakage.
- **Reproducibility:** fixed random seeds, deterministic dataloader settings,
  logged preprocessing pipeline, explicit STAC queries for data retrieval.
- **Cross-region validation:** train on one region (e.g., The Netherlands), test
  on a different region (e.g., Portugal) to evaluate generalization.

All dataset splits and configuration files will be versioned in Git for exact
reproducibility.

Data

**Data sources (fully open):**

- **Sentinel-2 L2A** (10–20 m multispectral imagery) Copernicus Data Space:
  [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)
- **ESA WorldCover 2021/2023** (10 m land-cover labels)
  [https://worldcover2021.esa.int/](https://worldcover2021.esa.int/)
- **Optionally:** CORINE Land Cover for Europe
  [https://land.copernicus.eu/pan-european/corine-land-cover](https://land.copernicus.eu/pan-european/corine-land-cover)

**Features and potential issues:**

- Cloud coverage → mitigated through SCL mask and QA60 confidence layers.
- Seasonal variability → optional temporal sampling.
- Class imbalance (e.g., small percentage of built-up areas).
- Scene heterogeneity between regions may cause domain shift.

Tile lists and preprocessing scripts will be provided for traceability.

Modeling

Baseline

The baseline system is a **lightweight UNet** trained from scratch on Sentinel-2
patches using ESA WorldCover labels. This provides a simple and well-understood
benchmark. Alternative baselines include:

- **ResNet-50** classifier applied patch-wise.
- **Vision Transformer Small (ViT-S)** without pretraining.

These baselines quantify how much benefit GFMs bring beyond classical models.

Main model

The main methods are **geospatial foundation models**, specifically:

1. **Prithvi-100M / Prithvi-300M / Prithvi 2.0 (NASA)**
   - Transformer encoder pretrained on global multispectral data using masked
     autoencoding.
   - GitHub: https://github.com/nasa-nccs/prithvi
   - Paper: “Prithvi: Foundation Models for Earth Observation”.
2. **RS-MAE (Masked Autoencoder for Remote Sensing)**
   - Pretrained on large Sentinel-2 archives.
   - GitHub: https://github.com/ZhengZixiang/RS-MAE
   - Paper: “Masked Autoencoders for Remote Sensing”.

These models will be fine-tuned for pixel-wise segmentation using a small
labeled dataset. Training includes:

- AdamW optimizer
- Linear warmup and cosine decay
- Mixed-precision training
- Experiment tracking with Weights & Biases or MLflow

The final comparison will quantify improvements in accuracy, label-efficiency,
and cross-region robustness.

## Environment Setup

This project uses **Conda for system/compiled dependencies** and **uv.lock for
PyPI packages** to ensure reproducible environments across macOS and Linux.

### Prerequisites

- **Conda**: Install [Miniforge](https://github.com/conda-forge/miniforge)
  (recommended) or Miniconda
  - macOS: `brew install miniforge`
  - Linux:
    `curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash Miniforge3-Linux-x86_64.sh`
- **uv**: Fast Python package installer
  - Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Setup Commands

#### macOS Development Setup

For local development on macOS (includes dev tools like Jupyter, pytest,
pre-commit):

```bash
./scripts/setup_dev.sh
conda activate gfm
```

Or using the base script with flags:

```bash
./scripts/setup_env.sh --dev
conda activate gfm
```

#### Linux Training Setup

For Linux training environments (includes training dependencies and GPU
support):

```bash
# Standard setup (may update dependencies)
./scripts/setup_train_linux.sh

# Frozen setup (exact reproduction, recommended for remote training)
./scripts/setup_train_linux.sh --frozen

conda activate gfm
```

Or using the base script:

```bash
./scripts/setup_env.sh --train --frozen
conda activate gfm
```

#### Advanced Options

The base `setup_env.sh` script supports multiple flags:

```bash
# Flags:
#   --dev              Install development dependencies
#   --train            Install training dependencies (+ GPU on Linux)
#   --refresh-conda    Force recreate conda environment from scratch
#   --lock             Regenerate uv.lock from pyproject.toml
#   --frozen           Frozen install (no dependency changes, for CI/remote)

# Examples:
./scripts/setup_env.sh --dev --train         # Dev + training
./scripts/setup_env.sh --refresh-conda       # Fresh conda install
./scripts/setup_env.sh --lock                # Update lockfile
./scripts/setup_env.sh --train --frozen      # Reproducible remote training
```

### Architecture

**Conda environments** (`environment.{macos,linux}.yml`):

- Python 3.11
- Geospatial libraries: GDAL, rasterio, fiona, proj, geos, pyproj, shapely
- PyTorch with platform-specific optimizations:
  - macOS: CPU/MPS support
  - Linux: CUDA 12.1 support

**PyPI packages** (`uv.lock`):

- Base runtime dependencies (PyTorch Lightning, Hydra, MLflow, etc.)
- Optional dependency groups:
  - `dev`: Development tools (pytest, Jupyter, pre-commit, DVC, etc.)
  - `train`: Training utilities (ONNX export, etc.)
  - `train-gpu`: GPU training utilities (Linux-only)

### Remote Training Workflow

For reproducible training on remote Linux machines:

1. **Local**: Commit your code and `uv.lock` to git
2. **Remote**: Clone repository
3. **Remote**: Run frozen setup:
   ```bash
   ./scripts/setup_train_linux.sh --frozen
   conda activate gfm
   ```
4. The `--frozen` flag ensures exact dependency reproduction without drift

### Updating Dependencies

To update the lockfile after modifying `pyproject.toml`:

```bash
./scripts/setup_env.sh --lock
git add uv.lock
git commit -m "Update dependencies"
```

### Troubleshooting

**Conda environment conflicts:**

```bash
# Force refresh the conda environment
./scripts/setup_env.sh --refresh-conda --dev
```

**TensorRT (Linux only):** TensorRT and torch-tensorrt are automatically
installed via uv on Linux when using `--train`. This is optional for GPU
inference optimization. If installation fails, you can skip it or install
manually:

```bash
conda activate gfm
uv pip install tensorrt torch-tensorrt
```

**Import errors after setup:** Ensure you've activated the environment:

```bash
conda activate gfm
```

---

## Deployment REST service for inference

# Training Data Contract (Frozen)

## Input Artifact

- CSV: `dataset_index_with_split.csv`
- One row = one training sample (patch)
- File is **read-only and immutable**

---

## Required Columns & Semantics

### Identity & Leakage Control

- `patch_id`: unique sample ID
- `tile_id`: parent Sentinel-2 tile
- `group_id`: grouping key **Invariant:** no `group_id` appears in more than one
  split

### Split Assignment (Authoritative)

- `split ∈ {train, val, test, ood}`
- Splits are **pre-assigned and frozen**
- **OOD is fully held out**

**OOD MUST NOT be used for:**

- normalization statistics
- class weights
- early stopping
- threshold tuning

---

### Data Paths

- `spectral_path`: 4-band S2 patch (B02, B03, B04, B08)
- `label_path`: pixel-aligned ESA WorldCover labels
- Paths are relative and must exist

---

### Quality & Filtering

- `cloud_frac`: primary quality signal
- `valid_frac`: always `1.0` (ignored)
- `is_usable`: ignored

**Filtering (Dataset-level only):**

- train / val: `cloud_frac ≤ 0.20`
- test: fixed per experiment
- ood: **no filtering**

CSV must **never** be modified by filtering.

---

### Diagnostic-Only Metadata (NOT used in training)

- Geometry: `row_off`, `col_off`, `patch_size`, `stride`, `center_x`, `center_y`
- Labels: `dominant_class`, `dominant_frac`, `unique_classes`
- Acquisition: `acq_start`, `acq_end`, `mosaic_method`
- Region: `country`, `aoi_id`

---

## Tensor Contract

**Input**

- `x`: `FloatTensor [4, 256, 256]`
- Channels: `[B02, B03, B04, B08]`
- Normalized using frozen IID-train stats

**Target**

- `y`: `LongTensor [256, 256]`
- Integer WorldCover class IDs

**Invariant**

- `x` and `y` are pixel-aligned

---

## Non-Assumptions

Training code must NOT assume:

- spatial adjacency
- class balance
- country balance
- cloud-free OOD data
- usability from `is_usable`

---

## Definition of Done

- Contract documented
- Dataset uses only allowed fields
- Runtime assertions enforce invariants
- No implicit data-dependent logic
