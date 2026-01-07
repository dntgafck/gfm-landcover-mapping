# Land-Cover Mapping with Geospatial Foundation Models

## ⚠️ Project Status

| Component                                | Status                 |
| ---------------------------------------- | ---------------------- |
| Data preparation pipeline (DVC)          | ✅ Implemented         |
| Data loading & transforms                | ✅ Implemented         |
| UNet baseline training                   | ✅ Implemented         |
| Model export (ONNX)                      | ✅ Implemented         |
| Inference server (FastAPI)               | ✅ Implemented         |
| **GFM model training (Prithvi, RS-MAE)** | ❌ **Not implemented** |

> **Note:** The original project scope included fine-tuning Geospatial
> Foundation Models (GFMs) such as Prithvi and RS-MAE. This functionality is not
> currently implemented—only the UNet baseline model is available for training.

---

# Part 1: Semantic Description

## Problem Statement

This project investigates whether geospatial foundation models (GFMs)—large
self-supervised transformers pretrained on global multispectral archives—can
significantly improve the accuracy, label efficiency, and generalization of
land-cover mapping from satellite imagery.

Traditional land-cover classification requires extensive labeled datasets and
regional models, limiting update frequency and global consistency. By
fine-tuning GFMs on small labeled subsets of Sentinel-2 data, the project aims
to determine how much labeling effort can be reduced while maintaining or
improving classification performance.

The results are relevant for environmental monitoring, agriculture, climate
modeling, and urban-planning systems that depend on timely, high-resolution
land-cover maps.

## Data

### Input

- **Sentinel-2 Level-2A** multispectral imagery (4 bands: B02, B03, B04, B08)
- Preprocessed into georeferenced image patches (256×256 pixels)
- Cloud-masked and normalized

### Output

- Pixel-wise land-cover segmentation maps
- Class labels follow **ESA WorldCover** taxonomy:
  - Tree cover, Shrubland, Grassland, Cropland, Built-up
  - Bare/sparse vegetation, Snow and ice, Permanent water bodies
  - Herbaceous wetland, Mangroves, Moss and lichen

### Data Sources (Open)

| Dataset        | Description                       | Link                                                      |
| -------------- | --------------------------------- | --------------------------------------------------------- |
| Sentinel-2 L2A | 10–20m multispectral imagery      | [Copernicus Data Space](https://dataspace.copernicus.eu/) |
| ESA WorldCover | 10m land-cover labels (2021/2023) | [worldcover2021.esa.int](https://worldcover2021.esa.int/) |

### Data Pipeline

The data preparation pipeline is managed with DVC:

1. **load_aoi** — Load Area of Interest boundaries
2. **generate_grid** — Create Sentinel-2 tile grid
3. **select_tiles** — Sample tiles for processing
4. **download_imagery** — Download Sentinel-2 L2A imagery
5. **generate_labels** — Create ESA WorldCover labels
6. **patchify** — Cut tiles into 256×256 patches
7. **build_index** — Build dataset index CSV
8. **assign_splits** — Assign train/val/test/ood splits
9. **compute_norm_stats** — Compute normalization statistics

## Metrics

### Computed Metrics

The training pipeline computes the following metrics using `torchmetrics`:

| Metric            | Description                                  | Logged As                      |
| ----------------- | -------------------------------------------- | ------------------------------ |
| **mIoU**          | Mean Intersection over Union (Jaccard Index) | `{split}/mIoU`                 |
| **Macro F1**      | F1-score averaged across all classes         | `{split}/macro_f1`             |
| **Loss**          | Cross-entropy (optionally weighted + Dice)   | `{split}/loss`                 |
| **Per-class IoU** | IoU for each class (test only)               | `test_{iid,ood}/iou_class_{i}` |

Metrics are computed separately for each split:

- `train/` — Training metrics
- `val/` — Validation metrics (used for early stopping)
- `test_iid/` — In-distribution test set
- `test_ood/` — Out-of-distribution test set

### Loss Functions

Configurable via `module.loss.name`:

- `ce` — Cross-entropy loss
- `weighted_ce` — Weighted cross-entropy (class weights from training data)
- `ce_dice` — Cross-entropy + Dice loss
- `weighted_ce_dice` — Weighted cross-entropy + Dice loss

### Validation Strategy

- **Train/val/test split** based on geographically disjoint Areas of Interest
  (AOIs) to prevent spatial leakage
- **OOD (out-of-distribution) split** fully held out for final evaluation
- **Reproducibility** — Fixed random seeds, deterministic dataloader, versioned
  configs

---

# Part 2: Technical Instructions

## Setup

### Prerequisites

- macOS (Apple Silicon) or Linux
- No manual dependency installation required

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd gfm-landcover-mapping

# Run setup script (auto-installs pixi if needed)
./setup.sh

source $HOME/.bashrc

# Activate the environment
pixi shell

cd gfm-landcover-mapping
```

The `setup.sh` script:

- Detects your platform (macOS/Linux)
- Installs [pixi](https://prefix.dev/tools/pixi) package manager if missing
- Installs all dependencies (PyTorch, rasterio, GDAL, etc.)
- Verifies the installation

### Platform Support

| Platform              | PyTorch Backend | GPU Support   |
| --------------------- | --------------- | ------------- |
| macOS (Apple Silicon) | MPS             | ✅ Apple GPU  |
| Linux                 | CUDA 12.x       | ✅ NVIDIA GPU |

### Why Pixi? (Avoiding "GDAL Hell")

This project uses [Pixi](https://prefix.dev/tools/pixi) as the dependency
manager instead of the traditional conda + pip/uv approach. Here's why:

#### The Problem: "GDAL Hell"

Geospatial Python projects depend on **compiled C/C++ libraries** like GDAL,
PROJ, GEOS, and rasterio. These libraries have complex interdependencies and
require exact version matching between:

- The compiled binaries (GDAL, PROJ, GEOS)
- Python bindings (rasterio, fiona, pyproj, shapely)
- System libraries (libgdal, libproj, etc.)

**"GDAL Hell"** refers to the common pain of:

1. **Binary incompatibilities** — `pip install rasterio` downloads wheels
   compiled against GDAL 3.6, but your conda environment has GDAL 3.8 →
   `ImportError` or silent data corruption
2. **Mixed package managers** — Conda installs GDAL, pip/uv installs rasterio →
   different library versions, broken links
3. **Platform-specific builds** — macOS ARM64 vs. Intel, Linux with/without CUDA
   → different binary requirements
4. **Version conflicts** — PyTorch needs specific NumPy, rasterio needs specific
   GDAL, versions clash
5. **Broken environments** — Hours spent debugging
   `Library not loaded: @rpath/libgdal.dylib`

A typical failure mode with conda + uv:

```
# Create conda env with GDAL
conda install gdal rasterio -c conda-forge

# Later, install ML packages with uv
uv pip install pytorch-lightning torch

# Result: uv overwrites numpy/other packages,
# breaking rasterio's link to conda's GDAL
# → "ImportError: libgdal.so: cannot open shared object file"
```

#### The Solution: Pixi

[Pixi](https://prefix.dev/) solves this by:

1. **Single resolver** — One tool resolves both conda-forge packages (GDAL,
   PROJ, GEOS) AND PyPI packages (PyTorch Lightning, Hydra) together
2. **Lockfile** — `pixi.lock` captures exact versions of ALL dependencies
   (conda + PyPI) for reproducibility
3. **Platform-aware** — Automatically selects correct binaries for macOS ARM64
   vs. Linux x64
4. **No mixing** — Never mix conda-installed and pip-installed versions of the
   same package
5. **Fast** — Uses Rust-based resolver, much faster than conda

**Configuration** (`pixi.toml`):

```toml
[dependencies]
# Compiled geospatial stack from conda-forge
gdal = "*"
rasterio = "*"
pyproj = "*"
shapely = "*"

[pypi-dependencies]
# ML packages from PyPI
pytorch-lightning = ">=2.6.0"
hydra-core = ">=1.3.2"
```

Pixi ensures GDAL, rasterio, and their dependencies come from conda-forge with
matching versions, while ML packages come from PyPI without conflicts.

**Result:** Setup that "just works" on both macOS and Linux, with a single
`./setup.sh` command.

---

## Train

### Pull Training Data

All data is managed with DVC, stored in S3 compatible storage (R2 Cloudflare),
and pulled on training start

**Required files:**

- `data/index/dataset_index_with_split.csv` — patch index with train/val/test
  splits
- `data/stats/norm_stats.json` — normalization statistics
- `data/patches/` — preprocessed image patches

### Training Commands

```bash
# Train with default config
python run.py train

# Train with custom run ID
python run.py train run_id=my-experiment

# Train with Hydra config overrides
python run.py train trainer.max_epochs=100 data.batch_size=64

# Debug training (small subset, fast iteration)
python run.py debug trainer.max_epochs=10 data.num_workers=0 data.batch_size=10
```

### Run Outputs

Each run creates `runs/<run_id>/`:

```
runs/<run_id>/
├── artifacts
│   ├── lineage.json
│   └── plots
│       ├── train_loss.png
│       ├── val_loss.png
│       └── val_miou.png
├── checkpoints
│   ├── best.ckpt
│   └── last.ckpt
├── config
│   ├── config.yaml
│   └── overrides.txt
├── export
│   ├── inference_config.yaml
│   ├── model.onnx
│   ├── model.onnx.data
│   └── norm_stats.json
└── logs
    ├── hparams.yaml
    ├── hydra.log
    └── metrics.csv
```

### Logging

- **CSV Logger**: Metrics saved to `runs/<run_id>/logs/metrics.csv`
- **MLflow** (optional): Enable with `logging.mlflow.enabled=true`

### Configuration

Override any config value via CLI:

```bash
python run.py train trainer.max_epochs=50 data.batch_size=16 module.lr=1e-4
```

Key config files:

- `configs/training.yaml` — Main training config
- `configs/training/data.yaml` — Dataset paths, batch size
- `configs/training/trainer.yaml` — PyTorch Lightning trainer
- `configs/model/unet_baseline.yaml` — UNet architecture

---

## Production (Export)

### Export Model

Export a trained model to ONNX:

```bash
# Export best checkpoint (default)
python run.py export <run_id>

# Export last checkpoint
python run.py export <run_id> export.checkpoint=last
```

Example:

```bash
python run.py export lcseg-20260107-160706-c2b5c83
```

Exports are saved to `runs/<run_id>/export/`.

### Using Pre-trained Models

Pull an example trained model from DVC:

```bash
dvc pull runs/lcseg-20260107-160706-c2b5c83.dvc
```

## Inference

The inference server is built with [FastAPI](https://fastapi.tiangolo.com/) and
provides a REST API for land-cover segmentation predictions. It operates on a
**catalog of tiles** organized by country.

**Interactive API Documentation** (after starting server):

- Swagger UI: `http://{host}:{port}/docs`
- ReDoc: `http://{host}:{port}/redoc`

Default: `http://0.0.0.0:8000/docs` (configurable via `server.host` and
`server.port`)

### Pull Inference Data (Optional)

Data for inference is pulled from DVC on the inference server startup

```bash
dvc pull data/inference.dvc
```

This downloads pre-processed tiles organized by country:

```
data/inference/
├── imagery/
│   ├── DEU/              # Germany
│   │   ├── <tile_hash>/
│   │   │   └── spectral.tif
│   │   └── ...
│   └── NLD/              # Netherlands
│       └── ...
└── labels/
    ├── DEU/
    │   ├── <tile_hash>/
    │   │   └── labels.tif
    │   └── ...
    └── NLD/
        └── ...
```

### Start Inference Server

```bash
# Using a local ONNX model
python run.py serve model.local.onnx_path=runs/lcseg-20260107-160706-c2b5c83/export/model.onnx

# Using an ONNX model from MLFlow
python run.py serve \
  model.source=mlflow \
  model.mlflow.tracking_uri=<mlflow tracking_uri> \
  'model.mlflow.model_uri=runs:/<mlflow run id>/export'

# Custom port
python run.py serve model.local.onnx_path=runs/<run_id>/export/model.onnx server.port=8080

# With GPU inference (Linux)
python run.py serve model.local.onnx_path=runs/<run_id>/export/model.onnx \
    'runtime.providers=["CUDAExecutionProvider","CPUExecutionProvider"]'
```

### Inference Workflow

The inference workflow follows these steps:

```
1. List Countries → 2. Select Country → 3. List Tiles → 4. Select Tile → 5. Run Inference
```

#### Step 1: List Available Countries

```bash
curl http://localhost:8000/countries
```

Response:

```json
{
  "countries": ["DEU", "NLD", "PRT"]
}
```

#### Step 2: List Tiles for a Country

```bash
curl http://localhost:8000/tiles?country=DEU
```

Response:

```json
{
  "country": "DEU",
  "tiles": ["a1b2c3d4", "e5f6g7h8", ...]
}
```

#### Step 3: Get Tile Metadata (Optional)

```bash
curl http://localhost:8000/tiles/DEU/a1b2c3d4/meta
```

Response:

```json
{
  "tile_id": "a1b2c3d4",
  "country": "DEU",
  "width": 10980,
  "height": 10980,
  "crs": "EPSG:32632",
  "bounds": [300000.0, 5790240.0, 409800.0, 5900040.0],
  "band_count": 4
}
```

#### Step 4: Run Inference

Send a POST request to `/infer` with a window specification:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/infer' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "country": "PRT",
  "tile_id": "ec3f9378e4aa45dd3b65e358225fc47f15b13dcbc7b7ad59eebb1ff2d3320f67",
  "row_off": 0,
  "col_off": 0,
  "height": 512,
  "width": 512
}' --output result.zip
```

**Understanding the Window Parameters:**

Tiles are large raster images (e.g., 10980×10980 pixels for Sentinel-2). To run
inference, you select a **window** (rectangular region) within the tile:

```
┌─────────────────────────────────────┐
│ Tile (10980 × 10980 pixels)         │
│                                     │
│    col_off                          │
│    ↓                                │
│    ┌─────────────┐ ← row_off        │
│    │             │                  │
│    │   Window    │ height           │
│    │             │                  │
│    └─────────────┘                  │
│         width                       │
└─────────────────────────────────────┘
```

- **`row_off`** — Vertical offset from top-left corner of tile (in pixels)
- **`col_off`** — Horizontal offset from top-left corner of tile (in pixels)
- **`height`**, **`width`** — Size of the window to extract and process

The window is then processed using a **sliding window** approach with patches:

- **`patch_size`** — Size of patches fed to the model (default: 256×256,
  matching training)
- **`stride`** — Step size between patches (default: same as patch_size, no
  overlap)

For windows larger than `patch_size`, the server automatically tiles the window
into patches, runs inference on each, and stitches results.

**Request Parameters:**

| Parameter         | Type   | Default  | Description                                              |
| ----------------- | ------ | -------- | -------------------------------------------------------- |
| `country`         | string | required | Country code (e.g., "DEU")                               |
| `tile_id`         | string | required | Tile identifier (hash)                                   |
| `row_off`         | int    | 0        | Row offset from top-left of tile (pixels)                |
| `col_off`         | int    | 0        | Column offset from top-left of tile (pixels)             |
| `height`          | int    | 256      | Window height (pixels)                                   |
| `width`           | int    | 256      | Window width (pixels)                                    |
| `patch_size`      | int    | 256      | Size of patches for model inference                      |
| `stride`          | int    | 256      | Stride for sliding window (use < patch_size for overlap) |
| `batch_size`      | int    | config   | Number of patches per batch                              |
| `include_pred`    | bool   | true     | Include prediction PNG                                   |
| `include_label`   | bool   | true     | Include ground-truth PNG                                 |
| `include_compare` | bool   | true     | Include side-by-side comparison                          |

**Response:**

Returns a ZIP file containing:

- `pred.png` — Predicted land-cover map (color-coded)
- `label.png` — Ground truth labels (if available)
- `compare.png` — Side-by-side comparison
- `stats.json` — Inference statistics and class histograms

Example `stats.json`:

```json
{
  "inference_id": "infer-20260107-224530-abc123",
  "timings": {
    "total": 1.234,
    "preprocess": 0.123,
    "inference": 0.890,
    "postprocess": 0.221
  },
  "window_info": {
    "row_off": 0,
    "col_off": 0,
    "height": 512,
    "width": 512,
    "actual_height": 512,
    "actual_width": 512
  },
  "pred_histogram": [
    {"class_id": 10, "class_name": "Tree cover", "pixel_count": 65536, "fraction": 0.25},
    {"class_id": 40, "class_name": "Cropland", "pixel_count": 131072, "fraction": 0.50},
    ...
  ],
  "label_histogram": [...]
}
```

### API Endpoints Summary

| Method | Endpoint                          | Description               |
| ------ | --------------------------------- | ------------------------- |
| GET    | `/health`                         | Health check              |
| GET    | `/model`                          | Model information         |
| GET    | `/countries`                      | List available countries  |
| GET    | `/tiles?country={code}`           | List tiles for a country  |
| GET    | `/tiles/{country}/{tile_id}/meta` | Get tile metadata         |
| POST   | `/infer`                          | Run inference on a window |

### Inference Configuration

Key config file: `configs/inference.yaml`

| Config                       | Description                             |
| ---------------------------- | --------------------------------------- |
| `model.source`               | `local` or `mlflow`                     |
| `model.mlflow.tracking_uri`  | MLFlow tracking URI                     |
| `model.local.onnx_path`      | Path to ONNX model                      |
| `data.imagery_root`          | Root directory for imagery tiles        |
| `data.labels_root`           | Root directory for label tiles          |
| `data.allowed_countries`     | Filter to specific countries (optional) |
| `server.host`, `server.port` | Server binding                          |
| `runtime.providers`          | ONNX Runtime execution providers        |

---

# Appendix

## Training Data Contract

### Input Artifact

- CSV: `dataset_index_with_split.csv`
- One row = one training sample (patch)
- File is **read-only and immutable**

### Required Columns

| Column          | Description                                   |
| --------------- | --------------------------------------------- |
| `patch_id`      | Unique sample ID                              |
| `tile_id`       | Parent Sentinel-2 tile                        |
| `group_id`      | Grouping key (no group spans multiple splits) |
| `split`         | One of: `train`, `val`, `test`, `ood`         |
| `spectral_path` | Path to 4-band S2 patch (B02, B03, B04, B08)  |
| `label_path`    | Path to ESA WorldCover labels                 |
| `cloud_frac`    | Cloud fraction (filtering threshold)          |

### Tensor Contract

**Input:**

- `x`: `FloatTensor [4, 256, 256]` — Channels: B02, B03, B04, B08
- Normalized using frozen train-split statistics

**Target:**

- `y`: `LongTensor [256, 256]` — Integer WorldCover class IDs

### Split Rules

- Splits are **pre-assigned and frozen**
- **OOD split is fully held out** — not used for normalization, class weights,
  or early stopping
- Filtering: `cloud_frac ≤ 0.20` for train/val

## Project Structure

```
├── run.py                     # CLI entrypoint (train, debug, export, serve)
├── setup.sh                   # Environment setup script
├── pixi.toml                  # Pixi package configuration
├── dvc.yaml                   # DVC pipeline definition
├── configs/                   # Hydra configuration
├── data_preparation/          # Data pipeline modules
├── landcover/                 # Training code
│   ├── datasets/              # DataModule, Dataset, transforms
│   ├── models/                # UNet baseline
│   ├── training/              # Training loop
│   └── callbacks/             # Custom callbacks
├── inference/                 # Inference server
├── runs/                      # Training run outputs
└── data/                      # Dataset (DVC-managed)
```
