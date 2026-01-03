---
trigger: always_on
---

# Project Context & AI Interaction Rules

This document summarizes the current state of the **GFM Landcover Mapping** project and provides specific instructions for subsequent AI interactions to ensure consistency, quality, and progress.

## 🚀 Project Mission
Investigate if Geospatial Foundation Models (GFMs) significantly improve accuracy and label efficiency in land-cover mapping compared to scratch-trained baselines (U-Net).

## 🛠 Tech Stack & Environment
- **Dependency Management (pixi)**: All Python packages and external binaries are managed via \`pixi\`. Always execute project commands through \`pixi\` (e.g., \`pixi run python script.py\` or \`pixi run -e dev ...\`).
- **Data Management (DVC)**: Data versioning and pipelines are managed with **DVC** (defined in \`dvc.yaml\`). Execute all data steps through DVC (e.g., \`pixi run -e dev dvc repro\`).
- **Core Framework**: PyTorch Lightning for training orchestration.
- **Hardware**: Supports \`osx-arm64\` (MPS) and \`linux-64\` (CUDA).
- **Secrets**: Credentials are in \`.env\` (git-ignored). Never commit secrets. Load via \`set -a; source .env; set +a\`.

---

## 🛰 Remote Training Workflow
- **Antigravity-Free Training**: Actual model training occurs on a remote GPU machine (VM) where Antigravity is **not** present.
- **Agent Responsibility**: Antigravity is responsible for local development, pipeline verification, and ensuring the code is "remote-ready" (linted, configured, and reproducible).
- **DVC as the Bridge**: Use DVC to synchronize data, code, and model artifacts between the local environment and the remote VM.
  - Push local changes/data: \`pixi run -e dev dvc push\`
  - Pull remote results/models: \`pixi run -e dev dvc pull\`

---

## 📂 System Architecture
\`\`\`mermaid
graph TD
    A["AOI & Natural Earth"] --> B["01_aoi.py"]
    B --> C["02_grid.py"]
    C --> D["03_select_tiles.py"]
    D --> E["04_download.py - Sentinel Hub"]
    E --> F["05_label.py - WorldCover"]
    F --> G["06_patchify.py"]
    G --> H["07_assign_splits.py"]
    H --> I["08_compute_norm_stats.py"]
    I --> J["train.py - U-Net Baseline"]
    J --> K["ONNX Export & Metadata"]
\`\`\`

---

## 📋 Interaction Rules for Antigravity

### 1. The \"Zero Lint\" Policy & Git Staging
- **Requirement**: Before completing any task with code changes, you **must** resolve all linter (ruff) and type (mypy) errors.
- **How to run**:
  - \`pixi run -e dev lint-all\`: Runs on **all files** in the repository. Use this to ensure everything is clean before finishing.
  - \`pre-commit run\`: Default \`pre-commit\` behavior ONLY tracks **staged files** (files in the git index).
- **Staging Awareness**: If you want to check specific changes without running on the whole codebase, ensure you \`git add\` the files first.

### 2. Data Contract Adherence
- Respect the **Training Data Contract** in \`README.md\`.
- Never modify the split assignment in \`dataset_index_with_split.csv\` during training.
- OOD data is \`strictly read-only\` for testing and must never influence normalization or early stopping.

### 3. Workflow Patterns
- **Logging**: Use \`utils.logging.setup_logging()\` in main entry points and \`logging.getLogger(__name__)\` in modules. No \`print\` statements in library code (\`landcover/\`) or pipeline scripts.
- **Config**: Hyperparameters and paths must be derived from \`conf/params.yaml\` or \`conf/*.yaml\` via Hydra.
- **DVC Updates**: Any change to data processing logic must be accompanied by an update/check of \`dvc.yaml\`.

---

## 📈 Recent Progress & Current Frontier
- **Done**: Robust multi-stage DVC pipeline, IID/OOD country-stratified splitting, OOP-based sampling, U-Net baseline orchestration, and ONNX export.
- **Current Focus**: Ensuring training reliability and validating export artifacts.
- **Future**: NASA Prithvi GFM fine-tuning, per-country/per-class metrics, and inference service.
