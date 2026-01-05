#!/usr/bin/env bash
# scripts/setup_env.sh
# One-shot environment setup for training/inference on a new machine.
# Prerequisites: conda and uv must be installed.
#
# Usage:
#   ./scripts/setup_env.sh              # Use lockfiles (default), fallback to fresh solve
#   ./scripts/setup_env.sh --fresh      # Force fresh solve from pyproject.toml
#   ./scripts/setup_env.sh --freeze     # Setup + create/update lockfiles
#
# Lockfiles are platform-specific:
#   - environment.linux.lock.yml / requirements.linux.lock.txt
#   - environment.macos.lock.yml / requirements.macos.lock.txt

set -euo pipefail

ENV_NAME="gfm"
PYTHON_VERSION="3.11"
FORCE_FRESH=false
FREEZE=false

# Parse args
for arg in "$@"; do
    case $arg in
        --fresh)
            FORCE_FRESH=true
            ;;
        --freeze)
            FREEZE=true
            ;;
    esac
done

echo "=== GFM Environment Setup ==="
echo "Environment: $ENV_NAME"
echo "Python: $PYTHON_VERSION"

# Detect platform
if [[ "$(uname -s)" == "Linux" ]]; then
    PLATFORM="linux"
    echo "Platform: Linux (CUDA support)"
elif [[ "$(uname -s)" == "Darwin" ]]; then
    PLATFORM="macos"
    echo "Platform: macOS (MPS support)"
else
    echo "Unsupported platform: $(uname -s). If you are using Windows, please use WSL."
    exit 1
fi

# Platform-specific lockfile names
CONDA_LOCK="environment.${PLATFORM}.lock.yml"
PIP_LOCK="requirements.${PLATFORM}.lock.txt"

# Check prerequisites
command -v conda >/dev/null 2>&1 || { echo "conda not found. Please install miniconda/miniforge."; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# Determine mode: lockfiles (default) or fresh solve
USE_LOCK=false
if [[ "$FORCE_FRESH" == false && -f "$CONDA_LOCK" && -f "$PIP_LOCK" ]]; then
    USE_LOCK=true
    echo "Mode: lockfiles (exact reproduction)"
    echo "  Using: $CONDA_LOCK, $PIP_LOCK"
else
    echo "Mode: fresh solve (from pyproject.toml)"
fi
echo ""

# --- LOCKFILE MODE ---
if [[ "$USE_LOCK" == true ]]; then
    echo "Installing from lockfiles..."

    # Create env from conda lockfile
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Updating existing environment from lockfile..."
        conda env update -n "$ENV_NAME" -f "$CONDA_LOCK" --prune
    else
        echo "Creating environment from lockfile..."
        conda env create -f "$CONDA_LOCK"
    fi

    # Install pip packages from lockfile
    echo "Installing pip packages from lockfile..."
    conda run -n "$ENV_NAME" uv pip install -r "$PIP_LOCK"

# --- FRESH SOLVE MODE ---
else
    # Create conda environment if it doesn't exist
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Conda environment '$ENV_NAME' already exists."
    else
        echo "Creating conda environment '$ENV_NAME'..."
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    fi

    # Install conda packages (compiled/system dependencies)
    echo ""
    echo "=== Installing conda packages ==="

    CONDA_PACKAGES=(
        "gdal"
        "rasterio"
        "fiona"
        "proj"
        "geos"
        "pyproj"
        "shapely"
        "numpy>=2.4.0,<3"
    )

    if [[ "$PLATFORM" == "linux" ]]; then
        # Linux: install CUDA-enabled PyTorch
        echo "Installing PyTorch with CUDA support..."
        conda install -n "$ENV_NAME" -c pytorch -c nvidia -c conda-forge -y \
            pytorch torchvision pytorch-cuda=12.1 \
            "${CONDA_PACKAGES[@]}"
    else
        # macOS: install MPS-enabled PyTorch
        echo "Installing PyTorch with MPS support..."
        conda install -n "$ENV_NAME" -c conda-forge -y \
            "pytorch>=2.5" "torchvision>=0.20" \
            "${CONDA_PACKAGES[@]}"
    fi

    # Install Python packages via uv
    echo ""
    echo "=== Installing Python packages via uv ==="
    conda run -n "$ENV_NAME" uv pip install -e .
fi

# Verify installation
echo ""
echo "=== Verifying installation ==="
conda run -n "$ENV_NAME" python -c "
import torch
import rasterio
import pytorch_lightning
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Rasterio: {rasterio.__version__}')
print('All imports OK!')
"

# Optional: freeze environment
if [[ "$FREEZE" == true ]]; then
    echo ""
    echo "=== Freezing environment for $PLATFORM ==="
    conda env export -n "$ENV_NAME" > "$CONDA_LOCK"
    conda run -n "$ENV_NAME" uv pip freeze > "$PIP_LOCK"
    echo "Created: $CONDA_LOCK, $PIP_LOCK"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_NAME"
