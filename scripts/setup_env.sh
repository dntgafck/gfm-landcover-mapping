#!/usr/bin/env bash
# scripts/setup_env.sh
# Idempotent environment setup for macOS and Linux
# Uses Conda for system/compiled deps + uv.lock for PyPI packages
#
# Prerequisites: conda and uv must be installed.
#
# Usage:
#   ./scripts/setup_env.sh                    # Basic setup (base deps only)
#   ./scripts/setup_env.sh --dev              # Setup with dev dependencies
#   ./scripts/setup_env.sh --train            # Setup with training dependencies
#   ./scripts/setup_env.sh --train --frozen   # Frozen install (no lock changes, for CI/remote)
#   ./scripts/setup_env.sh --refresh-conda    # Force recreate conda env
#   ./scripts/setup_env.sh --lock             # Regenerate uv.lock (deliberate action)
#   ./scripts/setup_env.sh --activate         # Activate env after setup (requires sourcing)
#
# To activate the environment in your current shell after setup:
#   source ./scripts/setup_env.sh --activate

set -euo pipefail

ENV_NAME="gfm"
PYTHON_VERSION="3.11"

# Default flags
INSTALL_DEV=false
INSTALL_TRAIN=false
REFRESH_CONDA=false
REGENERATE_LOCK=false
FROZEN_INSTALL=false
ACTIVATE_ENV=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dev)
            INSTALL_DEV=true
            ;;
        --train)
            INSTALL_TRAIN=true
            ;;
        --refresh-conda)
            REFRESH_CONDA=true
            ;;
        --lock)
            REGENERATE_LOCK=true
            ;;
        --frozen)
            FROZEN_INSTALL=true
            ;;
        --activate)
            ACTIVATE_ENV=true
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--dev] [--train] [--refresh-conda] [--lock] [--frozen] [--activate]"
            exit 1
            ;;
    esac
done

echo "=== GFM Environment Setup ==="
echo "Environment: $ENV_NAME"
echo "Python: $PYTHON_VERSION"

# Detect platform
if [[ "$(uname -s)" == "Linux" ]]; then
    PLATFORM="linux"
    CONDA_ENV_FILE="environment.linux.yml"
    echo "Platform: Linux (CUDA support)"
elif [[ "$(uname -s)" == "Darwin" ]]; then
    PLATFORM="macos"
    CONDA_ENV_FILE="environment.macos.yml"
    echo "Platform: macOS (MPS support)"
else
    echo "Error: Unsupported platform: $(uname -s)"
    echo "If you are using Windows, please use WSL."
    exit 1
fi

# Display configuration
echo ""
echo "Configuration:"
echo "  - Conda env file: $CONDA_ENV_FILE"
echo "  - Install dev deps: $INSTALL_DEV"
echo "  - Install train deps: $INSTALL_TRAIN"
echo "  - Refresh conda: $REFRESH_CONDA"
echo "  - Regenerate lock: $REGENERATE_LOCK"
echo "  - Frozen install: $FROZEN_INSTALL"
echo "  - Activate after setup: $ACTIVATE_ENV"
echo ""

# Check prerequisites
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found. Please install miniconda/miniforge."
    echo "  macOS: brew install miniforge"
    echo "  Linux: curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash Miniforge3-Linux-x86_64.sh"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv not found. Install with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check that conda env file exists
if [[ ! -f "$CONDA_ENV_FILE" ]]; then
    echo "Error: Conda environment file not found: $CONDA_ENV_FILE"
    exit 1
fi

# --- CONDA ENVIRONMENT SETUP ---
echo "=== Setting up Conda environment ==="

if conda env list | grep -q "^${ENV_NAME} "; then
    if [[ "$REFRESH_CONDA" == true ]]; then
        echo "Removing existing environment for fresh install..."
        conda env remove -n "$ENV_NAME" -y
        echo "Creating conda environment from $CONDA_ENV_FILE..."
        conda env create -f "$CONDA_ENV_FILE"
    else
        echo "Updating existing environment from $CONDA_ENV_FILE..."
        conda env update -n "$ENV_NAME" -f "$CONDA_ENV_FILE" --prune
    fi
else
    echo "Creating conda environment from $CONDA_ENV_FILE..."
    conda env create -f "$CONDA_ENV_FILE"
fi

# --- UV LOCK MANAGEMENT ---
echo ""
echo "=== Managing uv.lock ==="

if [[ "$REGENERATE_LOCK" == true ]]; then
    echo "Regenerating uv.lock from pyproject.toml..."
    conda run -n "$ENV_NAME" uv lock --upgrade
    echo "✓ uv.lock regenerated"
elif [[ ! -f "uv.lock" ]]; then
    echo "uv.lock not found, generating..."
    conda run -n "$ENV_NAME" uv lock
    echo "✓ uv.lock created"
else
    echo "Using existing uv.lock"
fi

# --- PYTHON DEPENDENCIES INSTALLATION ---
echo ""
echo "=== Installing Python dependencies ==="

# Build uv sync command with appropriate flags
UV_SYNC_CMD="uv sync --no-dev"

if [[ "$FROZEN_INSTALL" == true ]]; then
    UV_SYNC_CMD="$UV_SYNC_CMD --frozen"
    echo "Mode: FROZEN (no lock changes allowed)"
else
    echo "Mode: Standard (may update lock if needed)"
fi

# Add optional dependency groups
EXTRA_GROUPS=""
if [[ "$INSTALL_DEV" == true ]]; then
    EXTRA_GROUPS="$EXTRA_GROUPS --extra dev"
    echo "Including: dev dependencies"
fi

if [[ "$INSTALL_TRAIN" == true ]]; then
    EXTRA_GROUPS="$EXTRA_GROUPS --extra train"
    echo "Including: train dependencies"

    # On Linux, also include train-gpu extras
    if [[ "$PLATFORM" == "linux" ]]; then
        EXTRA_GROUPS="$EXTRA_GROUPS --extra train-gpu"
        echo "Including: train-gpu dependencies (Linux)"
    fi
fi

# Run uv sync in the conda environment
# Note: uv will detect the conda python and install into that environment
echo ""
echo "Running: conda run -n $ENV_NAME $UV_SYNC_CMD $EXTRA_GROUPS"
conda run -n "$ENV_NAME" bash -c "$UV_SYNC_CMD $EXTRA_GROUPS"

# --- TENSORRT INSTALLATION (Linux only) ---
if [[ "$PLATFORM" == "linux" && "$INSTALL_TRAIN" == true ]]; then
    echo ""
    echo "=== Installing TensorRT (Linux-only) ==="
    echo "Installing tensorrt and torch-tensorrt via uv..."
    conda run -n "$ENV_NAME" uv pip install tensorrt torch-tensorrt --no-deps || {
        echo "⚠ TensorRT installation failed. This is optional for GPU inference optimization."
        echo "  You can install it manually later if needed:"
        echo "  conda activate gfm && uv pip install tensorrt torch-tensorrt"
    }
fi

# --- VERIFICATION ---
echo ""
echo "=== Verifying installation ==="

# Run verification in the conda environment
conda run -n "$ENV_NAME" python -c "
import sys
print(f'Python: {sys.version}')
print()

# Test core imports
try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'  - CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  - CUDA version: {torch.version.cuda}')
        print(f'  - GPU count: {torch.cuda.device_count()}')
    print(f'  - MPS available: {torch.backends.mps.is_available()}')
except ImportError as e:
    print(f'✗ PyTorch import failed: {e}')
    sys.exit(1)

try:
    import rasterio
    print(f'✓ Rasterio: {rasterio.__version__}')
except ImportError as e:
    print(f'✗ Rasterio import failed: {e}')
    sys.exit(1)

try:
    import pytorch_lightning
    print(f'✓ PyTorch Lightning: {pytorch_lightning.__version__}')
except ImportError as e:
    print(f'✗ PyTorch Lightning import failed: {e}')
    sys.exit(1)

try:
    import geopandas
    print(f'✓ GeoPandas: {geopandas.__version__}')
except ImportError as e:
    print(f'✗ GeoPandas import failed: {e}')
    sys.exit(1)

# Check optional TensorRT (Linux only)
import platform
if platform.system() == 'Linux' and '$INSTALL_TRAIN' == 'true':
    try:
        import tensorrt
        print(f'✓ TensorRT: {tensorrt.__version__}')
    except ImportError:
        print('ℹ TensorRT not available (optional for GPU training)')

    try:
        import torch_tensorrt
        print(f'✓ torch-tensorrt: {torch_tensorrt.__version__}')
    except ImportError:
        print('ℹ torch-tensorrt not available (optional for GPU training)')

print()
print('All core imports successful!')
"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "=== Setup complete! ==="
    echo ""

    # Activate the environment if requested
    if [[ "$ACTIVATE_ENV" == true ]]; then
        echo "Activating conda environment: $ENV_NAME"
        # Initialize conda for the current shell
        eval "$(conda shell.bash hook)"
        conda activate "$ENV_NAME"
        echo "✓ Environment '$ENV_NAME' is now active"
        echo ""
    else
        echo "Activate the environment with:"
        echo "  conda activate $ENV_NAME"
        echo ""
        echo "Or re-run this script with --activate (must be sourced):"
        echo "  source ./scripts/setup_env.sh --activate"
        echo ""
    fi

    if [[ "$INSTALL_DEV" == false && "$INSTALL_TRAIN" == false ]]; then
        echo "To install additional dependencies:"
        echo "  ./scripts/setup_env.sh --dev          # Development tools"
        echo "  ./scripts/setup_env.sh --train        # Training dependencies"
    fi
else
    echo ""
    echo "=== Setup completed with errors ==="
    echo "Please check the error messages above."
    exit 1
fi
