#!/usr/bin/env bash
# scripts/setup_env.sh
# Idempotent environment setup using pixi for macOS and Linux
#
# Prerequisites: None - pixi will be installed automatically if missing
#
# Usage:
#   ./scripts/setup_env.sh           # Setup environment
#   source ./scripts/setup_env.sh --activate    # Activate environment

set -euo pipefail

ACTIVATE_ENV=true

# Parse arguments
for arg in "$@"; do
    case $arg in
        --activate)
            ACTIVATE_ENV=true
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--activate]"
            exit 1
            ;;
    esac
done

echo "=== GFM Environment Setup with Pixi ==="

# Detect platform
if [[ "$(uname -s)" == "Linux" ]]; then
    PLATFORM="linux-64"
    echo "Platform: Linux (CUDA support)"
elif [[ "$(uname -s)" == "Darwin" ]]; then
    PLATFORM="osx-arm64"
    echo "Platform: macOS (MPS support)"
else
    echo "Error: Unsupported platform: $(uname -s)"
    echo "If you are using Windows, please use WSL."
    exit 1
fi

# Check if pixi is installed
if ! command -v pixi >/dev/null 2>&1; then
    echo ""
    echo "=== Pixi not found, installing... ==="

    # Install pixi
    if [[ "$(uname -s)" == "Darwin" ]]; then
        # macOS - try brew first, fall back to curl
        if command -v brew >/dev/null 2>&1; then
            echo "Installing pixi via Homebrew..."
            brew install pixi
        else
            echo "Installing pixi via curl..."
            curl -fsSL https://pixi.sh/install.sh | bash

            # Add pixi to PATH for this session
            export PATH="$HOME/.pixi/bin:$PATH"
        fi
    else
        # Linux - use curl
        echo "Installing pixi via curl..."
        curl -fsSL https://pixi.sh/install.sh | bash

        # Add pixi to PATH for this session
        export PATH="$HOME/.pixi/bin:$PATH"
    fi

    # Verify installation
    if ! command -v pixi >/dev/null 2>&1; then
        echo "Error: pixi installation failed"
        echo "Please install manually: https://prefix.dev/docs/pixi/overview"
        exit 1
    fi

    echo "✓ Pixi installed successfully"
fi

# Show pixi version
PIXI_VERSION=$(pixi --version)
echo "Using: $PIXI_VERSION"
echo ""

# --- PIXI ENVIRONMENT SETUP ---
echo "=== Installing dependencies with pixi ==="

# Run pixi install for the default environment only
# Use --no-lockfile-update if lockfile exists to avoid cross-platform resolution
if [[ -f "pixi.lock" ]]; then
    echo "Using existing pixi.lock (no update)"
    pixi install -e default --frozen
else
    echo "Generating pixi.lock for current platform"
    pixi install -e default
fi

echo ""
echo "=== Verifying installation ==="

# Run verification in the pixi environment
pixi run python -c "
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

try:
    import fire
    print(f'✓ Fire: {fire.__version__}')
except ImportError as e:
    print(f'✗ Fire import failed: {e}')
    sys.exit(1)

# Check optional ONNX Runtime GPU (Linux only)
import platform
if platform.system() == 'Linux':
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        print(f'✓ ONNX Runtime: {onnxruntime.__version__}')
        print(f'  - Providers: {providers}')
        if 'CUDAExecutionProvider' in providers:
            print('  - CUDA inference: enabled')
        if 'TensorrtExecutionProvider' in providers:
            print('  - TensorRT inference: enabled')
    except ImportError:
        print('ℹ onnxruntime-gpu not available (optional for GPU inference)')

print()
print('All core imports successful!')
"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "=== Setup complete! ==="
    echo ""

    # Activate the environment if requested
    if [[ "$ACTIVATE_ENV" == true ]]; then
        echo "Activating pixi environment..."
        # This only works if the script is sourced
        if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
            echo "Warning: Script must be sourced to activate environment"
            echo "Run: source ./setup.sh --activate"
        else
            eval "$(pixi shell-hook)"
            echo "✓ Pixi environment is now active"
            echo ""
        fi
    else
        echo "Activate the environment with:"
        echo "  pixi shell"
        echo ""
        echo "Or run commands directly with:"
        echo "  pixi run <command>"
        echo ""
        echo "Or source this script:"
        echo "  source ./setup.sh --activate"
        echo ""
    fi
else
    echo ""
    echo "=== Setup completed with errors ==="
    echo "Please check the error messages above."
    exit 1
fi
