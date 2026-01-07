#!/usr/bin/env bash
# scripts/train_docker.sh
# Run training inside Docker container with GPU support
#
# Usage:
#   ./scripts/train_docker.sh -- train                          # Regular training
#   ./scripts/train_docker.sh -- debug                          # Debug mode
#   ./scripts/train_docker.sh -- train run_id=my-exp            # Custom run ID
#   ./scripts/train_docker.sh --gpus '"device=0"' -- train      # Specific GPU
#   ./scripts/train_docker.sh --build -- train                  # Rebuild image first
#
# Mounts:
#   - ./data    -> /app/data   (DVC data, read-only recommended)
#   - ./runs    -> /app/runs   (training outputs)
#   - ./.env    -> /app/.env   (AWS credentials for DVC push, optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="gfm-landcover:latest"
CONTAINER_NAME="gfm-train-$(date +%s)"

# Default settings
GPUS="all"
BUILD_IMAGE=false
INTERACTIVE=false

# Parse docker-specific arguments (before --)
DOCKER_ARGS=()
RUN_ARGS=()
PARSING_RUN_ARGS=false

for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        PARSING_RUN_ARGS=true
        continue
    fi

    if [[ "$PARSING_RUN_ARGS" == true ]]; then
        RUN_ARGS+=("$arg")
    else
        case $arg in
            --gpus=*)
                GPUS="${arg#*=}"
                ;;
            --gpus)
                # Next arg will be GPU spec, handled below
                DOCKER_ARGS+=("$arg")
                ;;
            --build)
                BUILD_IMAGE=true
                ;;
            -it|--interactive)
                INTERACTIVE=true
                ;;
            *)
                DOCKER_ARGS+=("$arg")
                ;;
        esac
    fi
done

# Extract --gpus value if passed as separate arg
for i in "${!DOCKER_ARGS[@]}"; do
    if [[ "${DOCKER_ARGS[$i]}" == "--gpus" ]] && [[ $((i + 1)) -lt ${#DOCKER_ARGS[@]} ]]; then
        GPUS="${DOCKER_ARGS[$((i + 1))]}"
        unset 'DOCKER_ARGS[$i]'
        unset 'DOCKER_ARGS[$((i + 1))]'
    fi
done

cd "$PROJECT_ROOT"

# Build image if requested or if it doesn't exist
if [[ "$BUILD_IMAGE" == true ]]; then
    echo "=== Building Docker image ==="
    docker build -t "$IMAGE_NAME" .
elif ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "=== Image not found, building ==="
    docker build -t "$IMAGE_NAME" .
fi

echo ""
echo "=== Running training in Docker ==="
echo "Image: $IMAGE_NAME"
echo "GPUs: $GPUS"
echo "Command: python run.py ${RUN_ARGS[*]:-}"
echo ""

# Build docker run command
DOCKER_RUN_CMD=(
    docker run
    --rm
    --name "$CONTAINER_NAME"
    --gpus "$GPUS"
    # Mount data directory (from DVC)
    -v "$PROJECT_ROOT/data:/app/data:ro"
    # Mount runs directory for outputs
    -v "$PROJECT_ROOT/runs:/app/runs"
    # Mount .env for AWS credentials (DVC push)
    # Only mount if exists
)

# Conditionally mount .env
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    DOCKER_RUN_CMD+=(-v "$PROJECT_ROOT/.env:/app/.env:ro")
fi

# Add interactive flag if requested
if [[ "$INTERACTIVE" == true ]]; then
    DOCKER_RUN_CMD+=(-it)
fi

# Add image name and run args
DOCKER_RUN_CMD+=("$IMAGE_NAME")
if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then
    DOCKER_RUN_CMD+=("${RUN_ARGS[@]}")
fi

# Execute
"${DOCKER_RUN_CMD[@]}"
