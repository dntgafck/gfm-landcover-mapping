# Dockerfile for GFM Land Cover Mapping Training
# Multi-stage build: Builder -> Runtime
#
# Usage:
#   docker build -t gfm-landcover:latest .
#   ./scripts/train_docker.sh -- train
#   ./scripts/train_docker.sh -- debug

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for geospatial stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Geospatial libraries
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    # Build tools for Python packages
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set GDAL config for building rasterio/fiona
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch>=2.5 \
    torchvision>=0.20 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy dependency files
WORKDIR /app
COPY pyproject.toml ./

# Install project dependencies (without the project itself)
# We install in multiple steps for better caching
RUN pip install --no-cache-dir \
    # Geospatial (needs system GDAL)
    rasterio \
    fiona \
    geopandas \
    pyproj \
    shapely

# Install remaining dependencies from pyproject.toml
# Using pip with [train] extras
RUN pip install --no-cache-dir \
    "sentinelhub>=3.11.3,<4" \
    requests \
    "pandas>=2.3.3,<3" \
    "matplotlib>=3.10.8,<4" \
    boto3 \
    "hydra-core>=1.3.2,<2" \
    "pytorch-lightning>=2.6.0,<3" \
    "torchmetrics>=1.8.2,<2" \
    mlflow \
    fire \
    # Train extras
    onnx \
    "onnxscript>=0.1.0" \
    # DVC with S3 support
    "dvc[s3]>=3.65.0,<4"


# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only (no dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    # Runtime geospatial libs
    gdal-bin \
    libgdal32 \
    libgeos-c1v5 \
    libproj22 \
    # Git for DVC and lineage
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy virtual environment from builder
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project source code
COPY . .

# Install project in editable mode (just registers the package, deps already installed)
RUN pip install --no-cache-dir --no-deps -e .

# Verify critical imports
RUN python -c "import torch; import rasterio; import pytorch_lightning; print('All imports OK')"

# Default entrypoint
ENTRYPOINT ["python", "run.py"]

# Default command (can be overridden)
CMD ["--help"]
