"""ONNX model loading from MLflow or local filesystem."""

import json
from pathlib import Path

import onnxruntime as ort
from omegaconf import DictConfig

from inference.utils import ensure_dir
from utils.logging import get_logger

logger = get_logger(__name__)


def load_model_from_mlflow(cfg: DictConfig) -> tuple[Path, Path]:
    """Download ONNX model and artifacts from MLflow.

    Args:
        cfg: Hydra configuration with model.mlflow settings

    Returns:
        Tuple of (path to ONNX file, path to cache dir with all artifacts)
    """
    import mlflow

    mlflow_cfg = cfg.model.mlflow
    cache_dir = Path(mlflow_cfg.cache_dir)
    ensure_dir(cache_dir)

    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    logger.info(f"MLflow tracking URI: {mlflow_cfg.tracking_uri}")

    model_uri = mlflow_cfg.model_uri

    logger.info(f"Downloading artifacts from MLflow: {model_uri}")

    # Download entire export directory to cache
    try:
        local_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=str(cache_dir),
        )
        artifacts_dir = Path(local_dir)

        # Find ONNX file
        artifact_subpath = mlflow_cfg.artifact_subpath
        onnx_path = artifacts_dir / artifact_subpath
        if not onnx_path.exists():
            # Try direct path if subpath doesn't work
            onnx_files = list(artifacts_dir.glob("*.onnx"))
            if onnx_files:
                onnx_path = onnx_files[0]
            else:
                raise FileNotFoundError(f"No ONNX file found in {artifacts_dir}")

        logger.info(f"Model downloaded to: {onnx_path}")
        logger.info(f"Artifacts directory: {artifacts_dir}")
        return onnx_path, artifacts_dir
    except Exception as e:
        logger.error(f"Failed to download artifacts from MLflow: {e}")
        raise


def load_model_from_local(cfg: DictConfig) -> Path:
    """Load ONNX model from local filesystem.

    Args:
        cfg: Hydra configuration with model.local settings

    Returns:
        Path to ONNX file

    Raises:
        FileNotFoundError: If ONNX file not found
        ValueError: If local.onnx_path not configured
    """
    local_cfg = cfg.model.local
    onnx_path_str = local_cfg.onnx_path

    if not onnx_path_str:
        raise ValueError(
            "model.local.onnx_path must be set when source=local. "
            "Example: inference.model.local.onnx_path=runs/<run_id>/export/model.onnx"
        )

    onnx_path = Path(onnx_path_str)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

    logger.info(f"Using local ONNX model: {onnx_path}")
    return onnx_path


def create_onnx_session(
    onnx_path: Path,
    providers: list[str],
) -> ort.InferenceSession:
    """Create ONNX Runtime inference session.

    Args:
        onnx_path: Path to ONNX model file
        providers: List of execution providers (priority order)

    Returns:
        ONNX Runtime InferenceSession
    """
    logger.info(f"Creating ONNX session with providers: {providers}")

    # Filter to only available providers
    available_providers = ort.get_available_providers()
    filtered_providers = [p for p in providers if p in available_providers]

    if not filtered_providers:
        logger.warning(
            f"None of requested providers {providers} available. "
            f"Available: {available_providers}. Falling back to CPU."
        )
        filtered_providers = ["CPUExecutionProvider"]
    elif filtered_providers != providers:
        logger.info(
            f"Some providers not available. Using: {filtered_providers} "
            f"(requested: {providers})"
        )

    session = ort.InferenceSession(str(onnx_path), providers=filtered_providers)

    # Log session info
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    logger.info(f"ONNX session created. Inputs: {input_names}, Outputs: {output_names}")

    return session


def load_norm_stats(path: str | Path) -> tuple[list[float], list[float]]:
    """Load normalization statistics from JSON file.

    Args:
        path: Path to norm_stats.json

    Returns:
        Tuple of (mean, std) lists per channel
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {path}")

    with open(path) as f:
        stats = json.load(f)

    mean = stats["mean"]
    std = stats["std"]
    logger.info(f"Loaded normalization stats from {path}")
    return mean, std


def load_model(cfg: DictConfig) -> tuple[ort.InferenceSession, Path, Path | None]:
    """Load ONNX model based on configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (ONNX session, path to model file, artifacts_dir or None)
        artifacts_dir is returned for MLflow source (contains norm_stats.json)
    """
    source = cfg.model.source
    providers = list(cfg.runtime.providers)

    artifacts_dir = None
    if source == "mlflow":
        onnx_path, artifacts_dir = load_model_from_mlflow(cfg)
    elif source == "local":
        onnx_path = load_model_from_local(cfg)
    else:
        raise ValueError(f"Unknown model source: {source}. Use 'mlflow' or 'local'.")

    session = create_onnx_session(onnx_path, providers)
    return session, onnx_path, artifacts_dir
