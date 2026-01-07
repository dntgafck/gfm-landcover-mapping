"""Hydra entrypoint for starting the inference server.

Usage:
    python -m inference.server
    python -m inference.server inference.model.source=mlflow
    python -m inference.server inference.model.local.onnx_path=runs/lcseg-xxx/export/model.onnx
"""

import sys
from pathlib import Path

import uvicorn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from utils.hydra_config import register_resolvers
from utils.logging import get_logger, setup_logging

# Register custom resolvers
register_resolvers()

# Absolute path to configs directory
CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load Hydra configuration for inference.

    Args:
        overrides: List of Hydra-style overrides

    Returns:
        Loaded configuration
    """
    # Clear any existing Hydra state
    GlobalHydra.instance().clear()

    # Initialize and compose
    initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3")
    cfg = compose(config_name="inference", overrides=overrides or [])

    return cfg


def main() -> None:
    """Main entrypoint for inference server."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    # Parse command line arguments as Hydra overrides
    overrides = sys.argv[1:] if len(sys.argv) > 1 else []

    # Load config
    logger.info("Loading inference configuration...")
    cfg = load_config(overrides)

    # Log configuration summary
    logger.info("=" * 60)
    logger.info("Inference Server Configuration")
    logger.info("=" * 60)
    logger.info(f"Model source: {cfg.model.source}")
    if cfg.model.source == "mlflow":
        logger.info(f"MLflow URI: {cfg.model.mlflow.model_uri}")
    else:
        logger.info(f"Local path: {cfg.model.local.onnx_path}")
    logger.info(f"Data: {cfg.data.imagery_root}")
    logger.info(f"DVC enabled: {cfg.dvc.enabled}")
    logger.info(f"Runtime providers: {cfg.runtime.providers}")
    logger.info(f"Server: {cfg.server.host}:{cfg.server.port}")
    logger.info("=" * 60)

    # Import app after config is loaded to allow proper initialization
    from inference.app import create_app

    # Create app with config
    app = create_app(cfg)

    # Run uvicorn server
    logger.info(f"Starting uvicorn on http://{cfg.server.host}:{cfg.server.port}")
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.server.log_level,
        reload=cfg.server.reload,
    )


if __name__ == "__main__":
    main()
