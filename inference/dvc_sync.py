"""DVC sync logic for pulling inference data at startup."""

from pathlib import Path

from dvc.repo import Repo
from omegaconf import DictConfig

from utils.logging import get_logger

logger = get_logger(__name__)


def pull_inference_data(cfg: DictConfig) -> bool:
    """Pull inference data from DVC remote.

    Args:
        cfg: Hydra configuration with sync settings

    Returns:
        True if pull succeeded or data already present, False on failure
    """
    sync_cfg = cfg.sync

    if not sync_cfg.enabled:
        logger.info("DVC pull disabled in config")
        return True

    pull_targets = list(sync_cfg.pull_targets)
    if not pull_targets:
        logger.warning("No DVC pull targets specified")
        return True

    logger.info(f"Pulling DVC targets: {pull_targets}")

    try:
        repo = Repo()
        # Pull specified targets
        result = repo.pull(targets=pull_targets)
        logger.info(f"DVC pull complete: {result}")
        return True
    except Exception as e:
        logger.error(f"DVC pull failed: {e}")
        return False


def validate_data_dirs(cfg: DictConfig) -> bool:
    """Validate that required data directories exist after DVC pull.

    Args:
        cfg: Hydra configuration with data settings

    Returns:
        True if all required directories exist

    Raises:
        FileNotFoundError: If required directories are missing
    """
    data_cfg = cfg.data
    imagery_root = Path(data_cfg.imagery_root)
    labels_root = Path(data_cfg.labels_root)

    missing = []
    if not imagery_root.exists():
        missing.append(str(imagery_root))
    if not labels_root.exists():
        missing.append(str(labels_root))

    if missing:
        msg = f"Required data directories not found: {missing}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(
        f"Data directories validated: imagery={imagery_root}, labels={labels_root}"
    )
    return True


def sync_inference_data(cfg: DictConfig) -> None:
    """Main entry point: pull data and validate directories.

    Args:
        cfg: Hydra configuration

    Raises:
        FileNotFoundError: If data directories missing after pull
        RuntimeError: If DVC pull fails
    """
    if cfg.sync.enabled:
        success = pull_inference_data(cfg)
        if not success:
            raise RuntimeError("DVC pull failed - cannot start inference server")

    validate_data_dirs(cfg)
    logger.info("Inference data sync complete")
