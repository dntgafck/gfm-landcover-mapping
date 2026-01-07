"""Run ID generation and directory management utilities.

This module provides a framework-agnostic way to create unique run directories
that serve as the single source of truth for all experiment outputs.
"""

import subprocess
from datetime import datetime
from pathlib import Path

from utils.logging import get_logger

logger = get_logger(__name__)


def get_git_short_hash() -> str:
    """Get the short (7-char) git revision hash.

    Returns:
        7-character git SHA, or 'unknown' if not in a git repo.
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short=7", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def generate_run_id(prefix: str = "lcseg") -> str:
    """Generate a unique run ID using timestamp and git hash.

    Format: {prefix}-YYYYMMDD-HHMMSS-{git_sha}
    Example: lcseg-20260107-001234-a1b2c3d

    Args:
        prefix: Optional prefix for the run ID (default: "lcseg")

    Returns:
        Unique run ID string.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    git_hash = get_git_short_hash()

    return f"{prefix}-{timestamp}-{git_hash}"


def create_run_directory(runs_root: str | Path, run_id: str) -> Path:
    """Create the complete run directory structure.

    Creates the following structure:
        {runs_root}/{run_id}/
            config/
            logs/
            checkpoints/
            artifacts/
                plots/
            export/

    Args:
        runs_root: Root directory for all runs (e.g., "runs")
        run_id: Unique run identifier

    Returns:
        Path to the created run directory.
    """
    runs_root = Path(runs_root)
    run_dir = runs_root / run_id

    # Define the complete directory structure
    subdirs = [
        "config",
        "logs",
        "checkpoints",
        "artifacts/plots",
        "export",
    ]

    # Create all directories
    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Created run directory: {run_dir}")

    return run_dir


def get_run_paths(run_dir: str | Path) -> dict[str, Path]:
    """Get a dictionary of standard paths within a run directory.

    Args:
        run_dir: Path to the run directory

    Returns:
        Dictionary mapping path names to Path objects.
    """
    run_dir = Path(run_dir)

    return {
        "root": run_dir,
        "config": run_dir / "config",
        "config_yaml": run_dir / "config" / "config.yaml",
        "overrides": run_dir / "config" / "overrides.txt",
        "logs": run_dir / "logs",
        "hydra_log": run_dir / "logs" / "run.log",
        "checkpoints": run_dir / "checkpoints",
        "artifacts": run_dir / "artifacts",
        "plots": run_dir / "artifacts" / "plots",
        "lineage": run_dir / "artifacts" / "lineage.json",
        "export": run_dir / "export",
        "onnx": run_dir / "export" / "model.onnx",
        "readme": run_dir / "README.txt",
    }
