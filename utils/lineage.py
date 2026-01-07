"""Lineage metadata computation and persistence.

This module computes and persists cryptographic hashes and metadata
for experiment reproducibility and data lineage tracking.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from utils.logging import get_logger
from utils.run_id import get_git_short_hash

logger = get_logger(__name__)


def compute_file_sha256(file_path: str | Path) -> str | None:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file to hash

    Returns:
        Hex-encoded SHA256 hash, or None if file doesn't exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"File not found for hashing: {file_path}")
        return None

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def compute_lineage_metadata(
    cfg: DictConfig,
    seed: int,
) -> dict[str, Any]:
    """Compute lineage metadata for experiment reproducibility.

    Computes SHA256 hashes of key data files and captures experiment
    configuration for full reproducibility tracking.

    Args:
        cfg: Hydra configuration object
        seed: Random seed used for the experiment

    Returns:
        Dictionary containing lineage metadata.
    """
    lineage: dict[str, Any] = {
        "seed": seed,
        "git_commit": get_git_short_hash(),
        "hashes": {},
        "split_definition": {},
    }

    # Hash dataset index
    index_path = cfg.data.get("index_path")
    if index_path:
        lineage["hashes"]["dataset_index"] = compute_file_sha256(index_path)
        lineage["paths"] = {"dataset_index": str(index_path)}

    # Hash normalization stats
    norm_stats_path = cfg.data.get("norm_stats_path")
    if norm_stats_path:
        lineage["hashes"]["norm_stats"] = compute_file_sha256(norm_stats_path)
        if "paths" not in lineage:
            lineage["paths"] = {}
        lineage["paths"]["norm_stats"] = str(norm_stats_path)

    # Hash class weights (if used)
    class_weights_path = cfg.module.loss.get("class_weights_path")
    if class_weights_path and Path(class_weights_path).exists():
        lineage["hashes"]["class_weights"] = compute_file_sha256(class_weights_path)
        lineage["paths"]["class_weights"] = str(class_weights_path)

    # Capture split definition from config
    if hasattr(cfg, "split") and cfg.split:
        split_config = cfg.split.get("config", [])
        if split_config:
            lineage["split_definition"] = OmegaConf.to_container(split_config)

    # Add model configuration summary
    if hasattr(cfg, "model"):
        lineage["model"] = {
            "name": cfg.model.get("name", "unknown"),
            "num_classes": cfg.model.get("num_classes"),
            "in_channels": cfg.model.get("in_channels"),
        }

    # Add training configuration summary
    if hasattr(cfg, "trainer"):
        lineage["training"] = {
            "max_epochs": cfg.trainer.get("max_epochs"),
            "precision": cfg.trainer.get("precision"),
        }

    if hasattr(cfg, "module"):
        lineage["training"]["lr"] = cfg.module.get("lr")
        lineage["training"]["loss_type"] = cfg.module.loss.get("name")

    return lineage


def save_lineage(
    lineage: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save lineage metadata to JSON file.

    Args:
        lineage: Lineage metadata dictionary
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(lineage, f, indent=2)

    logger.info(f"Saved lineage metadata to {output_path}")


def get_lineage_tags(lineage: dict[str, Any]) -> dict[str, str]:
    """Extract key lineage info as tags for MLflow.

    Args:
        lineage: Lineage metadata dictionary

    Returns:
        Dictionary of string tags suitable for MLflow.
    """
    tags = {
        "seed": str(lineage.get("seed", "")),
        "git_commit": lineage.get("git_commit", ""),
    }

    # Add hash prefixes (first 8 chars) as tags
    hashes = lineage.get("hashes", {})
    for key, value in hashes.items():
        if value:
            tags[f"hash_{key}"] = value[:8]

    return tags
