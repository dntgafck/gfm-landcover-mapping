import json
from datetime import datetime
from pathlib import Path

import dvc.api
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def compute_class_weights(
    index_path: Path | str,
    output_path: Path | str,
    split_name: str = "train",
    cloud_frac_max: float = 0.20,
    num_classes: int = 11,
    ignore_index: int = 255,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
    repo_root: Path | None = None,
) -> list[float]:
    """
    Computes class-balanced weights from the index file and returns them.
    Also saves to JSON output_path.
    """
    index_path = Path(index_path)
    output_path = Path(output_path)
    if repo_root is None:
        repo_root = Path.cwd()

    # Load Index via DVC
    logger.info(f"Loading index from {index_path} via dvc.api...")
    with dvc.api.open(str(index_path), repo=str(repo_root), mode="r") as f:
        df = pd.read_csv(f)

    # Filter
    if "split" in df.columns:
        df = df[df["split"] == split_name]
        logger.info(f"Filtered for split='{split_name}': {len(df)} patches remain.")

    if "cloud_frac" in df.columns:
        df = df[df["cloud_frac"] <= cloud_frac_max]
        logger.info(
            f"Filtered for cloud_frac<={cloud_frac_max}: {len(df)} patches remain."
        )

    if len(df) == 0:
        raise ValueError("No patches matched filter criteria.")

    # Compute Frequencies
    counts = compute_class_frequencies(df, repo_root, num_classes, ignore_index)

    # Calculate Weights: Inverse sqrt frequency
    total_pixels = counts.sum()
    frequencies = counts / (total_pixels + 1e-8)
    weights = 1.0 / np.sqrt(frequencies + 1e-8)

    # Clamp weights
    weights = np.clip(weights, min_weight, max_weight)

    # Normalize again after clamping to keep scale reasonable
    weights = weights / weights.mean()

    # Save to JSON
    output_result = {
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "class_counts": counts.tolist(),
        "class_frequencies": frequencies.tolist(),
        "class_weights": weights.tolist(),
        "computed_on": {
            "split": split_name,
            "cloud_frac_max": cloud_frac_max,
            "n_patches": len(df),
            "total_pixels": int(total_pixels),
        },
        "params": {
            "min_weight": min_weight,
            "max_weight": max_weight,
        },
        "created_at_utc": datetime.utcnow().isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_result, f, indent=2)

    logger.info(f"Class weights saved to {output_path}")
    from typing import cast

    return cast(list[float], weights.tolist())


def compute_class_frequencies(
    df: pd.DataFrame,
    repo_root: Path,
    num_classes: int = 11,
    ignore_index: int = 255,
) -> np.ndarray:
    """
    Computes total pixel counts per class across all patches in the dataframe.
    Uses pre-computed stats from the index if available.
    """
    counts = np.zeros(num_classes, dtype=np.int64)

    # Label mapping (ESA WorldCover -> Contiguous [0, 10])
    label_map = {
        10: 0,
        20: 1,
        30: 2,
        40: 3,
        50: 4,
        60: 5,
        70: 6,
        80: 7,
        90: 8,
        95: 9,
        100: 10,
    }

    if "class_counts" not in df.columns:
        raise ValueError(
            "Column 'class_counts' missing in index. "
            "Please re-run the 'build_index' stage (07_build_index.py)."
        )

    logger.info("Using pre-computed class_counts from index.")
    for _, row in df.iterrows():
        try:
            # class_counts is stored as a JSON string
            patch_counts = json.loads(row["class_counts"])
            for raw_val, count in patch_counts.items():
                raw_val_int = int(raw_val)
                if raw_val_int in label_map:
                    mapped_val = label_map[raw_val_int]
                    if 0 <= mapped_val < num_classes:
                        counts[mapped_val] += count
        except Exception as e:
            logger.warning(
                f"Failed to parse class_counts for patch {row.get('patch_id')}: {e}"
            )
            continue
    return counts


if __name__ == "__main__":
    setup_logging()

    @hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
    def run_as_script(cfg: DictConfig):
        # Configuration for class weights
        stage_cfg = cfg.get("class_weights", {})
        if not stage_cfg:
            logger.error(
                "No class_weights config found in params.yaml (Hydra context)."
            )
            return

        compute_class_weights(
            index_path=stage_cfg.get(
                "input_index", "data/index/dataset_index_with_split.csv"
            ),
            output_path=stage_cfg.get("output_path", "data/stats/class_weights.json"),
            split_name=stage_cfg.get("split_name", "train"),
            cloud_frac_max=stage_cfg.get("cloud_fraction_max", 0.20),
            num_classes=stage_cfg.get("num_classes", 11),
            ignore_index=stage_cfg.get("ignore_index", 255),
            min_weight=stage_cfg.get("min_weight", 0.25),
            max_weight=stage_cfg.get("max_weight", 4.0),
        )

    run_as_script()
