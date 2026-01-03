import json
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rasterio
from omegaconf import DictConfig
from tqdm import tqdm

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

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    df = pd.read_csv(index_path)

    # Filter
    if "split" in df.columns:
        df = df[df["split"] == split_name]
        logger.info(f"Filtered for split='{split_name}': {len(df)} patches remain.")

    if "cloud_frac" in df.columns:
        df = df[df["cloud_frac"] <= cloud_frac_max]
        logger.info(f"Filtered for cloud_frac<={cloud_frac_max}: {len(df)} patches remain.")

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
    """
    counts = np.zeros(num_classes, dtype=np.int64)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing class frequencies"):
        label_path = repo_root / row["label_path"]

        if not label_path.exists():
            logger.warning(f"Label file missing: {label_path}, skipping.")
            continue

        try:
            with rasterio.open(label_path) as src:
                mask = src.read(1)

                # Remap to [0, 10] and ignore 255
                # Note: The logic here should match LandCoverPatchDataset.__getitem__
                # However, the dataset does remapping on the fly.
                # If the saved labels are still ESA WorldCover [10, 20...], we remap.
                # BUT if we want to be consistent, we should probably use the same mapping.

                # ESA WorldCover mapping:
                # 10->0, 20->1, 30->2, 40->3, 50->4, 60->5, 70->6, 80->7, 90->8, 95->9, 100->10
                remapped = np.full_like(mask, ignore_index, dtype=np.int64)
                remapped[mask == 10] = 0
                remapped[mask == 20] = 1
                remapped[mask == 30] = 2
                remapped[mask == 40] = 3
                remapped[mask == 50] = 4
                remapped[mask == 60] = 5
                remapped[mask == 70] = 6
                remapped[mask == 80] = 7
                remapped[mask == 90] = 8
                remapped[mask == 95] = 9
                remapped[mask == 100] = 10

                unique, counts_unique = np.unique(remapped, return_counts=True)
                for val, count in zip(unique, counts_unique, strict=False):
                    if val != ignore_index and 0 <= val < num_classes:
                        counts[val] += count

        except Exception as e:
            logger.error(f"Error reading {label_path}: {e}")
            continue

    return counts


if __name__ == "__main__":
    setup_logging()

    @hydra.main(config_path="../../conf", config_name="params", version_base="1.2")
    def run_as_script(cfg: DictConfig):
        # Configuration for class weights
        stage_cfg = cfg.get("class_weights", {})
        if not stage_cfg:
            logger.error("No class_weights config found in params.yaml (Hydra context).")
            return

        compute_class_weights(
            index_path=stage_cfg.get("input_index", "data/index/dataset_index_with_split.csv"),
            output_path=stage_cfg.get("output_path", "data/stats/class_weights.json"),
            split_name=stage_cfg.get("split_name", "train"),
            cloud_frac_max=stage_cfg.get("cloud_fraction_max", 0.20),
            num_classes=stage_cfg.get("num_classes", 11),
            ignore_index=stage_cfg.get("ignore_index", 255),
            min_weight=stage_cfg.get("min_weight", 0.25),
            max_weight=stage_cfg.get("max_weight", 4.0),
        )

    run_as_script()
