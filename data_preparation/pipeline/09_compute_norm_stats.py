import json
import math
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rasterio
from dvc.repo import Repo
from omegaconf import DictConfig
from tqdm import tqdm

from utils.logging import get_logger, setup_logging
from utils.sampling import RandomSampler, Sampler, StratifiedSampler

logger = get_logger(__name__)


def ensure_dvc_files(paths: list[str | Path]):
    """
    Ensures that the specified files are available locally by pulling them from DVC.
    """
    paths_str = [str(p) for p in paths]
    logger.info("Ensuring DVC files are present: %s", paths_str)
    try:
        # Check if we are in a dvc repo
        if Path(".dvc").exists():
            repo = Repo(".")
            repo.pull(paths_str)
        else:
            logger.warning("Not inside a DVC repository. Skipping implicit pull.")
    except Exception as e:
        logger.warning("Failed to pull DVC files (might be local-only or error): %s", e)


def welford_update_batch(
    existing_stats: dict[int, tuple[int, float, float]], new_batch: np.ndarray
):
    """
    Updates global statistics (count, mean, M2) for each band using a batch of new data.
    new_batch: (C, H, W) or (C, N_pixels)
    existing_stats: dict mapping band_idx -> (count, mean, M2)
    """
    # Flatten if needed: (C, N)
    if new_batch.ndim == 3:
        C, H, W = new_batch.shape
        new_batch = new_batch.reshape(C, -1)

    C, N = new_batch.shape

    for c in range(C):
        band_data = new_batch[c, :]

        # Filter out nodata (assuming 0 is nodata for Sentinel-2 spectral data)
        valid_mask = band_data != 0
        valid_data = band_data[valid_mask]

        n = valid_data.size
        if n == 0:
            continue

        new_mean = np.mean(valid_data)
        new_m2 = np.sum((valid_data - new_mean) ** 2)

        if c not in existing_stats:
            existing_stats[c] = (0, 0.0, 0.0)

        count, mean, m2 = existing_stats[c]

        delta = new_mean - mean

        updated_count = count + n
        updated_mean = mean + delta * n / updated_count
        updated_m2 = m2 + new_m2 + delta**2 * count * n / updated_count

        existing_stats[c] = (updated_count, updated_mean, updated_m2)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging()

    # Configuration
    stage_cfg = cfg.get("norm_stats", {})
    index_path = Path(stage_cfg.get("input_index", "data/index/dataset_index_with_split.csv"))
    output_path = Path(stage_cfg.get("output_path", "data/stats/norm_stats.json"))
    split_name = stage_cfg.get("split_name", "train")
    cloud_frac_max = stage_cfg.get("cloud_fraction_max", 0.20)
    bands = list(stage_cfg.get("bands", ["B02", "B03", "B04", "B08"]))

    sampling_cfg = stage_cfg.get("sampling", {})
    n_patches = sampling_cfg.get("n_patches", 2000)
    seed = sampling_cfg.get("seed", 42)

    repo_root = Path.cwd()

    logger.info("Output path: %s", output_path)

    # 1. Load Index
    ensure_dvc_files([index_path])

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    df = pd.read_csv(index_path)

    # 2. Filter
    # Check if 'split' column exists
    if "split" not in df.columns:
        logger.warning(
            f"'split' column missing in {index_path}. Using all data (dangerous if not intended)."
        )
    else:
        df = df[df["split"] == split_name]
        logger.info(f"Filtered for split='{split_name}': {len(df)} patches remain.")

    if "cloud_frac" in df.columns:
        df = df[df["cloud_frac"] <= cloud_frac_max]
        logger.info(f"Filtered for cloud_frac<={cloud_frac_max}: {len(df)} patches remain.")

    if len(df) == 0:
        raise ValueError("No patches matched filter criteria.")

    # 3. Deterministic Sampling
    # Sort first for stability before sampling
    if "patch_id" in df.columns:
        df = df.sort_values("patch_id")
    else:
        df = df.sort_values(df.columns[0])  # Fallback

    sampling_strategy = sampling_cfg.get("strategy", "random")

    if len(df) <= n_patches:
        logger.info(f"Using all {len(df)} available patches (requested {n_patches}).")
        sampled_df = df
    elif sampling_strategy == "stratified":
        stratify_col = sampling_cfg.get("stratify_by", "country")
        min_per_strata = sampling_cfg.get("min_per_strata", 50)

        sampler: Sampler = StratifiedSampler(
            stratify_by=stratify_col, min_per_strata=min_per_strata, seed=seed
        )
        sampled_df = sampler.sample(df, n_patches)
        sampled_df = sampled_df.sort_values("patch_id")

    else:
        # Random
        logger.info(f"Sampling {n_patches} patches from {len(df)} candidates (seed={seed}).")
        sampler = RandomSampler(seed=seed)
        sampled_df = sampler.sample(df, n_patches)
        sampled_df = sampled_df.sort_values("patch_id")

    # 4. Compute Stats
    # band_idx -> (count, mean, M2)
    stats_accumulator: dict[int, tuple[int, float, float]] = {}

    patches_processed = 0
    # sampled_patch_ids = sampled_df["patch_id"].tolist()

    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Computing stats"):
        rel_path = row["spectral_path"]
        abs_path = repo_root / rel_path

        if not abs_path.exists():
            # Try to pull individual file if missing?
            # Doing this per file might be slow if dvc pull is slow.
            # But usually we expect data to be present if main valid dvc pull ran.
            # Let's try to ensure it
            ensure_dvc_files([abs_path])

        if not abs_path.exists():
            logger.warning(f"File missing: {abs_path}, skipping.")
            continue

        try:
            with rasterio.open(abs_path) as src:
                # Check band count
                if src.count != len(bands):
                    logger.warning(
                        f"Band count mismatch in {rel_path}: expected {len(bands)}, got {src.count}"
                    )
                    # Usually we trust the file, maybe it has more/less bands.
                    # If it has 4 bands and we expect 4, good.

                # Read data
                # If n_pixels_per_patch > 0, we could sample here.
                # For now implementing full patch reading or simple subsampling if needed.
                # Task says N=2000 is small enough for full patch reading if efficient.

                # Check for pixel subsampling
                data = src.read()  # (C, H, W)

                # Validation
                if data.shape[0] != len(bands):
                    logger.warning(f" skipping {rel_path}: {data.shape[0]} bands != {len(bands)}")
                    continue

                welford_update_batch(stats_accumulator, data)
                patches_processed += 1

        except Exception as e:
            logger.error(f"Error reading {rel_path}: {e}")
            continue

    # 5. Finalize Stats
    results_mean = []
    results_std = []

    for c in range(len(bands)):
        if c in stats_accumulator:
            count, mean, m2 = stats_accumulator[c]
            if count < 2:
                std = 0.0
            else:
                std = math.sqrt(m2 / (count - 1))
            results_mean.append(float(mean))
            results_std.append(float(std))
        else:
            logger.warning(f"No data for band {bands[c]} (idx {c})")
            results_mean.append(0.0)
            results_std.append(0.0)

    # 6. Save Artifact
    output_result = {
        "bands": bands,
        "mean": results_mean,
        "std": results_std,
        "computed_on": {
            "split": split_name,
            "cloud_frac_max": cloud_frac_max,
            "index_path": str(index_path),
            "n_patches_available": len(df),
            "n_patches_sampled": patches_processed,  # actually processed
            "seed": seed,
            # "sampled_patch_ids": ... # potentially large, omitting for brevity or write separate
        },
        "created_at_utc": datetime.utcnow().isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_result, f, indent=2)

    logger.info(f"Stats saved to {output_path}")
    logger.info(f"Means: {results_mean}")
    logger.info(f"Stds:  {results_std}")


if __name__ == "__main__":
    main()
