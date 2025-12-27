import os
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd
import rasterio.transform
import yaml
from rasterio.windows import Window
from tqdm import tqdm

from data_preparation.patchify.io import discover_tiles, read_manifest
from data_preparation.patchify.stats import (
    compute_cloud_frac,
    compute_label_stats,
    compute_valid_frac,
    is_usable,
)
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@click.command()
@click.option("--config", type=click.Path(exists=True), required=True, help="Path to config YAML")
@click.option("--limit-tiles", type=int, help="Limit number of tiles to process (for dev)")
@click.option("--skip-unusable", is_flag=True, help="Skip writing unusable patches")
@click.option("--force", is_flag=True, help="Force reprocessing of all tiles")
def main(config: str, limit_tiles: int | None, skip_unusable: bool, force: bool):
    with open(config) as f:
        cfg_full = yaml.safe_load(f)
        cfg = cfg_full.get("patchify", cfg_full)  # Fallback to root for backward compatibility

    # Setup logging
    setup_logging()
    # Suppress noisy rasterio warnings
    import logging

    logging.getLogger("rasterio._env").setLevel(logging.ERROR)

    repo_root = Path.cwd()

    # Extract params
    imagery_root = Path(cfg.get("imagery_root", "data/imagery"))
    labels_root = Path(cfg.get("labels_root", "data/labels"))
    output_root = Path(cfg.get("output_root", "data/patches"))
    aoi_path = Path(cfg.get("aoi_path", "data/aoi.geojson"))
    index_path = Path(cfg.get("dataset_index_path", "data/index/dataset_index.csv"))

    spectral_name = cfg.get("spectral_name", "spectral.tif")
    labels_name = cfg.get("labels_name", "labels.tif")
    scl_name = cfg.get("scl_name", "scl.tif")
    mask_name = cfg.get("mask_name", "mask.tif")

    patch_size = cfg.get("patch_size", 256)
    stride = cfg.get("stride", 256)

    cloud_scl_codes = cfg.get(
        "cloud_scl_codes", [3, 8, 9, 10, 11]
    )  # Default SCL cloud/shadow codes
    min_valid_frac = cfg.get("min_valid_frac", 0.90)
    max_cloud_frac = cfg.get("max_cloud_frac", 0.10)
    ignore_label_values = cfg.get("ignore_label_values", [])
    compression = cfg.get("compression", "LZW")

    # Load AOI for correct aoi_id lookup
    aoi_map = {}
    if aoi_path.exists():
        try:
            aoi_gdf = gpd.read_file(aoi_path)
            # Map iso_a3 code to aoi_id
            # The country variable in the loop is the iso_a3 code (e.g. FRA, ESP)
            if "iso_a3" in aoi_gdf.columns and "aoi_id" in aoi_gdf.columns:
                aoi_map = dict(zip(aoi_gdf["iso_a3"], aoi_gdf["aoi_id"], strict=False))
                logger.info("Loaded AOI mapping for %d countries.", len(aoi_map))
        except Exception as e:
            logger.warning("Could not load AOI mapping: %s", e)

    # Discovery
    tiles = discover_tiles(
        imagery_root=imagery_root,
        labels_root=labels_root,
        spectral_name=spectral_name,
        labels_name=labels_name,
        scl_name=scl_name,
        mask_name=mask_name,
    )

    if limit_tiles:
        tiles = tiles[:limit_tiles]

    # Load existing index for idempotency/resuming
    existing_index_df = pd.DataFrame()
    processed_tile_ids = set()

    if index_path.exists() and not force:
        try:
            existing_index_df = pd.read_csv(index_path)
            match_mask = (existing_index_df["patch_size"] == patch_size) & (
                existing_index_df["stride"] == stride
            )
            valid_existing = existing_index_df[match_mask]
            processed_tile_ids = set(valid_existing["tile_id"].unique())
            logger.info(
                "Found %d existing tiles in index with matching config. They will be skipped.",
                len(processed_tile_ids),
            )
        except Exception as e:
            logger.warning("Could not read existing index, starting fresh: %s", e)

    # Filter tiles to process
    tiles_to_process = [t for t in tiles if t["tile_id"] not in processed_tile_ids]

    if not tiles_to_process:
        logger.info("All tiles already processed. Nothing to do.")
        return

    all_patch_records = []
    skipped_tiles = []

    # Create output dirs
    spectral_out_dir = output_root / "spectral"
    labels_out_dir = output_root / "labels"
    spectral_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    for tile_info in tqdm(tiles_to_process, desc="Patchifying tiles"):
        tile_id = tile_info["tile_id"]
        country = tile_info["country"]

        try:
            # 1. Alignment Safety Assertions
            with (
                rasterio.open(tile_info["spectral"]) as src_spectral,
                rasterio.open(tile_info["labels"]) as src_labels,
                rasterio.open(tile_info["scl"]) as src_scl,
                rasterio.open(tile_info["mask"]) as src_mask,
            ):
                # Check CRS
                if src_spectral.crs != src_labels.crs:
                    raise ValueError(
                        f"CRS mismatch for tile {tile_id}: {src_spectral.crs} != {src_labels.crs}"
                    )

                # Check Transform (exact equality)
                if src_spectral.transform != src_labels.transform:
                    raise ValueError(f"Transform mismatch for tile {tile_id}")

                # Check Shapes
                if src_spectral.shape != src_labels.shape:
                    raise ValueError(
                        f"Shape mismatch for tile {tile_id}: "
                        f"{src_spectral.shape} != {src_labels.shape}"
                    )

                H, W = src_spectral.shape

                # Read metadata from manifest
                manifest = read_manifest(tile_info["manifest"])
                time_data = manifest.get("time")
                acq_start = time_data.get("start") if isinstance(time_data, dict) else ""
                acq_end = time_data.get("end") if isinstance(time_data, dict) else ""
                mosaic_method = manifest.get("mosaic")

                # Get correct aoi_id
                aoi_id = aoi_map.get(country, "")
                if not aoi_id:
                    # Fallback to manifest if available or empty string
                    aoi_id = manifest.get("aoi_id", "")

                # 2. Window Enumeration (deterministic row-major)
                windows = []
                for row_off in range(0, H - patch_size + 1, stride):
                    for col_off in range(0, W - patch_size + 1, stride):
                        windows.append((row_off, col_off))

                tile_patches_written = 0
                tile_usable_count = 0

                # 3. Process each window
                for row_off, col_off in windows:
                    window = Window(col_off, row_off, patch_size, patch_size)

                    # Deterministic Patch ID
                    patch_id = f"{tile_id}_r{row_off}_c{col_off}_p{patch_size}_s{stride}"

                    # Write spectral path
                    spec_out_path = spectral_out_dir / country / tile_id / f"{patch_id}.tif"
                    label_out_path = labels_out_dir / country / tile_id / f"{patch_id}.tif"

                    # Check if patches already exist to skip writing
                    if not force and spec_out_path.exists() and label_out_path.exists():
                        # We still need stats. If they aren't in processed_tile_ids,
                        # we either have to read the file or re-evaluate.
                        # Since we filtered tiles_to_process by CSV already,
                        # hitting this means the CSV was missing but files existed.
                        pass

                    # Read patches
                    patch_spectral = src_spectral.read(window=window)
                    patch_labels = src_labels.read(1, window=window)
                    patch_scl = src_scl.read(1, window=window)
                    patch_mask = src_mask.read(1, window=window)

                    # Assert shapes
                    assert patch_spectral.shape == (src_spectral.count, patch_size, patch_size)
                    assert patch_labels.shape == (patch_size, patch_size)

                    # QA Stats
                    vf = compute_valid_frac(patch_mask)
                    cf = compute_cloud_frac(patch_scl, cloud_scl_codes)
                    ls = compute_label_stats(patch_labels, ignore_label_values)

                    usable = is_usable(vf, cf, min_valid_frac, max_cloud_frac)
                    if usable:
                        tile_usable_count += 1

                    if skip_unusable and not usable:
                        continue

                    # Write files
                    spec_out_path.parent.mkdir(parents=True, exist_ok=True)
                    label_out_path.parent.mkdir(parents=True, exist_ok=True)

                    spec_meta = src_spectral.meta.copy()
                    spec_meta.update(
                        {
                            "height": patch_size,
                            "width": patch_size,
                            "transform": rasterio.windows.transform(window, src_spectral.transform),
                            "compress": compression,
                        }
                    )
                    with rasterio.open(spec_out_path, "w", **spec_meta) as dst:
                        dst.write(patch_spectral)

                    label_meta = src_labels.meta.copy()
                    label_meta.update(
                        {
                            "height": patch_size,
                            "width": patch_size,
                            "transform": rasterio.windows.transform(window, src_labels.transform),
                            "compress": compression,
                        }
                    )
                    with rasterio.open(label_out_path, "w", **label_meta) as dst:
                        dst.write(patch_labels, 1)

                    # Spatial Anchors (Stage B integration)
                    transform = rasterio.windows.transform(window, src_spectral.transform)
                    cx, cy = rasterio.transform.xy(
                        transform, patch_size / 2.0, patch_size / 2.0, offset="center"
                    )

                    # Collect Record
                    patch_rec = {
                        "patch_id": patch_id,
                        "tile_id": tile_id,
                        "country": country,
                        "aoi_id": aoi_id,
                        "row_off": row_off,
                        "col_off": col_off,
                        "patch_size": patch_size,
                        "stride": stride,
                        "spectral_path": os.path.relpath(spec_out_path, repo_root),
                        "label_path": os.path.relpath(label_out_path, repo_root),
                        "valid_frac": vf,
                        "cloud_frac": cf,
                        "unique_classes": ls["unique_classes"],
                        "dominant_class": ls["dominant_class"],
                        "dominant_frac": ls["dominant_frac"],
                        "is_usable": usable,
                        "acq_start": acq_start,
                        "acq_end": acq_end,
                        "mosaic_method": mosaic_method,
                        "center_x": float(cx),
                        "center_y": float(cy),
                    }
                    all_patch_records.append(patch_rec)
                    tile_patches_written += 1

        except Exception as e:
            logger.error("Failed to process tile %s: %s", tile_id, str(e))
            skipped_tiles.append(tile_id)
            # Re-raise if it's an alignment error or assertion error
            if (
                "mismatch" in str(e).lower()
                or "assertion" in str(e).lower()
                or isinstance(e, AssertionError)
            ):
                raise
            continue

    # Merge with existing index
    new_records_df = pd.DataFrame(all_patch_records)

    if not existing_index_df.empty:
        # Keep only existing records for tiles we didn't just process
        existing_to_keep = existing_index_df[
            ~existing_index_df["tile_id"].isin(new_records_df["tile_id"].unique())
        ]
        final_df = pd.concat([existing_to_keep, new_records_df], ignore_index=True)
    else:
        final_df = new_records_df

    # Write Index
    if not final_df.empty:
        # Ensure deterministic order
        final_df = final_df.sort_values(["tile_id", "row_off", "col_off"])
        index_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(index_path, index=False)
        logger.info("Updated %d patch records in %s", len(final_df), index_path)

    # Write skipped tiles
    skipped_path = output_root / "skipped_tiles.txt"
    with open(skipped_path, "w") as f:
        for tid in skipped_tiles:
            f.write(f"{tid}\n")

    logger.info("Patchification complete.")


if __name__ == "__main__":
    main()
