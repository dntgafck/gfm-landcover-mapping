import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import geopandas as gpd
import hydra
import pandas as pd
import rasterio
from omegaconf import DictConfig, OmegaConf
from rasterio.windows import Window
from tqdm import tqdm

from data_preparation.index.utils import extract_metadata, get_patch_spatial_anchors
from data_preparation.patchify.io import discover_tiles, read_manifest
from data_preparation.patchify.stats import (
    compute_cloud_frac,
    compute_label_stats,
    compute_valid_frac,
)
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def process_tile(
    tile_info: dict[str, Any],
    patch_spectral_files: list[Path],
    index_cfg_dict: dict[str, Any],
    aoi_map: dict[str, str],
    patches_root: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """
    Processes all patches for a single tile.
    """
    # Suppress noisy rasterio warnings in workers
    import logging

    logging.getLogger("rasterio._env").setLevel(logging.ERROR)

    tile_id = tile_info["tile_id"]
    country = tile_info["country"]
    patch_size = index_cfg_dict.get("patch_size", 256)
    stride = index_cfg_dict.get("stride", 256)
    cloud_scl_codes = list(index_cfg_dict.get("cloud_scl_codes", [8, 9, 10, 11]))
    ignore_label_values = list(index_cfg_dict.get("ignore_label_values", [0]))

    tile_records = []

    try:
        # Read manifest and extract metadata
        manifest = read_manifest(tile_info["manifest"])
        meta = extract_metadata(manifest)

        # AOI ID override from map if available
        aoi_id = aoi_map.get(country, meta.get("aoi_id", ""))

        # Open tile-level SCL and Mask once
        with (
            rasterio.open(tile_info["scl"]) as src_scl,
            rasterio.open(tile_info["mask"]) as src_mask,
        ):
            for spec_path in patch_spectral_files:
                patch_id = spec_path.stem
                # Expected label path
                label_path = (
                    patches_root / "labels" / country / tile_id / f"{patch_id}.tif"
                )

                if not label_path.exists():
                    continue

                try:
                    # Patch ID format: {tile_id}_r{row_off}_c{col_off}_p{patch_size}_s{stride}
                    parts = patch_id.split("_")
                    row_off = int(parts[1][1:]) if parts[1].startswith("r") else None
                    col_off = int(parts[2][1:]) if parts[2].startswith("c") else None

                    if row_off is None or col_off is None:
                        continue

                    window = Window(col_off, row_off, patch_size, patch_size)

                    # Read from original tile's SCL and Mask
                    patch_scl = src_scl.read(1, window=window)
                    patch_mask = src_mask.read(1, window=window)

                    with rasterio.open(label_path) as src_label:
                        patch_labels = src_label.read(1)

                    # QA Stats
                    vf = compute_valid_frac(patch_mask)
                    cf = compute_cloud_frac(patch_scl, cloud_scl_codes)
                    ls = compute_label_stats(patch_labels, ignore_label_values)

                    # Spatial Anchors
                    cx, cy = get_patch_spatial_anchors(spec_path)

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
                        "spectral_path": os.path.relpath(spec_path, repo_root),
                        "label_path": os.path.relpath(label_path, repo_root),
                        "valid_frac": vf,
                        "cloud_frac": cf,
                        "unique_classes": ls["unique_classes"],
                        "dominant_class": ls["dominant_class"],
                        "dominant_frac": ls["dominant_frac"],
                        "class_counts": json.dumps(ls["class_counts"]),
                        "acq_start": meta["acq_start"],
                        "acq_end": meta["acq_end"],
                        "mosaic_method": meta["mosaic_method"],
                        "center_x": float(cx),
                        "center_y": float(cy),
                    }
                    tile_records.append(patch_rec)

                except Exception:
                    # Specific patch failure
                    continue

    except Exception:
        # Tile-level failure
        return []

    return tile_records


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging()

    # Suppress noisy rasterio warnings
    import logging

    logging.getLogger("rasterio._env").setLevel(logging.ERROR)

    repo_root = Path.cwd()
    index_cfg = cfg.get("index", {})

    # Extract params
    imagery_root = Path(index_cfg.get("imagery_root", "data/imagery"))
    labels_root = Path(index_cfg.get("labels_root", "data/labels"))
    patches_root = Path(index_cfg.get("output_root", "data/patches"))
    aoi_path = Path(index_cfg.get("aoi_path", "data/aoi.geojson"))
    index_out_path = Path(
        index_cfg.get("dataset_index_path", "data/index/dataset_index.csv")
    )
    num_workers = index_cfg.get("num_workers", 1)

    # Load AOI for correct aoi_id lookup
    aoi_map = {}
    if aoi_path.exists():
        try:
            aoi_gdf = gpd.read_file(aoi_path)
            if "iso_a3" in aoi_gdf.columns and "aoi_id" in aoi_gdf.columns:
                aoi_map = dict(zip(aoi_gdf["iso_a3"], aoi_gdf["aoi_id"], strict=False))
                logger.info("Loaded AOI mapping for %d countries.", len(aoi_map))
        except Exception as e:
            logger.warning("Could not load AOI mapping: %s", e)

    # 1. Discover Tiles
    tiles = discover_tiles(
        imagery_root=imagery_root,
        labels_root=labels_root,
        spectral_name=index_cfg.get("spectral_name", "spectral.tif"),
        labels_name=index_cfg.get("labels_name", "labels.tif"),
        scl_name=index_cfg.get("scl_name", "scl.tif"),
        mask_name=index_cfg.get("mask_name", "mask.tif"),
    )

    tile_data_map = {t["tile_id"]: t for t in tiles}

    # 2. Discover Patches
    patch_spectral_files = list((patches_root / "spectral").glob("**/*.tif"))
    logger.info("Found %d spectral patches.", len(patch_spectral_files))

    # Group patches by tile_id
    tile_to_patches: dict[str, list[Path]] = {}
    for spec_path in patch_spectral_files:
        tile_id = spec_path.parent.name
        if tile_id not in tile_to_patches:
            tile_to_patches[tile_id] = []
        tile_to_patches[tile_id].append(spec_path)

    all_patch_records = []

    # Convert Hydra config to plain dict for pickling safety
    index_cfg_dict = (
        OmegaConf.to_container(index_cfg, resolve=True)
        if isinstance(index_cfg, DictConfig)
        else index_cfg
    )

    if num_workers > 1:
        logger.info("Processing with %d workers...", num_workers)
        tasks = []
        for tile_id, patches in tile_to_patches.items():
            tile_info = tile_data_map.get(tile_id)
            if not tile_info:
                logger.warning("Tile info missing for %s, skipping.", tile_id)
                continue
            tasks.append(
                (tile_info, patches, index_cfg_dict, aoi_map, patches_root, repo_root)
            )

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_tile, *task) for task in tasks]
            for future in tqdm(futures, desc="Indexing tiles (parallel)"):
                all_patch_records.extend(future.result())
    else:
        logger.info("Processing sequentially...")
        for tile_id, patches in tqdm(tile_to_patches.items(), desc="Indexing tiles"):
            tile_info = tile_data_map.get(tile_id)
            if not tile_info:
                logger.warning("Tile info missing for %s, skipping.", tile_id)
                continue
            res = process_tile(
                tile_info, patches, index_cfg_dict, aoi_map, patches_root, repo_root
            )
            all_patch_records.extend(res)

    if not all_patch_records:
        logger.info("No patch records collected.")
        return

    final_df = pd.DataFrame(all_patch_records)
    final_df = final_df.sort_values(["tile_id", "row_off", "col_off"])

    index_out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(index_out_path, index=False)
    logger.info(
        "Created index with %d patch records in %s", len(final_df), index_out_path
    )


if __name__ == "__main__":
    main()
