from pathlib import Path

import hydra
import rasterio.transform
from omegaconf import DictConfig
from rasterio.windows import Window
from tqdm import tqdm

from data_preparation.patchify.io import discover_tiles
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Setup logging
    setup_logging()
    # Suppress noisy rasterio warnings
    import logging

    logging.getLogger("rasterio._env").setLevel(logging.ERROR)

    # Access params
    patchify_cfg = cfg.get("patchify", {})

    # Extract params
    imagery_root = Path(patchify_cfg.get("imagery_root", "data/imagery"))
    labels_root = Path(patchify_cfg.get("labels_root", "data/labels"))
    output_root = Path(patchify_cfg.get("output_root", "data/patches"))

    spectral_name = patchify_cfg.get("spectral_name", "spectral.tif")
    labels_name = patchify_cfg.get("labels_name", "labels.tif")
    scl_name = patchify_cfg.get("scl_name", "scl.tif")
    mask_name = patchify_cfg.get("mask_name", "mask.tif")

    patch_size = patchify_cfg.get("patch_size", 256)
    stride = patchify_cfg.get("stride", 256)
    compression = patchify_cfg.get("compression", "LZW")

    force = False  # DVC handles reruns typically

    # Discovery
    tiles = discover_tiles(
        imagery_root=imagery_root,
        labels_root=labels_root,
        spectral_name=spectral_name,
        labels_name=labels_name,
        scl_name=scl_name,
        mask_name=mask_name,
    )

    if not tiles:
        logger.info("No tiles found to process.")
        return

    skipped_tiles = []

    # Create output dirs
    spectral_out_dir = output_root / "spectral"
    labels_out_dir = output_root / "labels"
    spectral_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    for tile_info in tqdm(tiles, desc="Patchifying tiles"):
        tile_id = tile_info["tile_id"]
        country = tile_info["country"]

        try:
            # 1. Alignment Safety Assertions
            with (
                rasterio.open(tile_info["spectral"]) as src_spectral,
                rasterio.open(tile_info["labels"]) as src_labels,
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

                # 2. Window Enumeration (deterministic row-major)
                windows = []
                for row_off in range(0, H - patch_size + 1, stride):
                    for col_off in range(0, W - patch_size + 1, stride):
                        windows.append((row_off, col_off))

                # 3. Process each window
                for row_off, col_off in windows:
                    window = Window(col_off, row_off, patch_size, patch_size)

                    # Deterministic Patch ID
                    patch_id = (
                        f"{tile_id}_r{row_off}_c{col_off}_p{patch_size}_s{stride}"
                    )

                    # Write paths
                    spec_out_path = (
                        spectral_out_dir / country / tile_id / f"{patch_id}.tif"
                    )
                    label_out_path = (
                        labels_out_dir / country / tile_id / f"{patch_id}.tif"
                    )

                    # Check if patches already exist to skip writing
                    if not force and spec_out_path.exists() and label_out_path.exists():
                        continue

                    # Read patches
                    patch_spectral = src_spectral.read(window=window)
                    patch_labels = src_labels.read(1, window=window)

                    # Assert shapes
                    assert patch_spectral.shape == (
                        src_spectral.count,
                        patch_size,
                        patch_size,
                    )
                    assert patch_labels.shape == (patch_size, patch_size)

                    # Write files
                    spec_out_path.parent.mkdir(parents=True, exist_ok=True)
                    label_out_path.parent.mkdir(parents=True, exist_ok=True)

                    spec_meta = src_spectral.meta.copy()
                    spec_meta.update(
                        {
                            "height": patch_size,
                            "width": patch_size,
                            "transform": rasterio.windows.transform(
                                window, src_spectral.transform
                            ),
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
                            "transform": rasterio.windows.transform(
                                window, src_labels.transform
                            ),
                            "compress": compression,
                        }
                    )
                    with rasterio.open(label_out_path, "w", **label_meta) as dst:
                        dst.write(patch_labels, 1)

        except Exception as e:
            logger.error("Failed to process tile %s: %s", tile_id, str(e))
            skipped_tiles.append(tile_id)
            # Re-raise if it's an alignment error or assertion error
            error_msg = str(e).lower()
            is_mismatch = "mismatch" in error_msg
            is_assertion = "assertion" in error_msg or isinstance(e, AssertionError)

            if is_mismatch or is_assertion:
                raise
            continue

    # Write skipped tiles
    if skipped_tiles:
        skipped_path = output_root / "skipped_tiles.txt"
        with open(skipped_path, "w") as f:
            for tid in skipped_tiles:
                f.write(f"{tid}\n")

    logger.info("Patchification complete.")


if __name__ == "__main__":
    main()
