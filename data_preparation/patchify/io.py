import json
from pathlib import Path

from utils.logging import get_logger

logger = get_logger(__name__)


def discover_tiles(
    imagery_root: Path,
    labels_root: Path,
    spectral_name: str = "spectral.tif",
    labels_name: str = "labels.tif",
    scl_name: str = "scl.tif",
    mask_name: str = "mask.tif",
    manifest_name: str = "manifest.json",
) -> list[dict[str, str]]:
    """
    Scans the imagery and labels roots to discover tiles.
    Returns a list of dictionaries with paths for each tile.
    """
    tiles = []

    # We assume the structure is <root>/<country>/<tile_id>/...
    # We find all manifest.json files as a proxy for tiles in imagery
    manifest_paths = list(imagery_root.glob(f"**/{manifest_name}"))

    for manifest_path in manifest_paths:
        tile_dir = manifest_path.parent
        tile_id = tile_dir.name
        country_code = tile_dir.parent.name

        # Construct paths
        spectral_path = tile_dir / spectral_name
        scl_path = tile_dir / scl_name
        mask_path = tile_dir / mask_name

        # Label path might follow same nesting or flattened
        # Requirement says labels root and labels filename
        # Based on exploration: labels_root/<country>/<tile_id>/labels.tif
        label_path = labels_root / country_code / tile_id / labels_name

        # Check if required files exist
        missing = []
        if not spectral_path.exists():
            missing.append(spectral_name)
        if not scl_path.exists():
            missing.append(scl_name)
        if not mask_path.exists():
            missing.append(mask_name)
        if not label_path.exists():
            missing.append(labels_name)

        if missing:
            logger.warning(
                "Skipping tile %s (country %s) due to missing files: %s",
                tile_id,
                country_code,
                ", ".join(missing),
            )
            continue

        tiles.append(
            {
                "tile_id": tile_id,
                "country": country_code,
                "spectral": str(spectral_path),
                "scl": str(scl_path),
                "mask": str(mask_path),
                "labels": str(label_path),
                "manifest": str(manifest_path),
            }
        )

    logger.info("Discovered %d valid tiles.", len(tiles))
    return tiles


def read_manifest(manifest_path: str) -> dict:
    """Reads the manifest.json file."""
    with open(manifest_path) as f:
        data: dict = json.load(f)
        return data
