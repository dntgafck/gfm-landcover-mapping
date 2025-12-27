from pathlib import Path
from typing import Any

import rasterio
import rasterio.transform


def get_patch_spatial_anchors(patch_path: str | Path) -> tuple[float, float]:
    """
    Computes the center (x, y) coordinates of a patch in its native CRS.
    """
    with rasterio.open(patch_path) as src:
        bounds = src.bounds
        # Center of the bounds
        cx = (bounds.left + bounds.right) / 2.0
        cy = (bounds.top + bounds.bottom) / 2.0
        return cx, cy


def extract_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """
    Normalizes metadata from various manifest versions.
    """
    # Extract ID
    aoi_id = manifest.get("aoi_id") or manifest.get("aoi", "")
    country = manifest.get("country") or manifest.get("country_code", "")

    # Extract Time
    time_data = manifest.get("time")
    if isinstance(time_data, dict):
        acq_start = time_data.get("start", "")
        acq_end = time_data.get("end", "")
    else:
        # Fallback for old simple string time
        acq_start = str(time_data) if time_data else ""
        acq_end = acq_start

    # Extract Method
    mosaic_method = manifest.get("mosaic") or manifest.get("mosaicking_order", "")

    return {
        "aoi_id": aoi_id,
        "country": country,
        "acq_start": acq_start,
        "acq_end": acq_end,
        "mosaic_method": mosaic_method,
    }
