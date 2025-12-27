import numpy as np
import rasterio
from rasterio.transform import from_origin

from data_preparation.index.manifest_reader import extract_metadata
from data_preparation.index.spatial import get_patch_spatial_anchors


def test_get_patch_spatial_anchors(tmp_path):
    # Create a synthetic GeoTIFF
    patch_path = tmp_path / "test_patch.tif"
    # EPSG:3035 origin (example)
    # x=4000000, y=3000000
    # res=10m
    transform = from_origin(4000000, 3000000, 10, 10)
    data = np.zeros((1, 256, 256), dtype=np.float32)

    with rasterio.open(
        patch_path,
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=1,
        dtype=np.float32,
        crs="EPSG:3035",
        transform=transform,
    ) as dst:
        dst.write(data)

    cx, cy = get_patch_spatial_anchors(patch_path)

    # Center pixel (256/2.0) = 128
    # Center coord = origin + (pixel_offset * resolution)
    # Note: rasterio transform.xy takes image offsets (rows/cols)
    # cy_pix = row_off = 128
    # cx_pix = col_off = 128
    expected_cx = 4000000 + (128 * 10) + 5
    expected_cy = 3000000 - (128 * 10) - 5  # y decreases from top

    assert cx == expected_cx
    assert cy == expected_cy


def test_extract_metadata():
    manifest = {
        "aoi_id": "AOI_01",
        "country": "FRA",
        "time": {"start": "2021-01-01", "end": "2021-12-31"},
        "mosaic": "mostRecent",
    }
    meta = extract_metadata(manifest)
    assert meta["aoi_id"] == "AOI_01"
    assert meta["country"] == "FRA"
    assert meta["acq_start"] == "2021-01-01"
    assert meta["acq_end"] == "2021-12-31"
    assert meta["mosaic_method"] == "mostRecent"


def test_extract_metadata_alternate_keys():
    manifest = {
        "aoi": "AOI_02",
        "country_code": "DEU",
        "time": "2022-01-01",
        "mosaicking_order": "leastCloudy",
    }
    meta = extract_metadata(manifest)
    assert meta["aoi_id"] == "AOI_02"
    assert meta["country"] == "DEU"
    assert meta["acq_start"] == "2022-01-01"
    assert meta["mosaic_method"] == "leastCloudy"
