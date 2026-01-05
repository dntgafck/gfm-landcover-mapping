import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from data_preparation.load_data.worldcover_labels import WorldCoverLabeler


@pytest.fixture
def mock_grid_geojson(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    grid_path = tmp_path / "grid.geojson"
    # Create a dummy grid tile in 4326
    # EPSG:3035 0,0 is roughly in Western Europe (not exactly, but for mock doesn't matter)
    # Let's say we have a tile around 0,0 in 4326
    gdf = gpd.GeoDataFrame(
        {"ll_tile": ["S01E000"], "geometry": [box(-1, -1, 1, 1)]}, crs="EPSG:4326"
    )
    gdf.to_file(grid_path, driver="GeoJSON")
    return str(grid_path)


@mock.patch("data_preparation.load_data.worldcover_labels.boto3.client")
def test_labeling_alignment_assertions(mock_s3, mock_grid_geojson, tmp_path):
    # Setup mock S3
    mock_s3.return_value = mock.Mock()

    # Setup mock reference image (EPSG:3035)
    ref_path = tmp_path / "spectral.tif"
    ref_crs = "EPSG:3035"
    ref_transform = from_origin(4000000, 3000000, 10, 10)
    ref_w, ref_h = 100, 100

    with rasterio.open(
        ref_path,
        "w",
        driver="GTiff",
        height=ref_h,
        width=ref_w,
        count=4,
        dtype="float32",
        crs=ref_crs,
        transform=ref_transform,
    ) as ds:
        ds.write(np.zeros((4, ref_h, ref_w), dtype="float32"))

    labeler = WorldCoverLabeler(mock_grid_geojson)

    # Mock tile downloading and mosaicking
    # We'll mock the internal methods to avoid actual processing
    with mock.patch.object(labeler, "tiles_for_bounds", return_value=["S01E000"]):
        with mock.patch.object(
            labeler, "_download_tile_if_missing", return_value=Path("mock.tif")
        ):
            with mock.patch.object(
                labeler,
                "_mosaic_tiles",
                return_value=(
                    np.zeros((10, 10)),
                    from_origin(0, 0, 0.1, 0.1),
                    "EPSG:4326",
                    0,
                ),
            ):
                with mock.patch(
                    "data_preparation.load_data.worldcover_labels.reproject"
                ) as mock_reproj:
                    out_path = labeler.write_labels_for_image(ref_path)

                    # Verify reproject was called with correct target params
                    args, kwargs = mock_reproj.call_args
                    assert kwargs["dst_crs"] == ref_crs
                    assert kwargs["dst_transform"] == ref_transform
                    assert kwargs["dst_width"] == ref_w
                    assert kwargs["dst_height"] == ref_h
                    assert kwargs["resampling"].name == "nearest"

                    # Verify output file exists and matches reference
                    assert out_path.name == "labels.tif"
                    with rasterio.open(out_path) as labels:
                        assert labels.crs == ref_crs
                        assert labels.transform == ref_transform
                        assert labels.width == ref_w
                        assert labels.height == ref_h
                        assert labels.count == 1
                        assert labels.dtypes[0] == "uint8"


def test_tiles_for_bounds_transformation(mock_grid_geojson):
    labeler = WorldCoverLabeler(mock_grid_geojson)
    # Bounds in 3035 that should project to around 0,0 in 4326
    # (very rough, just testing that transform_bounds is called)
    bounds = (4000000, 3000000, 4001000, 3001000)
    crs = "EPSG:3035"

    with mock.patch(
        "rasterio.warp.transform_bounds", return_value=(-0.5, -0.5, 0.5, 0.5)
    ) as mock_trans:
        tiles = labeler.tiles_for_bounds(bounds, crs)
        assert mock_trans.called
        assert tiles == ["S01E000"]
