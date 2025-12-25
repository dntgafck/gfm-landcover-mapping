import unittest.mock as mock

from sentinelhub import CRS, BBox

from scripts.load_data.loader import SentinelDataLoader


@mock.patch("scripts.load_data.loader.get_config")
def test_compute_cache_key_determinism(mock_get_config):
    mock_get_config.return_value = mock.Mock()
    loader = SentinelDataLoader()
    bbox = BBox(bbox=(0, 0, 10, 10), crs=CRS(3035))
    time_interval = ("2023-01-01", "2023-01-02")
    resolution = 10
    collection_name = "SENTINEL2_L2A"
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            output: [{ id: 'default', bands: 3 }]
        };
    }
    """

    key1 = loader.compute_cache_key(bbox, time_interval, resolution, collection_name, evalscript)
    key2 = loader.compute_cache_key(bbox, time_interval, resolution, collection_name, evalscript)

    assert key1 == key2
    assert isinstance(key1, str)
    assert len(key1) == 64  # SHA256 length


@mock.patch("scripts.load_data.loader.get_config")
def test_compute_cache_key_sensitivity(mock_get_config):
    mock_get_config.return_value = mock.Mock()
    loader = SentinelDataLoader()
    bbox = BBox(bbox=(0, 0, 10, 10), crs=CRS(3035))
    time_interval = ("2023-01-01", "2023-01-02")
    resolution = 10
    collection_name = "SENTINEL2_L2A"
    evalscript = "script1"

    key_base = loader.compute_cache_key(
        bbox, time_interval, resolution, collection_name, evalscript
    )

    # Different resolution
    key_res = loader.compute_cache_key(bbox, time_interval, 20, collection_name, evalscript)
    assert key_base != key_res

    # Different date
    key_date = loader.compute_cache_key(
        bbox, ("2023-01-01", "2023-01-03"), resolution, collection_name, evalscript
    )
    assert key_base != key_date

    # Different evalscript
    key_script = loader.compute_cache_key(
        bbox, time_interval, resolution, collection_name, "script2"
    )
    assert key_base != key_script


@mock.patch("scripts.load_data.loader.get_config")
def test_compute_cache_key_quantization(mock_get_config):
    mock_get_config.return_value = mock.Mock()
    loader = SentinelDataLoader()
    # BBox coordinates with many decimals
    bbox1 = BBox(bbox=(0.1234567, 0.1234567, 10.1234567, 10.1234567), crs=CRS(3035))
    bbox2 = BBox(bbox=(0.1234568, 0.1234568, 10.1234568, 10.1234568), crs=CRS(3035))

    time_interval = ("2023-01-01", "2023-01-02")
    resolution = 10
    collection_name = "SENTINEL2_L2A"
    evalscript = "script"

    key1 = loader.compute_cache_key(bbox1, time_interval, resolution, collection_name, evalscript)
    # 0.1234567 rounds to 0.123457
    # Use something that rounds to 0.123458 (e.g. 0.1234578)
    bbox2 = BBox(bbox=(0.1234578, 0.1234578, 10.1234578, 10.1234578), crs=CRS(3035))
    key2 = loader.compute_cache_key(bbox2, time_interval, resolution, collection_name, evalscript)
    assert key1 != key2

    # Verify rounding to 6 decimals: 0.1234561 should round to same as 0.1234562
    bbox_a = BBox(bbox=(0.1234561, 0, 1, 1), crs=CRS(3035))
    bbox_b = BBox(bbox=(0.1234562, 0, 1, 1), crs=CRS(3035))
    key_a = loader.compute_cache_key(bbox_a, time_interval, resolution, collection_name, evalscript)
    key_b = loader.compute_cache_key(bbox_b, time_interval, resolution, collection_name, evalscript)
    assert key_a == key_b
