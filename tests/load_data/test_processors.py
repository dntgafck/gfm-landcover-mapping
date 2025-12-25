import geopandas as gpd
from shapely.geometry import Polygon

from scripts.load_data.processors import GridPreprocessor


def test_generate_grid_basic():
    # Create a simple square AOI in WGS84
    # Roughly 10km x 10km square
    # (0.1 degree is roughly 11km at equator, but let's use a small box)
    aoi_geom = Polygon([(0, 0), (0, 0.1), (0.1, 0.1), (0.1, 0), (0, 0)])
    gdf = gpd.GeoDataFrame({"geometry": [aoi_geom]}, crs="EPSG:4326")

    # Generate 5km grid
    grid_size_km = 5
    grid_gdf = GridPreprocessor.generate_grid(gdf, grid_size_km)

    assert not grid_gdf.empty
    assert grid_gdf.crs == "EPSG:3035"
    # Centroid based within should keep some cells
    assert len(grid_gdf) > 0


def test_generate_grid_empty():
    gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    grid_gdf = GridPreprocessor.generate_grid(gdf, 10)
    assert grid_gdf.empty


def test_generate_grid_crs_handling():
    # Verify it handles input that is already in target CRS (or others)
    aoi_geom = Polygon([(0, 0), (0, 1000), (1000, 1000), (1000, 0), (0, 0)])
    gdf = gpd.GeoDataFrame({"geometry": [aoi_geom]}, crs="EPSG:3035")

    # 1km grid
    grid_gdf = GridPreprocessor.generate_grid(gdf, 1)
    assert grid_gdf.crs == "EPSG:3035"
    # Should be roughly 1 cell if centroid logic permits
    assert len(grid_gdf) >= 1
