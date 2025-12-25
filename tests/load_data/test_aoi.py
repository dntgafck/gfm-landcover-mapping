import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

from scripts.load_data.aoi import AOILoader


def test_resolve_iso_a3():
    loader = AOILoader()

    # Valid ISO
    row = {"ISO_A3": "ITA", "ADM0_A3": "ITA"}
    assert loader._resolve_iso_a3(row) == "ITA"

    # Invalid ISO (-99) with fallback
    row = {"ISO_A3": "-99", "ADM0_A3": "FRA"}
    assert loader._resolve_iso_a3(row) == "FRA"

    # Invalid ISO (-99) integer
    row = {"ISO_A3": -99, "ADM0_A3": "ESP"}
    assert loader._resolve_iso_a3(row) == "ESP"


def test_keep_largest_polygon():
    # MultiPolygon with one large and one small part
    p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])  # Area 100
    p2 = Polygon([(20, 20), (21, 20), (21, 21), (20, 21), (20, 20)])  # Area 1
    mp = MultiPolygon([p1, p2])

    gdf = gpd.GeoDataFrame({"geometry": [mp]}, crs="EPSG:4326")
    loader = AOILoader()

    filtered_gdf = loader.keep_largest_polygon(gdf)
    assert len(filtered_gdf) == 1
    assert filtered_gdf.geometry.iloc[0].geom_type == "Polygon"
    assert filtered_gdf.geometry.iloc[0].area == 100


def test_to_aoi_schema():
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [p],
            "NAME": ["Italy"],
            "ISO_A3": ["ITA"],
            "CONTINENT": ["Europe"],
            "SUBREGION": ["Southern Europe"],
        },
        crs="EPSG:4326",
    )

    loader = AOILoader()
    aoi_gdf = loader.to_aoi_schema(gdf, "TEST")

    assert "aoi_id" in aoi_gdf.columns
    assert aoi_gdf["aoi_id"].iloc[0] == "TEST_00"
    assert aoi_gdf["country"].iloc[0] == "Italy"
    assert aoi_gdf["iso_a3"].iloc[0] == "ITA"
    assert aoi_gdf["continent"].iloc[0] == "Europe"
