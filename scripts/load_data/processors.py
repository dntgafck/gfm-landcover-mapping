import logging

import geopandas as gpd
import numpy as np
from shapely.geometry import box

logger = logging.getLogger(__name__)


class GridPreprocessor:
    """
    Handles preprocessing steps for AOIs, such as grid generation.
    """

    @staticmethod
    def generate_grid(gdf: gpd.GeoDataFrame, grid_size_km: int) -> gpd.GeoDataFrame:
        """
        Splits the given GeoDataFrame into a grid of squares of grid_size_km.
        Input gdf is expected to be in WGS84 (EPSG:4326).
        """
        if gdf.empty:
            logger.warning("Empty GeoDataFrame provided for grid generation.")
            return gdf

        # Reproject to a metric CRS to calculate grid in meters
        # Try to estimate a suitable UTM CRS, fallback to 3857 if fails or old geopandas
        try:
            target_crs = gdf.estimate_utm_crs()
        except Exception:
            # Fallback for older geopandas or global coverage issues
            target_crs = "EPSG:3857"

        gdf_metric = gdf.to_crs(target_crs)

        # Get total bounds
        minx, miny, maxx, maxy = gdf_metric.total_bounds

        # Grid size in meters
        step = grid_size_km * 1000

        # Create grid cells
        x_ranges = np.arange(minx, maxx, step)
        y_ranges = np.arange(miny, maxy, step)

        grid_cells = []
        for x in x_ranges:
            for y in y_ranges:
                cell = box(x, y, x + step, y + step)
                grid_cells.append(cell)

        # Create GeoDataFrame from grid cells
        grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=target_crs)

        # Keep only cells whose centroid is within the AOI
        # This avoids sliver tiles that barely touch the AOI
        # Keep only grid cells whose centroid lies inside the AOI
        mask = grid_gdf.centroid.within(gdf_metric.unary_union)
        grid_gdf = grid_gdf[mask].reset_index(drop=True)

        # Reproject back to WGS84
        grid_gdf_wgs84 = grid_gdf.to_crs("EPSG:4326")

        logger.info(f"Generated {len(grid_gdf_wgs84)} grid cells of size {grid_size_km}km.")
        return grid_gdf_wgs84
