import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from utils.logging import get_logger
from utils.sampling import StratifiedSampler

logger = get_logger(__name__)


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

        target_crs = "EPSG:3035"  # Europe LAEA, meter

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
        mask = grid_gdf.centroid.within(gdf_metric.unary_union)
        grid_gdf = grid_gdf[mask].reset_index(drop=True)

        logger.info(f"Generated {len(grid_gdf)} grid cells of size {grid_size_km}km.")
        return grid_gdf


class TileSampler:
    """
    Handles deterministic sampling of grid tiles.
    """

    @staticmethod
    def sample_tiles(gdf, sampling_params):
        """
        Executes the sampling process on a GeoDataFrame.
        """
        if not sampling_params.get("enabled", True):
            logger.info("Sampling disabled using param enabled=False.")
            return gdf

        total_tiles = sampling_params.get("total_tiles", 2500)
        per_country_min = sampling_params.get("per_country_min", 50)
        weight_power = sampling_params.get("weight_power", 1.0)
        seed = sampling_params.get("seed", 42)
        country_key = sampling_params.get("country_key", "iso_a3")

        # Handle missing key by falling back to auto-detection or global
        if country_key not in gdf.columns:
            if "iso_a3" in gdf.columns:
                country_key = "iso_a3"
            elif len(gdf) > 0:
                logger.warning(f"{country_key} not in grid. Treating as single group for sampling.")
                # We won't modify gdf in place to add a column, just pass a dummy column name?
                # The utility handles missing column by adding temporary one if we pass dataframe.
                # But here we are passing GDF.
                # Let's trust the utility's fallback if we pass the dataframe.

        # Determine if we should pass the GDF directly or handle the fallback here.
        # The utility `sample_stratified` handles the groupby.
        # We need to import it.
        # The utility `sample_stratified` handles the groupby.
        # We need to import it.

        sampler = StratifiedSampler(
            stratify_by=country_key,
            min_per_strata=per_country_min,
            weight_power=weight_power,
            seed=seed,
        )
        final_gdf = sampler.sample(gdf, total_tiles)

        if isinstance(final_gdf, pd.DataFrame) and not isinstance(final_gdf, gpd.GeoDataFrame):
            # Ensure it stays GeoDataFrame if pandas conversion happened (unlikely but safe)
            final_gdf = gpd.GeoDataFrame(final_gdf, crs=gdf.crs)

        logger.info(f"Selected {len(final_gdf)} tiles.")
        return final_gdf
