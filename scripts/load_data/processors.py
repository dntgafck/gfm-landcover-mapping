import logging

import geopandas as gpd
import numpy as np
import pandas as pd
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
    def allocate_budget(groups, total_budget, weight_power, min_per_country):
        """
        Allocate budget per country based on the number of tiles.
        Returns a dict {group_key: n_samples}.
        """

        # 1. Calculate weights
        stats = []
        for key, df in groups:
            n_c = len(df)
            weight = n_c**weight_power
            stats.append({"key": key, "n_c": n_c, "weight": weight})

        stats_df = pd.DataFrame(stats)
        if stats_df.empty:
            return {}
        if total_budget == 0:
            return dict(zip(stats_df["key"], [0] * len(stats_df), strict=False))
        total_weight = stats_df["weight"].sum()

        # 2. Initial allocation
        if total_weight == 0:
            stats_df["allocated"] = 0
        else:
            stats_df["allocated"] = np.floor(
                total_budget * stats_df["weight"] / total_weight
            ).astype(int)

        # 3. Enforce min/max constraints
        def apply_constraints(row):
            target = row["allocated"]
            floor_val = min(row["n_c"], min_per_country)
            target = max(target, floor_val)
            target = min(target, row["n_c"])
            return int(target)

        stats_df["allocated"] = stats_df.apply(apply_constraints, axis=1)

        # 4. Adjust to match total_budget exactly
        current_total = stats_df["allocated"].sum()
        diff = total_budget - current_total

        if diff != 0:
            stats_df = stats_df.sort_values("key")

            if diff > 0:
                # Need to ADD samples
                while diff > 0:
                    mask = stats_df["allocated"] < stats_df["n_c"]
                    if not mask.any():
                        break

                    stats_df["ideal"] = total_budget * stats_df["weight"] / total_weight
                    stats_df["residual"] = stats_df["ideal"] - stats_df["allocated"]

                    best_idx = (
                        stats_df.loc[mask]
                        .sort_values(by=["residual", "key"], ascending=[False, True])
                        .index[0]
                    )

                    stats_df.at[best_idx, "allocated"] += 1
                    diff -= 1

            elif diff < 0:
                # Need to REMOVE samples
                while diff < 0:

                    def get_floor(row):
                        return min(row["n_c"], min_per_country)

                    stats_df["floor"] = stats_df.apply(get_floor, axis=1)
                    candidates = stats_df[stats_df["allocated"] > stats_df["floor"]]

                    if candidates.empty:
                        logger.warning(
                            "Could not reduce count to total_tiles due to minimum constraints."
                        )
                        break

                    stats_df["ideal"] = total_budget * stats_df["weight"] / total_weight
                    stats_df["residual"] = stats_df["ideal"] - stats_df["allocated"]

                    best_idx = (
                        stats_df.loc[candidates.index]
                        .sort_values(by=["residual", "key"], ascending=[True, True])
                        .index[0]
                    )

                    stats_df.at[best_idx, "allocated"] -= 1
                    diff += 1

        return dict(zip(stats_df["key"], stats_df["allocated"], strict=False))

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

        if country_key not in gdf.columns:
            if "iso_a3" in gdf.columns:
                country_key = "iso_a3"
            elif len(gdf) > 0:
                logger.warning(f"{country_key} not in grid. Treating as single group.")
                gdf["_global"] = "global"
                country_key = "_global"

        # Group and Allocate
        groups = list(gdf.groupby(country_key))
        allocation = TileSampler.allocate_budget(groups, total_tiles, weight_power, per_country_min)

        logger.info(f"Sampling Allocation (Total requested: {total_tiles}):")
        for k, v in allocation.items():
            logger.info(f"  {k}: {v}")

        # Sample
        rng = np.random.default_rng(seed)
        groups_sorted = sorted(groups, key=lambda x: x[0])

        sampled_list = []
        for key, group in groups_sorted:
            n_samples = allocation.get(key, 0)
            if n_samples >= len(group):
                selected = group.copy()
                selected["selected"] = True
                sampled_list.append(selected)
            else:
                indices = group.index.to_numpy()
                chosen_indices = rng.choice(indices, size=n_samples, replace=False)
                chosen_indices.sort()

                selected = group.loc[chosen_indices].copy()
                selected["selected"] = True
                sampled_list.append(selected)

        if not sampled_list:
            return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)

        final_gdf = gpd.GeoDataFrame(pd.concat(sampled_list, ignore_index=True), crs=gdf.crs)
        logger.info(f"Selected {len(final_gdf)} tiles.")
        return final_gdf
