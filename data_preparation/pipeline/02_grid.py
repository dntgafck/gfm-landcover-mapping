from pathlib import Path

import geopandas as gpd
import hydra
import pandas as pd
from omegaconf import DictConfig

from data_preparation.load_data.processors import GridPreprocessor
from utils.logging import setup_logging


@hydra.main(config_path="../../conf", config_name="params", version_base="1.2")
def main(cfg: DictConfig):
    setup_logging()

    grid_size = cfg["preprocessing"]["grid_size_km"]

    aoi_path = Path("data/aoi.geojson")
    if not aoi_path.exists():
        raise FileNotFoundError(f"{aoi_path} not found. Run existing stages first.")

    aoi_gdf = gpd.read_file(aoi_path)

    all_grids = []

    for _idx, row in aoi_gdf.iterrows():
        # Create a single-row GDF for the processor
        single_gdf = gpd.GeoDataFrame([row], crs=aoi_gdf.crs)

        # Generate grid
        grid = GridPreprocessor.generate_grid(single_gdf, grid_size)

        # Propagate metadata (like iso_a3) to the grid cells
        # The generate_grid returns new geometries. We should attach the parent attributes.
        # We can do a spatial join or just assign if we do it per feature.
        # Assigning is safer per feature loop.
        for col in row.index:
            if col != "geometry":
                grid[col] = row[col]

        all_grids.append(grid)

    if not all_grids:
        print("No grids generated.")
        return

    final_grid = gpd.GeoDataFrame(pd.concat(all_grids, ignore_index=True), crs=all_grids[0].crs)

    out_path = Path("data/grid.gpkg")
    final_grid.to_file(out_path, layer="grid", driver="GPKG")
    print(f"Saved grid to {out_path} with {len(final_grid)} cells.")


if __name__ == "__main__":
    main()
