from pathlib import Path

import geopandas as gpd
import yaml

from scripts.load_data.processors import TileSampler


def load_params():
    with open("conf/load_data/params.yaml") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    sampling_params = params.get("sampling", {})

    grid_path = Path("data/grid.gpkg")
    out_path = Path("data/grid_selected.gpkg")

    if not grid_path.exists():
        raise FileNotFoundError(f"{grid_path} not found.")

    gdf = gpd.read_file(grid_path)

    # Use the refactored sampler
    final_gdf = TileSampler.sample_tiles(gdf, sampling_params)

    # Save
    final_gdf.to_file(out_path, driver="GPKG")


if __name__ == "__main__":
    main()
