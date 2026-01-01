from pathlib import Path

import geopandas as gpd
import hydra
from omegaconf import DictConfig, OmegaConf

from data_preparation.load_data.processors import TileSampler
from utils.logging import setup_logging


@hydra.main(config_path="../../conf", config_name="params", version_base="1.2")
def main(cfg: DictConfig):
    setup_logging()

    # OmegaConf -> dict for function
    sampling_params = OmegaConf.to_container(cfg.get("sampling", {}), resolve=True)

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
