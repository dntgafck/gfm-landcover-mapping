from pathlib import Path

import geopandas as gpd
import hydra
from omegaconf import DictConfig

from data_preparation.load_data.loader import SentinelDataLoader
from utils.logging import setup_logging


@hydra.main(config_path="../../conf", config_name="params", version_base="1.2")
def main(cfg: DictConfig):
    setup_logging()

    download_params = cfg["download"]

    grid_path = Path("data/grid_selected.gpkg")
    if not grid_path.exists():
        raise FileNotFoundError(f"{grid_path} not found.")

    grid_gdf = gpd.read_file(grid_path)

    if grid_gdf.empty:
        print("Grid is empty.")
        Path(download_params["output_dir"]).mkdir(parents=True, exist_ok=True)
        return

    loader = SentinelDataLoader()

    # Group by ISO_A3
    if "iso_a3" not in grid_gdf.columns:
        # Fallback if no iso_a3
        groups = [("UNK", grid_gdf)]
    else:
        groups = grid_gdf.groupby("iso_a3")

    for iso_a3, group in groups:
        print(f"Processing {iso_a3} with {len(group)} tiles...")

        # Define output for this country
        # data/<ISO_A3>
        country_out = Path(download_params["output_dir"]) / iso_a3

        loader.download_batch(
            input_data=group,
            time_interval=(download_params["start_date"], download_params["end_date"]),
            resolution=cfg["preprocessing"]["resolution_m"],
            output_folder=str(country_out),
        )


if __name__ == "__main__":
    main()
