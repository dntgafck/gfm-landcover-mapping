import logging
import os
from pathlib import Path

import click
import geopandas as gpd
import yaml

from scripts.load_data.aoi import AOILoader

# Use absolute imports assuming the script is run as a module: python -m scripts.load_data.run
# Or relative if we rely on sys.path.
# Given the user context, absolute imports from 'scripts.load_data' seem standard.
from scripts.load_data.loader import SentinelDataLoader
from scripts.load_data.processors import GridPreprocessor
from scripts.load_data.worldcover_labels import WorldCoverLabeler, WorldCoverS3Config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_params():
    params_path = Path("conf/load_data/params.yaml")
    if params_path.exists():
        with open(params_path) as f:
            return yaml.safe_load(f)
    # Check root params.yaml as fallback
    if Path("params.yaml").exists():
        with open("params.yaml") as f:
            return yaml.safe_load(f)
    return {}


@click.command()
# AOI Selection Options
@click.option("--source", help="Path or URL to the input GeoJSON file (AOI source).")
@click.option(
    "--iso-a3",
    required=False,  # Changed to False as it can be in params
    multiple=True,
    help="Filter AOI by ISO_A3 (case-insensitive).",
)
@click.option("--grid-size", type=int, help="Split AOI into grid cells of this size in km.")
# Download Options
@click.option("--input-file", help="Path to input GeoJSON/file with PRE-GENERATED grid/AOIs.")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--resolution", type=int, help="Resolution in meters")
@click.option(
    "--output-folder", required=True, help="Output folder for downloaded data (REQUIRED)."
)
@click.option("--skip-label", is_flag=True, help="Skip the labeling step.")
def main(
    source,
    iso_a3,
    grid_size,
    input_file,
    start_date,
    end_date,
    resolution,
    output_folder,
    skip_label,
):
    """
    Unified CLI tool to load AOIs, generate grids, download Sentinel-2 data, and generate labels.
    Defaults are loaded from conf/load_data/params.yaml.
    """
    params = load_params()
    aoi_params = params.get("aoi", {})
    prep_params = params.get("preprocessing", {})
    dl_params = params.get("download", {})
    lbl_params = params.get("labels", {})

    # Resolve defaults
    source = source or aoi_params.get("source")

    iso_a3 = list(iso_a3) if iso_a3 else aoi_params.get("iso_a3")

    grid_size = grid_size if grid_size is not None else prep_params.get("grid_size_km")

    start_date = start_date or dl_params.get("start_date")
    end_date = end_date or dl_params.get("end_date")

    if resolution is None:
        resolution = prep_params.get("resolution_m")
    # output_folder is required, so no default fallback needed here

    if resolution is None:
        resolution = 10

    loader = None  # Lazy initialization
    labeler = None

    # Initialize Labeler if not skipping
    if not skip_label:
        wc_grid_path = "data/worldcover/v200/2021/esa_worldcover_grid.geojson"  # Default
        cache_dir = lbl_params.get("cache_dir", "data/worldcover/cache")
        try:
            s3_cfg = WorldCoverS3Config(cache_dir=cache_dir)
            # Warn if grid doesn't exist but let Labeler handle or crash if it's critical
            if not os.path.exists(wc_grid_path):
                logger.warning(
                    f"WorldCover grid not found at {wc_grid_path}. "
                    "Proceeding, but labeling may fail."
                )

            labeler = WorldCoverLabeler(wc_grid_path, s3_cfg)
        except Exception as e:
            logger.error(f"Failed to initialize labeler: {e}")
            if not skip_label:
                logger.warning("Labeling will be skipped due to initialization failure.")
                skip_label = True

    # 1. Determine Input Data
    # Priority: input-file > AOI filtering
    target_gdf = None

    try:
        if input_file:
            logger.info(f"Using input file: {input_file}")
            target_gdf = gpd.read_file(input_file)

            if not target_gdf.empty:
                if loader is None:
                    loader = SentinelDataLoader()

                loader.download_batch(
                    input_data=target_gdf,
                    time_interval=(start_date, end_date),
                    resolution=resolution,
                    output_folder=output_folder,
                )

                # For input-file mode, we need to know the structure to label.

                if not skip_label and labeler:
                    logger.info("Starting labeling for input file batch...")
                    # Iterate over subdirectories in output_folder
                    for item in os.listdir(output_folder):
                        item_path = os.path.join(output_folder, item)

                        if os.path.isdir(item_path):
                            response_tiff_path = os.path.join(item_path, "response.tiff")
                            if os.path.exists(response_tiff_path):
                                label_path = os.path.join(item_path, "labels.tiff")

                                if os.path.exists(label_path):
                                    continue

                                try:
                                    labeler.write_labels_for_response_tiff(
                                        response_tiff_path, out_path=label_path
                                    )
                                    logger.info(f"Generated labels for {item}")
                                except Exception as e:
                                    logger.error(f"Failed to label {item}: {e}")

        elif iso_a3:
            logger.info("Using AOI Loader parameters...")
            aoi_loader = AOILoader(source)
            raw_gdf = aoi_loader.load_aoi(
                iso_a3=iso_a3,
            )
            logger.info(f"Loaded {len(raw_gdf)} features from source.")

            if raw_gdf.empty:
                logger.warning("No features found.")
                return

            # Apply processing pipeline: Split -> Keep Largest -> Schema
            logger.info("Splitting AOI into countries...")
            split_gdf = aoi_loader.split_into_countries(raw_gdf)

            logger.info("Keeping largest polygons...")
            cleaned_gdf = aoi_loader.keep_largest_polygon(split_gdf)

            # Convert to standard schema
            final_gdf = aoi_loader.to_aoi_schema(cleaned_gdf, aoi_id_prefix="AOI")

            # Process each country/feature individually
            for idx, row in final_gdf.iterrows():
                country_iso = row.get("iso_a3")
                if not country_iso:
                    country_iso = f"UNK_{idx}"

                # Create country folder
                country_dir = os.path.join(output_folder, country_iso)
                os.makedirs(country_dir, exist_ok=True)

                # Save AOI
                country_aoi_path = os.path.join(country_dir, "aoi.geojson")

                # Create a single-row GDF for this country
                country_gdf = gpd.GeoDataFrame([row], crs=final_gdf.crs)
                country_gdf.to_file(country_aoi_path, driver="GeoJSON")
                logger.info(f"Saved AOI for {country_iso} to {country_aoi_path}")

                # Generate Grid if requested (per country)
                download_gdf = country_gdf
                if grid_size:
                    logger.info(f"Generating {grid_size}km grid for {country_iso}...")
                    download_gdf = GridPreprocessor.generate_grid(country_gdf, grid_size)

                # Download Data
                # Flattened structure: data/<ISO_A3>/<cache_key>/
                if not download_gdf.empty:
                    if loader is None:
                        loader = SentinelDataLoader()

                    loader.download_batch(
                        input_data=download_gdf,
                        time_interval=(start_date, end_date),
                        resolution=resolution,
                        output_folder=country_dir,
                    )

                    # Labeling Step
                    if not skip_label and labeler:
                        logger.info(f"Generating labels for {country_iso}...")

                        # Iterate over subdirectories in country_dir
                        # Each subdirectory is potentially a cache_key folder
                        for item in os.listdir(country_dir):
                            item_path = os.path.join(country_dir, item)

                            if os.path.isdir(item_path):
                                # Check if it's a valid data directory (has response.tiff)
                                response_tiff_path = os.path.join(item_path, "response.tiff")
                                if os.path.exists(response_tiff_path):
                                    # Label output path: co-located with response.tiff
                                    label_path = os.path.join(item_path, "labels.tiff")

                                    if os.path.exists(label_path):
                                        continue

                                    try:
                                        labeler.write_labels_for_response_tiff(
                                            response_tiff_path, out_path=label_path
                                        )
                                        logger.info(f"Generated labels for {item}")
                                    except Exception as e:
                                        logger.error(f"Failed to label {item}: {e}")

                else:
                    logger.warning(f"No valid geometry/grid for {country_iso} to download.")

        else:
            click.echo(
                "Please provide either --input-file or AOI parameters "
                "(e.g. in params.yaml or --iso-a3)."
            )
            click.echo("Run --help for details.")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


if __name__ == "__main__":
    main()
