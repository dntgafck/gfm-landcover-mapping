from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from data_preparation.load_data.aoi import AOILoader
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def load_params():
    with open("conf/load_data/params.yaml") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    aoi_params = params["aoi"]

    # Initialize loader
    loader = AOILoader(aoi_params.get("source"))

    out_path = Path("data/aoi.geojson")
    existing_gdf = None
    existing_countries = set()

    # Check for existing data
    if out_path.exists():
        try:
            existing_gdf = gpd.read_file(out_path)
            # Assuming 'country' column holds the name we filter by, or use iso_a3 if reliable
            if "country" in existing_gdf.columns:
                existing_countries.update(existing_gdf["country"].tolist())
            logger.info(f"Loaded existing AOI with {len(existing_gdf)} features.")
        except Exception as e:
            logger.error(f"Failed to load existing AOI: {e}. Starting fresh.")

    # Filter params based on what we already have
    # We primarily filter by 'name' (country list) as that seems to be the main driver
    requested_names = aoi_params.get("name", [])
    if requested_names:
        # Filter out names that are already present
        # Normalize for comparison? (e.g. strict string match for now as per schema)
        new_names = [n for n in requested_names if n not in existing_countries]

        if not new_names:
            logger.info("All requested countries are already present in aoi.geojson.")
            return

        # Update params to only fetch new names
        logger.info(f"Fetching new countries: {new_names}")
        # We need to construct a query that only targets these new names
        # Copy params and override 'name'
        query_params = {k: v for k, v in aoi_params.items() if k != "source"}
        query_params["name"] = new_names
    else:
        # If no specific names requested (e.g. by continent?), we might need different logic
        # For now, assuming 'name' is the primary filter as per user request
        query_params = {k: v for k, v in aoi_params.items() if k != "source"}

    # Load NEW AOIs
    try:
        raw_gdf = loader.load_aoi(**query_params)
    except Exception as e:
        # If filtering returns nothing (e.g. country name wrong), handle gracefully
        logger.error(f"Error loading AOIs: {e}")
        return

    if raw_gdf.empty:
        logger.info("No new AOIs found with given parameters.")
        return

    # Process new data
    split_gdf = loader.split_into_countries(raw_gdf)
    cleaned_gdf = loader.keep_largest_polygon(split_gdf)
    final_new_gdf = loader.to_aoi_schema(cleaned_gdf, aoi_id_prefix="AOI")

    # Combine with existing
    if existing_gdf is not None and not existing_gdf.empty:
        # Ensure CRS match
        if final_new_gdf.crs != existing_gdf.crs:
            final_new_gdf = final_new_gdf.to_crs(existing_gdf.crs)

        # We need to regenerate IDs or keep them?
        # If we append, IDs might clash if prefix is static "AOI_00".
        # Let's re-generate IDs for the whole set or append with offset?
        # Simple approach: Concat, then perhaps deduplicate if needed, or trust the filter.
        # Ideally, we should recalculate IDs to be unique.

        combined_gdf = pd.concat([existing_gdf, final_new_gdf], ignore_index=True)
        # Recalculate IDs to ensure uniqueness? Or keep original IDs?
        # User prompt didn't specify ID constraints, but "AOI_00" style implies sequential.
        # Let's update IDs.
        combined_gdf["aoi_id"] = [f"AOI_{i:02d}" for i in range(len(combined_gdf))]

        final_gdf = combined_gdf
    else:
        final_gdf = final_new_gdf

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_gdf.to_file(out_path, driver="GeoJSON")
    logger.info(f"Saved merged AOI to {out_path} with {len(final_gdf)} features.")


if __name__ == "__main__":
    setup_logging()
    main()
