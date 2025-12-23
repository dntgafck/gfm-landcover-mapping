from pathlib import Path

import yaml

from scripts.load_data.aoi import AOILoader


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    aoi_params = params["aoi"]

    # Initialize loader
    loader = AOILoader(aoi_params.get("source"))

    # Load AOI
    # Construct args from params, filtering out 'source'
    query_params = {k: v for k, v in aoi_params.items() if k != "source"}
    # The AOILoader.load_aoi expects 'name_contains' as bool, but params might use it differently
    # Let's handle name_contains if it's not in params explicitly as a bool
    if "name" in query_params and query_params["name"]:
        # If name is provided, we might want name_contains to be true if intent is substring
        # But let's stick to strict if not specified?
        # The original CLI had --name-contains flag.
        # Let's add it to params if we want it, or default to False.
        pass

    raw_gdf = loader.load_aoi(**query_params)

    if raw_gdf.empty:
        raise ValueError("No AOI found with given parameters.")

    # Process
    split_gdf = loader.split_into_countries(raw_gdf)
    cleaned_gdf = loader.keep_largest_polygon(split_gdf)
    final_gdf = loader.to_aoi_schema(cleaned_gdf, aoi_id_prefix="AOI")

    # Save
    out_path = Path("data/aoi.geojson")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    final_gdf.to_file(out_path, driver="GeoJSON")
    print(f"Saved AOI to {out_path} with {len(final_gdf)} features.")


if __name__ == "__main__":
    main()
