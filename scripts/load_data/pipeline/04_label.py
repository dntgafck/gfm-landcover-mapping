import os
from pathlib import Path

import yaml

from scripts.load_data.worldcover_labels import WorldCoverLabeler, WorldCoverS3Config


def load_params():
    with open("conf/load_data/params.yaml") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    label_params = params["labels"]
    download_params = params["download"]

    # Check for WorldCover grid file
    # If not configured, use default path relative to project root?
    # The class default is "data/worldcover/v200/2021/esa_worldcover_grid.geojson"
    # We should ensure this file exists or is downloaded.
    # For now, let's assume the user has it or we point to it.
    # We can pass it if we have it in params.

    wc_grid_path = (
        "data/worldcover/v200/2021/esa_worldcover_grid.geojson"  # Hardcoded backup or need param
    )

    # Initialize Labeler
    s3_cfg = WorldCoverS3Config(cache_dir=label_params.get("cache_dir", "data/worldcover/cache"))

    # Initialize labeler with lazy grid loading if possible, but __init__ reads it.
    if not os.path.exists(wc_grid_path):
        print(f"WARNING: WorldCover grid not found at {wc_grid_path}. Labeling might fail.")
        # Try to find it if possible or rely on error

    try:
        labeler = WorldCoverLabeler(wc_grid_path, s3_cfg)
    except Exception as e:
        print(f"Failed to initialize WorldCoverLabeler: {e}")
        return

    # Walk through data/imagery
    input_root = Path(download_params["output_dir"])
    output_root = Path(label_params["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        print(f"Input directory {input_root} does not exist.")
        return

    count = 0
    # Walk: data/imagery/<ISO_A3>/sh/tile_X/response.tiff
    for root, _dirs, files in os.walk(input_root):
        for file in files:
            if file == "response.tiff":
                resp_path = Path(root) / file

                # Determine relative path to mirror structure
                rel_path = resp_path.relative_to(input_root)
                # rel_path is <ISO_A3>/sh/tile_X/response.tiff

                # Construct output path
                # data/labels/<ISO_A3>/sh/tile_X/labels.tiff
                out_path = output_root / rel_path.parent / "labels.tiff"

                if out_path.exists():
                    continue

                print(f"Generating labels for {rel_path}...")
                try:
                    labeler.write_labels_for_response_tiff(str(resp_path), out_path=str(out_path))
                    count += 1
                except Exception as e:
                    print(f"Failed to generate label for {rel_path}: {e}")

    print(f"Generated {count} label files.")


if __name__ == "__main__":
    main()
