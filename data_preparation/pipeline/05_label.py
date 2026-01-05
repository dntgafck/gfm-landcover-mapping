import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from data_preparation.load_data.worldcover_labels import (
    WorldCoverLabeler,
    WorldCoverS3Config,
)
from utils.logging import setup_logging


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging()

    label_params = cfg["labels"]
    download_params = cfg["download"]

    # Check for WorldCover grid file
    wc_grid_path = "data/worldcover/v200/2021/esa_worldcover_grid.geojson"  # Hardcoded backup or need param

    # Initialize Labeler
    s3_cfg = WorldCoverS3Config(
        cache_dir=label_params.get("cache_dir", "data/worldcover/cache")
    )

    # Initialize labeler with lazy grid loading if possible, but __init__ reads it.
    if not os.path.exists(wc_grid_path):
        print(
            f"WARNING: WorldCover grid not found at {wc_grid_path}. Labeling might fail."
        )

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

    # Collect tasks
    tasks = []
    for root, _dirs, files in os.walk(input_root):
        for file in files:
            if file == "spectral.tif":
                ref_path = Path(root) / file
                rel_path = ref_path.relative_to(input_root)
                out_path = output_root / rel_path.parent / "labels.tif"

                if not out_path.exists():
                    tasks.append((ref_path, out_path))

    if not tasks:
        print("No new labels to generate.")
        return

    print(f"Generating {len(tasks)} labels...")
    count = 0
    for ref_path, out_path in tqdm(tasks, desc="Labeling"):
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            labeler.write_labels_for_image(str(ref_path), out_path=str(out_path))
            count += 1
        except Exception as e:
            print(f"\nCRITICAL: Failed to generate aligned label for {ref_path}: {e}")
            raise

    print(f"Generated {count} label files.")


if __name__ == "__main__":
    main()
