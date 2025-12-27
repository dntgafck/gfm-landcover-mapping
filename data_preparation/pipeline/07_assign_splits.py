from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from data_preparation.index.hash_split import get_stable_hash_float, validate_fractions
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def assign_splits_to_rank(ranks: np.ndarray, fractions_dict: dict) -> list:
    """Assign splits based on rank proportion [0, 1]."""
    # Sort splits by fraction for consistent binning
    items = sorted(fractions_dict.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in items]
    fractions = [x[1] for x in items]

    # Calculate cumulative thresholds
    thresholds = np.cumsum(fractions)

    results = []
    for r in ranks:
        assigned = False
        for i, t in enumerate(thresholds):
            if r <= t:
                results.append(names[i])
                assigned = True
                break
        if not assigned:
            # Fallback to last one due to precision
            results.append(names[-1])
    return results


@click.command()
@click.option(
    "--config", type=click.Path(exists=True), required=True, help="Path to split config YAML"
)
def main(config: str):
    setup_logging()
    with open(config) as f:
        cfg_full = yaml.safe_load(f)
        cfg = cfg_full.get("split", cfg_full)

    input_csv = Path(cfg.get("input_csv", "data/index/dataset_index.csv"))
    output_csv = Path(cfg.get("output_csv", "data/index/dataset_index_with_split.csv"))
    output_dir = Path(cfg.get("output_dir", "data/index"))
    aoi_path = Path("data/aoi.geojson")
    seed = cfg.get("seed", 42)
    split_configs = cfg.get("config", [])

    if not input_csv.exists():
        logger.error(f"Input index CSV missing at {input_csv}")
        return

    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} records for split assignment.")

    # 1. Load Country Name -> ISO_A3 mapping from AOI
    iso_map = {}
    if aoi_path.exists():
        try:
            aoi_gdf = gpd.read_file(aoi_path)
            if "country" in aoi_gdf.columns and "iso_a3" in aoi_gdf.columns:
                iso_map = dict(zip(aoi_gdf["country"], aoi_gdf["iso_a3"], strict=False))
                logger.info("Loaded ISO mapping for %d countries.", len(iso_map))
        except Exception as e:
            logger.warning(f"Failed to load AOI mapping: {e}")

    # 2. Build ISO_A3 -> Group Mapping and Validate Fractions
    iso_to_config = {}
    all_split_names = set()

    for sc in split_configs:
        countries = sc.get("countries", [])
        splits = sc.get("splits", {})

        try:
            validate_fractions(splits)
        except Exception as e:
            logger.error(f"Invalid fractions for group {sc['name']}: {e}")
            return

        for c in countries:
            iso = iso_map.get(c, c)
            iso_to_config[iso] = sc

        all_split_names.update(splits.keys())

    df["group_id"] = df["tile_id"]

    # 3. Stratified Assignment Logic: Per Country
    df["split"] = "excluded"
    unique_countries = df["country"].unique()

    for country in sorted(unique_countries):
        config_group = iso_to_config.get(country)
        if not config_group:
            logger.warning(f"Country {country} not found in split config. Excluding.")
            continue

        country_df = df[df["country"] == country]
        unique_groups = sorted(country_df["group_id"].unique())

        # Calculate hashes for each group in this country
        hashes = [get_stable_hash_float(str(gid), seed) for gid in unique_groups]

        # Determine ranks (normalized [0, 1])
        # We sort by hash and then use relative position to assign based on fractions
        sorted_indices = np.argsort(hashes)
        ranks = np.zeros(len(unique_groups))
        for i, idx in enumerate(sorted_indices):
            # Midpoint rank: (i + 0.5) / len
            ranks[idx] = (i + 0.5) / len(unique_groups)

        group_to_split = dict(
            zip(unique_groups, assign_splits_to_rank(ranks, config_group["splits"]), strict=False)
        )

        # Map back to main dataframe
        df.loc[df["country"] == country, "split"] = country_df["group_id"].map(group_to_split)

    # 4. Remove excluded
    if (df["split"] == "excluded").any():
        logger.info(
            f"Removing {(df['split'] == 'excluded').sum()} patches from countries not in config."
        )
        df = df[df["split"] != "excluded"].copy()

    # 5. Log distribution per country
    logger.info("Split Distribution per Country:")
    for country in sorted(df["country"].unique()):
        sub = df[df["country"] == country]
        counts = sub["split"].value_counts().to_dict()
        count_str = ", ".join([f"{k}: {v}" for k, v in counts.items()])
        logger.info(f"  {country:4}: {count_str}")

    # 6. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote canonical index with splits to {output_csv}")

    # 7. Dynamically generate subset files
    final_splits = sorted(df["split"].unique())
    for s in final_splits:
        sub = df[df["split"] == s]
        sub_csv = output_dir / f"{s}_index.csv"
        sub.to_csv(sub_csv, index=False)
        logger.info(f"Wrote {len(sub)} records to {sub_csv}")


if __name__ == "__main__":
    main()
