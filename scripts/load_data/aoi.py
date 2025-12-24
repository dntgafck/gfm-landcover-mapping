import logging
from typing import cast

import geopandas as gpd

# Setup basic logging
logger = logging.getLogger(__name__)

DEFAULT_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/refs/tags/v5.1.2/geojson/ne_50m_admin_0_countries.geojson"


class AOILoader:
    """
    Load and filter AOIs from Natural Earth (or any GeoJSON), with reproducibility
    and EO-ML hygiene.
    """

    def __init__(self, geojson_source: str | None = None, force_wgs84: bool = True):
        """
        Initializes the AOILoader with a local path or URL.
        """
        self.geojson_source = geojson_source or DEFAULT_URL
        self.force_wgs84 = force_wgs84
        logger.info(f"Using GeoJSON source: {self.geojson_source}")

    def load(self) -> gpd.GeoDataFrame:
        """
        Loads the basic GeoJSON and ensures CRS/validity.
        """
        try:
            gdf = gpd.read_file(self.geojson_source)
        except Exception as e:
            raise RuntimeError(f"Failed to load AOI from {self.geojson_source}: {e}") from e

        # Ensure CRS
        if self.force_wgs84:
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            else:
                gdf = gdf.to_crs("EPSG:4326")

        # Drop empty geometries
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

        # Patch ISO_A3 if needed (e.g. France is -99 in some datasets)
        if "ISO_A3" in gdf.columns:
            gdf["ISO_A3"] = gdf.apply(self._resolve_iso_a3, axis=1)

        return gdf

    @staticmethod
    def _resolve_iso_a3(row) -> str | None:
        """
        Resolve ISO_A3 code, falling back to ADM0_A3 if ISO_A3 is invalid (-99).
        """
        iso = cast(str | None, row.get("ISO_A3"))
        # Check for invalid -99 (often as string "-99" or integer -99)
        if iso == "-99" or iso == -99:
            # Fallback to ADM0_A3 if available
            return cast(str | None, row.get("ADM0_A3", iso))
        return iso

    @staticmethod
    def _filter_by_values(series, values: list[str], substring_match: bool = False):
        """
        Filter series by a list of values (case-insensitive).
        If substring_match is True, checks if series value contains any of the values.
        Otherwise checks for equality.
        """
        # Normalize series to lowercase string
        series_lower = series.astype(str).str.lower()
        # Normalize values
        values_lower = [v.lower() for v in values]

        if substring_match:
            # Construct a regex pattern to match any value
            # abstract -> abstract|...
            import re

            pattern = "|".join(map(re.escape, values_lower))
            return series_lower.str.contains(pattern, na=False)
        else:
            return series_lower.isin(values_lower)

    def load_aoi(
        self,
        subregion: list[str] | None = None,
        name: list[str] | None = None,
        name_contains: bool = False,
        continent: list[str] | None = None,
        iso_a3: list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Loads and filters the countries layer using lists of allowed values.
        """
        gdf = self.load()

        if continent:
            if "CONTINENT" in gdf.columns:
                gdf = gdf[self._filter_by_values(gdf["CONTINENT"], continent)]
            else:
                logger.warning("Column CONTINENT not found; skipping continent filter.")

        if subregion:
            if "SUBREGION" in gdf.columns:
                gdf = gdf[self._filter_by_values(gdf["SUBREGION"], subregion)]
            else:
                logger.warning("Column SUBREGION not found; skipping subregion filter.")

        if iso_a3:
            if "ISO_A3" in gdf.columns:
                gdf = gdf[self._filter_by_values(gdf["ISO_A3"], iso_a3)]
            else:
                logger.warning("Column ISO_A3 not found; skipping iso_a3 filter.")

        if name:
            if "NAME" in gdf.columns:
                gdf = gdf[self._filter_by_values(gdf["NAME"], name, substring_match=name_contains)]
            else:
                logger.warning("Column NAME not found; skipping name filter.")

        if gdf.empty:
            logger.warning(
                f"Filter returned no data: continent={continent}, subregion={subregion}, "
                f"name={name}, iso_a3={iso_a3}"
            )

        return gdf

    @staticmethod
    def keep_largest_polygon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        For each feature, keep only the largest polygon part (useful to drop tiny islands/exclaves).
        """
        out = gdf.copy()

        def largest_part(geom):
            if geom is None or geom.is_empty:
                return geom
            if geom.geom_type == "Polygon":
                return geom
            if geom.geom_type == "MultiPolygon":
                parts = list(geom.geoms)
                parts.sort(key=lambda p: p.area, reverse=True)
                return parts[0]
            return geom

        out["geometry"] = out.geometry.apply(largest_part)
        out = out.reset_index(drop=True)
        return out

    def split_into_countries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Splits the geometry of the input GeoDataFrame by intersecting with country borders.
        Prioritizes attributes from the country dataset for standard fields.
        """
        countries = self.load()

        # Ensure CRSs match
        if gdf.crs != countries.crs:
            gdf = gdf.to_crs(countries.crs)

        # Identify overlapping columns (excluding geometry)
        overlap_cols = list(set(gdf.columns) & set(countries.columns) - {"geometry"})

        # Drop overlapping columns from input gdf to let countries' attributes take precedence
        if overlap_cols:
            gdf_pre = gdf.drop(columns=overlap_cols)
        else:
            gdf_pre = gdf

        # Spatial overlay (intersection)
        split_gdf = gpd.overlay(gdf_pre, countries, how="intersection", keep_geom_type=True)

        return split_gdf

    @staticmethod
    def to_aoi_schema(
        gdf: gpd.GeoDataFrame,
        aoi_id_prefix: str,
        extra_props: dict | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Convert rows into canonical aois.geojson schema.
        """
        extra_props = extra_props or {}

        # Create a small, clean AOI layer with consistent properties
        props = []
        for i, row in gdf.reset_index(drop=True).iterrows():
            # Try to get standard fields, fallback to generic
            name = row.get("NAME", row.get("ADMIN", f"AOI_{i}"))
            iso = row.get("ISO_A3", None)
            cont = row.get("CONTINENT", None)
            subr = row.get("SUBREGION", None)

            props.append(
                {
                    "aoi_id": f"{aoi_id_prefix}_{i:02d}",
                    "country": str(name),
                    "iso_a3": iso,
                    "continent": cont,
                    "subregion": subr,
                    **extra_props,
                }
            )

        if not props:
            return gdf

        out = gdf[["geometry"]].copy()
        for k in props[0].keys():
            out[k] = [p.get(k) for p in props]

        out = out.set_crs("EPSG:4326", allow_override=True)
        return out
