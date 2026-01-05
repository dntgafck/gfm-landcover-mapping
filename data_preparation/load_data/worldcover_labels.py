# worldcover_labels.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import boto3
import geopandas as gpd
import numpy as np
import rasterio
from botocore import UNSIGNED
from botocore.config import Config
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from shapely.geometry import box


@dataclass(frozen=True)
class WorldCoverS3Config:
    bucket: str = "esa-worldcover"
    region: str = "eu-central-1"
    version: str = "v200"
    year: str = "2021"
    layer: str = "Map"  # "Map" or "InputQuality"
    cache_dir: str = "data/worldcover/cache"


class WorldCoverLabeler:
    """
    JIT WorldCover -> labels.tiff aligned to Sentinel Hub response.tiff.

    - Uses WorldCover grid GeoJSON to map bbox -> tile ids
    - Downloads required tiles from public S3 with boto3 (unsigned)
    - Mosaics tiles over bbox, then warps to reference grid (nearest)
    """

    def __init__(
        self,
        worldcover_grid_geojson: str,
        s3_cfg: WorldCoverS3Config | None = None,
        tile_id_col: str | None = None,
    ):
        self.s3_cfg = s3_cfg or WorldCoverS3Config()
        self.cache_dir = Path(self.s3_cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.grid = gpd.read_file(worldcover_grid_geojson)
        if self.grid.crs is None:
            self.grid = self.grid.set_crs("EPSG:4326")
        else:
            self.grid = self.grid.to_crs("EPSG:4326")

        self.tile_id_col = tile_id_col or self._detect_tile_id_col(self.grid)

        # unsigned public S3 client
        self.s3 = boto3.client(
            "s3",
            region_name=self.s3_cfg.region,
            config=Config(signature_version=UNSIGNED),
        )

    @staticmethod
    def _detect_tile_id_col(gdf: gpd.GeoDataFrame) -> str:
        # Common for ESA WorldCover grid: "ll_tile"
        preferred = ["ll_tile", "LL_TILE", "tile_id", "TILE_ID", "tile", "name", "Name"]
        for c in preferred:
            if c in gdf.columns:
                return c
        for c in gdf.columns:
            if "tile" in c.lower():
                return c
        raise ValueError(
            f"Could not detect tile id column. Available columns: {list(gdf.columns)}"
        )

    def _tile_filename(self, tile_id: str) -> str:
        # Example: ESA_WorldCover_10m_2021_v200_S48E165_Map.tif
        return (
            f"ESA_WorldCover_10m_{self.s3_cfg.year}_{self.s3_cfg.version}_{tile_id}_"
            f"{self.s3_cfg.layer}.tif"
        )

    def _tile_s3_key(self, tile_id: str) -> str:
        # Map tiles live at: v200/2021/map/...
        # InputQuality tiles (if needed) are usually at: v200/2021/input_quality/...
        subdir = "map" if self.s3_cfg.layer == "Map" else "input_quality"
        return f"{self.s3_cfg.version}/{self.s3_cfg.year}/{subdir}/{self._tile_filename(tile_id)}"

    def _tile_cache_path(self, tile_id: str) -> Path:
        return self.cache_dir / self._tile_filename(tile_id)

    def tiles_for_bounds(
        self,
        bounds: tuple[float, float, float, float],
        crs: str | rasterio.crs.CRS,
    ) -> list[str]:
        """
        Find tiles intersecting bounds in any CRS.
        """
        if str(crs).upper() != "EPSG:4326":
            from rasterio.warp import transform_bounds

            # Be careful with CRS84 axis order: treat as (lon, lat)
            bbox_ll = transform_bounds(crs, "EPSG:4326", *bounds)
        else:
            bbox_ll = bounds

        geom = box(*bbox_ll)
        hits = self.grid[self.grid.intersects(geom)]
        tiles = sorted(hits[self.tile_id_col].astype(str).unique().tolist())
        return tiles

    def _download_tile_if_missing(self, tile_id: str) -> Path:
        out = self._tile_cache_path(tile_id)
        if out.exists() and out.stat().st_size > 0:
            return out

        key = self._tile_s3_key(tile_id)
        tmp = out.with_suffix(out.suffix + ".part")

        # stream to disk
        with open(tmp, "wb") as f:
            self.s3.download_fileobj(self.s3_cfg.bucket, key, f)

        tmp.replace(out)
        return out

    @staticmethod
    def _mosaic_tiles(
        tile_paths: list[Path],
    ):
        """
        Mosaic tiles in their native CRS.
        Returns (array_2d, transform, crs, nodata).
        """
        srcs = [rasterio.open(str(p)) for p in tile_paths]
        try:
            mosaic, transform = merge(srcs)
            crs = srcs[0].crs
            nodata = srcs[0].nodata
            arr = mosaic[0]  # first band
            return arr, transform, crs, nodata
        finally:
            for s in srcs:
                s.close()

    def write_labels_for_image(
        self,
        image_path: str | Path,
        out_path: str | Path | None = None,
        dst_nodata: int = 0,
        compress: str = "deflate",
    ) -> Path:
        """
        Create labels.tif aligned to a reference image (e.g. spectral.tif).
        Follows strict pixel alignment requirements.
        """
        image_path = Path(image_path)
        with rasterio.open(str(image_path)) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_w, ref_h = ref.width, ref.height
            ref_bounds = ref.bounds

        if ref_crs is None:
            raise ValueError(f"Reference raster has no CRS: {image_path}")

        # Find intersecting tiles
        tile_ids = self.tiles_for_bounds(ref_bounds, ref_crs)
        if not tile_ids:
            # If no tiles found, we return an empty (nodata) raster
            # but usually this indicates an AOI outside WorldCover
            import warnings

            warnings.warn(
                f"No WorldCover tiles found for image {image_path}. Outputting nodata.",
                stacklevel=2,
            )
            tile_paths = []
        else:
            # Download required tiles
            tile_paths = [self._download_tile_if_missing(tid) for tid in tile_ids]

        # Mosaic and Warp
        dst = np.full((ref_h, ref_w), dst_nodata, dtype=np.uint8)

        if tile_paths:
            src_arr, src_transform, src_crs, src_nodata = self._mosaic_tiles(tile_paths)

            reproject(
                source=src_arr,
                destination=dst,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                dst_width=ref_w,
                dst_height=ref_h,
                resampling=Resampling.nearest,
                src_nodata=src_nodata,
                dst_nodata=dst_nodata,
            )

        # Output path
        if out_path is None:
            out_path_obj = image_path.parent / "labels.tif"
        else:
            out_path_obj = Path(out_path)

        out_path_obj.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            "driver": "GTiff",
            "height": ref_h,
            "width": ref_w,
            "count": 1,
            "dtype": "uint8",
            "crs": ref_crs,
            "transform": ref_transform,
            "nodata": dst_nodata,
            "compress": compress,
            "predictor": 2,
            "tiled": True,
        }

        with rasterio.open(str(out_path_obj), "w", **profile) as dst_ds:
            dst_ds.write(dst, 1)

        # Hard alignment assertions
        with rasterio.open(str(out_path_obj)) as labels:
            assert labels.crs == ref_crs, f"CRS mismatch: {labels.crs} != {ref_crs}"
            assert (
                labels.transform == ref_transform
            ), f"Transform mismatch: {labels.transform} != {ref_transform}"
            assert labels.width == ref_w, f"Width mismatch: {labels.width} != {ref_w}"
            assert (
                labels.height == ref_h
            ), f"Height mismatch: {labels.height} != {ref_h}"

        return out_path_obj
