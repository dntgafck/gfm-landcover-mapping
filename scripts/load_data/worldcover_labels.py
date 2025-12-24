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
        raise ValueError(f"Could not detect tile id column. Available columns: {list(gdf.columns)}")

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

    def tiles_for_bbox(self, bbox_ll: tuple[float, float, float, float]) -> list[str]:
        """
        bbox_ll = (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
        """
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
    def _mosaic_tiles_over_bbox(
        tile_paths: list[Path],
        bbox_ll: tuple[float, float, float, float],
    ):
        """
        Mosaic tiles and crop to bbox (EPSG:4326).
        Returns (array_2d, transform, crs).
        """
        srcs = [rasterio.open(str(p)) for p in tile_paths]
        try:
            # Crop to bbox to reduce processing
            mosaic, transform = merge(srcs, bounds=bbox_ll)
            crs = srcs[0].crs
            arr = mosaic[0]  # first band
            return arr, transform, crs
        finally:
            for s in srcs:
                s.close()

    def write_labels_for_response_tiff(
        self,
        response_tiff_path: str,
        out_path: str | None = None,
        nodata_value: int = 0,
        compress: str = "LZW",
    ) -> Path:
        """
        Create labels.tiff aligned to Sentinel Hub response.tiff.

        Parameters
        ----------
        response_tiff_path : str
            Path to the Sentinel Hub tile (response.tiff).
        out_path : Optional[str]
            If None, writes next to response.tiff as labels.tiff
        nodata_value : int
            Output nodata (WorldCover commonly uses 0 for nodata)
        compress : str
            GeoTIFF compression

        Returns
        -------
        Path to written labels GeoTIFF.
        """
        response_tiff_path = str(response_tiff_path)
        with rasterio.open(response_tiff_path) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_w, ref_h = ref.width, ref.height
            b = ref.bounds

        if ref_crs is None:
            raise ValueError(f"Reference raster has no CRS: {response_tiff_path}")

        # Best practice: keep Sentinel Hub output in EPSG:4326 for simplicity here
        if ref_crs.to_string() != "EPSG:4326":
            raise ValueError(
                f"Expected EPSG:4326 reference CRS, got {ref_crs}. "
                "Configure Sentinel Hub request to return EPSG:4326."
            )

        bbox_ll = (b.left, b.bottom, b.right, b.top)

        tile_ids = self.tiles_for_bbox(bbox_ll)
        if not tile_ids:
            raise RuntimeError(f"No WorldCover tiles found for bbox {bbox_ll}")

        # Download only required tiles
        tile_paths = [self._download_tile_if_missing(tid) for tid in tile_ids]

        # Mosaic over bbox
        src_arr, src_transform, src_crs = self._mosaic_tiles_over_bbox(tile_paths, bbox_ll)

        # Warp to reference grid
        dst = np.full((ref_h, ref_w), nodata_value, dtype=np.uint8)

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
            src_nodata=None,
            dst_nodata=nodata_value,
        )

        # Output path
        if out_path is None:
            out_path_obj = Path(response_tiff_path).parent / "labels.tiff"
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
            "nodata": nodata_value,
            "compress": compress,
            "tiled": True,
        }

        with rasterio.open(str(out_path_obj), "w", **profile) as dst_ds:
            dst_ds.write(dst, 1)

        return out_path_obj


def generate_labels_for_aoi_folder(
    aoi_root: str,
    worldcover_grid_geojson: str = "data/worldcover/v200/2021/esa_worldcover_grid.geojson",
    cache_dir: str = "data/worldcover/cache",
    overwrite: bool = False,
) -> int:
    """
    Convenience utility: walk `data/<AOI>/sh/**/response.tiff` and
    create `labels.tiff` next to each.

    Returns number of labels written.
    """
    cfg = WorldCoverS3Config(cache_dir=cache_dir)
    labeler = WorldCoverLabeler(worldcover_grid_geojson, cfg)

    aoi_root_path = Path(aoi_root)
    written = 0

    for resp in aoi_root_path.rglob("response.tiff"):
        out = resp.parent / "labels.tiff"
        if out.exists() and not overwrite:
            continue
        labeler.write_labels_for_response_tiff(str(resp), out_path=str(out))
        written += 1

    return written
