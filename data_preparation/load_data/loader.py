import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import geopandas as gpd
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    MosaickingOrder,
    ResamplingType,
    SentinelHubRequest,
)
from tqdm import tqdm

from utils.logging import get_logger

from .auth import get_config

EU_CRS = CRS(3035)  # EPSG:3035
# Setup logging
logger = get_logger(__name__)


class SentinelDataLoader:
    """
    A class to handle data loading from Sentinel Hub.
    """

    CDSE_S2_L2A = DataCollection.SENTINEL2_L2A.define_from(
        "CDSE_S2_L2A", service_url="https://sh.dataspace.copernicus.eu"
    )

    def __init__(self):
        self.config = get_config()

    def get_evalscript(self) -> str:
        """
        Returns the evalscript for downloading B02, B03, B04, B08, and SCL bands.
        """
        return """
        //VERSION=3
        function setup() {
          return {
            input: [
              {
                bands: ["B02", "B03", "B04", "B08", "SCL", "dataMask"]
              }
            ],
            output: [
              {
                id: "spectral",
                bands: 4,
                sampleType: "FLOAT32"
              },
              {
                id: "scl",
                bands: 1,
                sampleType: "UINT8"
              },
              {
                id: "mask",
                bands: 1,
                sampleType: "UINT8"
              }
            ]
          };
        }

        function evaluatePixel(sample) {
          return {
            spectral: [
              sample.B02,
              sample.B03,
              sample.B04,
              sample.B08
            ],
            scl: [sample.SCL],
            mask: [sample.dataMask]
          };
        }
        """

    def compute_cache_key(
        self,
        bbox: BBox,
        time_interval: tuple[str, str],
        resolution: int,
        collection_name: str,
        evalscript: str,
    ) -> str:
        """
        Computes a deterministic SHA256 cache key based on all request parameters.
        """
        # Quantize bbox to 6 decimal places to handle float precision issues
        # BBox object is iterable and yields (minx, miny, maxx, maxy)
        bbox_list = [round(x, 6) for x in bbox]

        # Create a dictionary of parameters
        params = {
            "bbox": bbox_list,
            "crs": str(bbox.crs),
            "start_date": time_interval[0],
            "end_date": time_interval[1],
            "collection": collection_name,
            "resolution": resolution,
            "evalscript": evalscript,
            "mosaicking_order": "mostRecent",  # Hardcoded in current implementation
        }

        # Validate that evalscript is not empty
        if not evalscript or not evalscript.strip():
            raise ValueError("Evalscript cannot be empty for cache key generation")

        # Serializing to JSON with sorting keys ensuring determinism
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(params_str.encode("utf-8")).hexdigest()

    def download_data(
        self,
        bbox_coords: tuple[float, float, float, float],
        time_interval: tuple[str, str],
        resolution: int = 10,
        output_folder: str = "sh_out",
    ):
        # IMPORTANT: bbox must be in projected CRS
        aoi_bbox = BBox(bbox=bbox_coords, crs=EU_CRS)

        request = SentinelHubRequest(
            evalscript=self.get_evalscript(),
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.CDSE_S2_L2A,
                    time_interval=time_interval,
                    mosaicking_order=MosaickingOrder.MOST_RECENT,
                    upsampling=ResamplingType.NEAREST,
                    downsampling=ResamplingType.NEAREST,
                )
            ],
            responses=[
                SentinelHubRequest.output_response("spectral", MimeType.TIFF),
                SentinelHubRequest.output_response("scl", MimeType.TIFF),
                SentinelHubRequest.output_response("mask", MimeType.TIFF),
            ],
            bbox=aoi_bbox,
            resolution=(resolution, resolution),
            config=self.config,
            data_folder=output_folder,
        )

        data = request.get_data(save_data=True)
        return data

    def download_batch(
        self,
        input_data: str | gpd.GeoDataFrame,
        time_interval: tuple[str, str],
        resolution: int = 10,
        output_folder: str = "sh_out",
        id_column: str | None = None,
    ):
        """
        Downloads data for each feature in the input GeoDataFrame or GeoJSON file.
        """
        if isinstance(input_data, str):
            gdf = gpd.read_file(input_data)
        else:
            gdf = input_data

        if gdf.empty:
            logger.warning("Input grid is empty.")
            return

        logger.info(f"Starting batch download for {len(gdf)} items.")

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Downloading tiles"):
            try:
                # Use total_bounds of the geometry
                bounds = row.geometry.bounds  # now in EPSG:3035 meters
                aoi_bbox = BBox(bbox=bounds, crs=EU_CRS)

                # Determine tile ID (for logging/manifest only, not directory structure)
                if id_column and id_column in gdf.columns:
                    tile_id = str(row[id_column])
                else:
                    tile_id = str(idx)

                # Parameters for cache key
                evalscript = self.get_evalscript()
                collection_name = self.CDSE_S2_L2A.name

                # Compute cache key
                cache_key = self.compute_cache_key(
                    bbox=aoi_bbox,
                    time_interval=time_interval,
                    resolution=resolution,
                    collection_name=collection_name,
                    evalscript=evalscript,
                )

                # Target directory: output_folder/<cache_key>
                target_dir = Path(output_folder) / cache_key

                # 1. Check idempotency
                # We check for spectral.tif because response.tiff is only for single-file outputs
                if (target_dir / "spectral.tif").exists():
                    logger.info(f"Tile {tile_id} (key: {cache_key}) exists. Skipping.")
                    continue

                # 2. Download to a temporary location
                # We use a temp dir specific to this download to avoid collisions
                temp_download_dir = Path(output_folder) / f"temp_{cache_key}"
                if temp_download_dir.exists():
                    shutil.rmtree(temp_download_dir)
                temp_download_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Downloading tile {tile_id}...")
                self.download_data(
                    bbox_coords=bounds,
                    time_interval=time_interval,
                    resolution=resolution,
                    output_folder=str(temp_download_dir),
                )

                # 3. Locate the result
                # SentinelHubRequest creates a subfolder with the request_id
                # We expect exactly one subfolder in our temp dir
                subfolders = [f for f in temp_download_dir.iterdir() if f.is_dir()]
                if not subfolders:
                    raise FileNotFoundError(
                        "Sentinel Hub did not create an output directory."
                    )

                sh_result_dir = subfolders[0]

                # 4. Move to target location (atomic-ish rename)
                if target_dir.exists():
                    shutil.rmtree(target_dir)

                shutil.move(str(sh_result_dir), str(target_dir))

                # 4.5 Extract tar if present
                tar_path = target_dir / "response.tar"
                if tar_path.exists():
                    logger.info(f"Extracting {tar_path}...")
                    import tarfile

                    with tarfile.open(tar_path) as tar:
                        tar.extractall(path=target_dir)
                    tar_path.unlink()

                # Cleanup temp dir
                shutil.rmtree(temp_download_dir)

                # 5. Write manifest.json
                manifest = {
                    "aoi_id": tile_id,
                    "tile_id": tile_id,
                    "bbox": [round(x, 6) for x in bounds],
                    "time": {"start": time_interval[0], "end": time_interval[1]},
                    "collection": collection_name,
                    "resolution_m": resolution,
                    "crs": "EPSG:3035",
                    "outputs": {
                        "spectral": ["B02", "B03", "B04", "B08"],
                        "scl": ["SCL"],
                        "mask": ["dataMask"],
                    },
                    "mosaic": "mostRecent",
                    "evalscript_sha256": hashlib.sha256(
                        evalscript.encode("utf-8")
                    ).hexdigest(),
                    "cache_key": cache_key,
                    "created_utc": datetime.utcnow().isoformat(),
                }

                with open(target_dir / "manifest.json", "w") as f:
                    json.dump(manifest, f, indent=2)

                # 6. Ensure request.json exists (Sentinel Hub SDK usually writes it)
                # If SDK writes it as request.json in the result dir, it's already there.

                logger.info(f"Saved tile {tile_id} to {target_dir}")

            except Exception as e:
                logger.error(f"Failed to download tile {idx}: {e}")

        logger.info("Batch download complete.")
