import logging
import os

import geopandas as gpd
import numpy as np
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
)
from tqdm import tqdm

from .auth import get_config

# Setup logging
logger = logging.getLogger(__name__)


class SentinelDataLoader:
    """
    A class to handle data loading from Sentinel Hub.
    """

    # Define Custom DataCollection for CDSE
    # This forces the SDK to use the CDSE endpoint instead of the legacy Sentinel Hub URL
    CDSE_S2_L2A = DataCollection.define(
        "SENTINEL2_L2A_CDSE",
        api_id="sentinel-2-l2a",
        service_url="https://sh.dataspace.copernicus.eu",
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
            input: ["B02", "B03", "B04", "B08", "SCL"],
            output: { bands: 5, sampleType: "FLOAT32" }
          };
        }

        function evaluatePixel(sample) {
          return [sample.B02, sample.B03, sample.B04, sample.B08, sample.SCL];
        }
        """

    def download_data(
        self,
        bbox_coords: tuple[float, float, float, float],
        time_interval: tuple[str, str],
        resolution: int = 10,
        output_folder: str = "sh_out",
    ) -> np.ndarray:
        """
        Downloads data for a given bounding box and time interval.
        """
        aoi_bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

        request = SentinelHubRequest(
            evalscript=self.get_evalscript(),
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.CDSE_S2_L2A,
                    time_interval=time_interval,
                    mosaicking_order="mostRecent",
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=aoi_bbox,
            size=size,
            config=self.config,
            data_folder=output_folder,
        )

        data = request.get_data(save_data=True)
        arr = data[0]

        logger.info(f"Downloaded data shape: {arr.shape}")
        return arr

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

        os.makedirs(output_folder, exist_ok=True)

        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Downloading tiles"):
            try:
                # Use total_bounds of the geometry
                # row.geometry.bounds gives (minx, miny, maxx, maxy)
                bounds = row.geometry.bounds

                # Construct file name
                if id_column and id_column in row:
                    file_name = f"tile_{row[id_column]}.tiff"
                else:
                    file_name = f"tile_{idx}.tiff"

                # Check if already exists to skip (simple caching)
                output_path = os.path.join(output_folder, file_name)
                if os.path.exists(output_path):
                    continue

                self.download_data(
                    bbox_coords=bounds,
                    time_interval=time_interval,
                    resolution=resolution,
                    output_folder=os.path.join(output_folder, f"tile_{idx}"),
                )

            except Exception as e:
                logger.error(f"Failed to download tile {idx}: {e}")

        logger.info("Batch download complete.")
