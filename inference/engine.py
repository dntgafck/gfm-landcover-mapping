"""Inference engine with sliding window support."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import rasterio
from omegaconf import DictConfig
from rasterio.windows import Window

from inference.catalog import TileCatalog
from inference.schemas import ClassHistogram, InferRequest, WindowInfo
from inference.utils import TimingAccumulator
from utils.logging import get_logger

logger = get_logger(__name__)

# ESA WorldCover class names (indices 0-10)
WORLDCOVER_CLASSES = [
    "Tree cover",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Bare / sparse vegetation",
    "Snow and ice",
    "Permanent water bodies",
    "Herbaceous wetland",
    "Mangroves",
    "Moss and lichen",
]


class InferenceEngine:
    """Engine for running ONNX inference with sliding window support."""

    def __init__(
        self,
        session: ort.InferenceSession,
        catalog: TileCatalog,
        mean: list[float],
        std: list[float],
        cfg: DictConfig,
    ) -> None:
        """Initialize inference engine.

        Args:
            session: ONNX Runtime inference session
            catalog: Tile catalog for data access
            mean: Normalization mean per channel
            std: Normalization std per channel
            cfg: Hydra configuration
        """
        self.session = session
        self.catalog = catalog
        self.mean = np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(1, -1, 1, 1)
        self.cfg = cfg

        # Get input/output names
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

        # Default runtime settings
        self.default_patch_size = cfg.runtime.patch_size
        self.default_stride = cfg.runtime.stride
        self.default_batch_size = cfg.runtime.batch_size

    def read_window(
        self,
        tile_path: Path,
        row_off: int,
        col_off: int,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, int, int]:
        """Read a window from a GeoTIFF tile.

        Args:
            tile_path: Path to tile file
            row_off: Row offset
            col_off: Column offset
            height: Window height
            width: Window width

        Returns:
            Tuple of (image array [C, H, W], actual_height, actual_width)
        """
        with rasterio.open(tile_path) as src:
            # Clip window to tile bounds
            actual_height = min(height, src.height - row_off)
            actual_width = min(width, src.width - col_off)

            if actual_height <= 0 or actual_width <= 0:
                raise ValueError(
                    f"Window out of bounds: offset ({row_off}, {col_off}) "
                    f"for tile of size ({src.height}, {src.width})"
                )

            window = Window(col_off, row_off, actual_width, actual_height)
            image = src.read(window=window)  # [C, H, W]

        return image.astype(np.float32), actual_height, actual_width

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference.

        Args:
            image: Image array [C, H, W] or [N, C, H, W]

        Returns:
            Normalized image array [N, C, H, W]
        """
        # Add batch dimension if needed
        if image.ndim == 3:
            image = image[np.newaxis, ...]  # [1, C, H, W]

        # Normalize
        normalized = (image - self.mean) / self.std
        return normalized

    def _extract_patches(
        self,
        image: np.ndarray,
        patch_size: int,
        stride: int,
    ) -> tuple[np.ndarray, list[tuple[int, int]], int, int]:
        """Extract patches from image using sliding window.

        Args:
            image: Image array [C, H, W]
            patch_size: Size of each patch
            stride: Stride between patches

        Returns:
            Tuple of (patches [N, C, H, W], positions [(row, col), ...], n_rows, n_cols)
        """
        c, h, w = image.shape
        patches = []
        positions = []

        # Calculate number of patches
        n_rows = max(1, (h - patch_size) // stride + 1)
        n_cols = max(1, (w - patch_size) // stride + 1)

        # Handle edge cases where image is smaller than patch_size
        if h <= patch_size:
            n_rows = 1
        if w <= patch_size:
            n_cols = 1

        for i in range(n_rows):
            for j in range(n_cols):
                row = min(i * stride, max(0, h - patch_size))
                col = min(j * stride, max(0, w - patch_size))

                # Extract patch (pad if necessary)
                patch = np.zeros((c, patch_size, patch_size), dtype=image.dtype)
                end_row = min(row + patch_size, h)
                end_col = min(col + patch_size, w)
                patch[:, : end_row - row, : end_col - col] = image[
                    :, row:end_row, col:end_col
                ]

                patches.append(patch)
                positions.append((row, col))

        return np.stack(patches), positions, n_rows, n_cols

    def _stitch_predictions(
        self,
        predictions: np.ndarray,
        positions: list[tuple[int, int]],
        output_shape: tuple[int, int],
        patch_size: int,
    ) -> np.ndarray:
        """Stitch patch predictions back into full image.

        Uses averaging for overlapping regions.

        Args:
            predictions: Patch predictions [N, num_classes, H, W]
            positions: List of (row, col) positions for each patch
            output_shape: Target output shape (H, W)
            patch_size: Size of each patch

        Returns:
            Stitched logits array [num_classes, H, W]
        """
        h, w = output_shape
        num_classes = predictions.shape[1]

        # Accumulate logits and counts for averaging
        logits_sum = np.zeros((num_classes, h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

        for pred, (row, col) in zip(predictions, positions):
            end_row = min(row + patch_size, h)
            end_col = min(col + patch_size, w)
            pred_h = end_row - row
            pred_w = end_col - col

            logits_sum[:, row:end_row, col:end_col] += pred[:, :pred_h, :pred_w]
            counts[row:end_row, col:end_col] += 1

        # Average overlapping regions
        counts = np.maximum(counts, 1)  # Avoid division by zero
        logits_avg = logits_sum / counts[np.newaxis, :, :]

        return logits_avg

    def run_batch_inference(self, patches: np.ndarray, batch_size: int) -> np.ndarray:
        """Run inference on patches.

        Note: ONNX model has fixed batch size 1, so we process one patch at a time.

        Args:
            patches: Input patches [N, C, H, W]
            batch_size: Not used (kept for API compatibility)

        Returns:
            Output logits [N, num_classes, H, W]
        """
        outputs = []

        for i in range(len(patches)):
            # Process one patch at a time (model expects batch=1)
            patch = patches[i : i + 1]  # Keep [1, C, H, W] shape
            patch_normalized = self.preprocess(patch)

            result = self.session.run(
                [self.output_name],
                {self.input_name: patch_normalized},
            )[
                0
            ]  # [1, num_classes, H, W]
            outputs.append(result)

        return np.concatenate(outputs, axis=0)

    def sliding_window_inference(
        self,
        image: np.ndarray,
        patch_size: int,
        stride: int,
        batch_size: int,
    ) -> np.ndarray:
        """Run sliding window inference on an image.

        Args:
            image: Input image [C, H, W]
            patch_size: Size of inference patches
            stride: Stride between patches
            batch_size: Batch size for inference

        Returns:
            Class predictions [H, W]
        """
        c, h, w = image.shape

        # If image fits in single patch, run directly
        if h <= patch_size and w <= patch_size:
            # Pad to patch_size if needed
            padded = np.zeros((c, patch_size, patch_size), dtype=image.dtype)
            padded[:, :h, :w] = image
            normalized = self.preprocess(padded)

            logits = self.session.run(
                [self.output_name],
                {self.input_name: normalized},
            )[0][
                0
            ]  # [num_classes, H, W]

            # Crop to original size and argmax
            return np.argmax(logits[:, :h, :w], axis=0)

        # Extract patches
        patches, positions, _, _ = self._extract_patches(image, patch_size, stride)

        # Run batch inference
        patch_logits = self.run_batch_inference(patches, batch_size)

        # Stitch back together
        stitched_logits = self._stitch_predictions(
            patch_logits, positions, (h, w), patch_size
        )

        # Argmax to get class predictions
        return np.argmax(stitched_logits, axis=0).astype(np.uint8)

    def run_inference(
        self,
        request: InferRequest,
        timings: TimingAccumulator,
    ) -> tuple[np.ndarray, np.ndarray | None, WindowInfo]:
        """Run full inference pipeline for a request.

        Args:
            request: Inference request
            timings: Timing accumulator

        Returns:
            Tuple of (predictions, labels or None, window_info)
        """
        # Get runtime settings (use overrides if provided)
        patch_size = request.patch_size or self.default_patch_size
        stride = request.stride or self.default_stride
        batch_size = request.batch_size or self.default_batch_size

        # Get tile paths
        with timings.time("catalog_lookup"):
            paths = self.catalog.get_tile_paths(request.country, request.tile_id)

        # Read imagery window
        with timings.time("read_imagery"):
            image, actual_h, actual_w = self.read_window(
                paths["imagery"],
                request.row_off,
                request.col_off,
                request.height,
                request.width,
            )

        # Run inference
        with timings.time("inference"):
            predictions = self.sliding_window_inference(
                image, patch_size, stride, batch_size
            )

        # Read labels if available and requested
        labels = None
        if request.include_label and paths["label"]:
            with timings.time("read_labels"):
                with rasterio.open(paths["label"]) as src:
                    window = Window(
                        request.col_off, request.row_off, actual_w, actual_h
                    )
                    labels = src.read(1, window=window)

                    # Remap ESA WorldCover labels to contiguous [0, 10]
                    remapped = np.full_like(labels, 255)
                    remapped[labels == 10] = 0
                    remapped[labels == 20] = 1
                    remapped[labels == 30] = 2
                    remapped[labels == 40] = 3
                    remapped[labels == 50] = 4
                    remapped[labels == 60] = 5
                    remapped[labels == 70] = 6
                    remapped[labels == 80] = 7
                    remapped[labels == 90] = 8
                    remapped[labels == 95] = 9
                    remapped[labels == 100] = 10
                    labels = remapped

        window_info = WindowInfo(
            row_off=request.row_off,
            col_off=request.col_off,
            height=request.height,
            width=request.width,
            actual_height=actual_h,
            actual_width=actual_w,
        )

        return predictions, labels, window_info


def compute_histogram(array: np.ndarray, num_classes: int = 11) -> list[ClassHistogram]:
    """Compute class histogram from prediction/label array.

    Args:
        array: Class ID array
        num_classes: Number of classes

    Returns:
        List of ClassHistogram objects
    """
    total_pixels = array.size
    histograms = []

    for class_id in range(num_classes):
        count = int(np.sum(array == class_id))
        histograms.append(
            ClassHistogram(
                class_id=class_id,
                class_name=WORLDCOVER_CLASSES[class_id],
                pixel_count=count,
                fraction=count / total_pixels if total_pixels > 0 else 0.0,
            )
        )

    return histograms
