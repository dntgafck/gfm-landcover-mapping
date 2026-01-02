import json
from collections.abc import Callable
from typing import Any

import dvc.api
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from landcover.datasets.transforms import normalize_image
from utils.logging import get_logger

logger = get_logger(__name__)


class LandCoverPatchDataset(Dataset):
    """
    PyTorch Dataset for Land Cover patches.
    Enforces strict split policy, cloud filtering, and frozen normalization.
    Uses dvc.api for all file access.
    """

    def __init__(
        self,
        index_path: str,
        split: str,
        norm_stats_path: str,
        cloud_frac_max: float = 0.20,
        apply_cloud_filter: bool = True,
        augmentations: Callable | None = None,
        debug_limit: int | None = None,
        repo_root: str = ".",
    ):
        """
        Args:
            index_path: Path to the dataset index CSV.
            split: 'train', 'val', 'test', or 'ood'.
            norm_stats_path: Path to the normalization statistics JSON.
            cloud_frac_max: Maximum allowed cloud fraction (for filtered splits).
            apply_cloud_filter: Whether to apply cloud filtering.
            augmentations: Optional augmentation callable.
            debug_limit: Limit number of samples for debugging.
            repo_root: Root of the DVC repository.
        """
        self.index_path = index_path
        self.split = split
        self.repo_root = repo_root
        self.augmentations = augmentations

        # Load Index CSV via DVC
        logger.info(f"Loading index from {index_path} via dvc.api...")
        with dvc.api.open(index_path, repo=repo_root, mode="r") as f:
            self.df = pd.read_csv(f)

        # Filter by Split
        initial_count = len(self.df)
        self.df = self.df[self.df["split"] == split].copy()
        logger.info(f"Split '{split}': {len(self.df)}/{initial_count} patches.")

        # Apply Cloud Filtering
        if apply_cloud_filter:
            pre_filter_count = len(self.df)
            self.df = self.df[self.df["cloud_frac"] <= cloud_frac_max]
            logger.info(
                f"Cloud Filter (max {cloud_frac_max}): "
                f"{len(self.df)}/{pre_filter_count} patches kept."
            )
        else:
            logger.info(f"Cloud Filter: DISABLED for split '{split}'.")

        if debug_limit:
            self.df = self.df.iloc[:debug_limit]

        # Load Norm Stats via DVC
        logger.info(f"Loading norm stats from {norm_stats_path} via dvc.api...")
        with dvc.api.open(norm_stats_path, repo=repo_root, mode="r") as f:
            stats = json.load(f)
            self.bands = stats["bands"]
            self.mean = stats["mean"]
            self.std = stats["std"]

        logger.info(f"Dataset '{split}' initialized. Size: {len(self.df)}")
        if len(self.df) == 0:
            logger.warning("WARNING: Dataset is empty!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        # Paths
        spectral_path = row["spectral_path"]
        label_path = row["label_path"]

        # Read Spectral Data (4 bands)
        # using dvc.api.open -> read bytes -> rasterio.MemoryFile
        try:
            with dvc.api.open(spectral_path, repo=self.repo_root, mode="rb") as f:
                content = f.read()
                with rasterio.MemoryFile(content) as memfile:
                    with memfile.open() as src:
                        # shape: (C, H, W)
                        image = src.read()

                        if image.shape[0] != 4:
                            raise ValueError(
                                f"Expected 4 bands, got {image.shape[0]} at {spectral_path}"
                            )

                        image = image.astype(np.float32)

            # Read Label Data (1 band)
            with dvc.api.open(label_path, repo=self.repo_root, mode="rb") as f:
                content = f.read()
                with rasterio.MemoryFile(content) as memfile:
                    with memfile.open() as src:
                        # shape: (1, H, W) -> squeeze to (H, W)
                        mask = src.read(1)
                        mask = mask.astype(np.int64)

        except Exception as e:
            logger.error(f"Error loading sample {idx} (patch_id: {row.get('patch_id')}): {e}")
            raise e

        # Assert shapes
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError(f"Shape mismatch: Image {image.shape} vs Mask {mask.shape}")

        # Convert to Tensor
        image_t = torch.from_numpy(image)  # (C, H, W)
        mask_t = torch.from_numpy(mask)  # (H, W)

        # Remap ESA WorldCover labels to contiguous [0, 10]
        # 10->0, 20->1, 30->2, 40->3, 50->4, 60->5, 70->6, 80->7, 90->8, 95->9, 100->10
        # Others -> 255 (ignore)
        remapped_mask = torch.full_like(mask_t, 255)
        remapped_mask[mask_t == 10] = 0
        remapped_mask[mask_t == 20] = 1
        remapped_mask[mask_t == 30] = 2
        remapped_mask[mask_t == 40] = 3
        remapped_mask[mask_t == 50] = 4
        remapped_mask[mask_t == 60] = 5
        remapped_mask[mask_t == 70] = 6
        remapped_mask[mask_t == 80] = 7
        remapped_mask[mask_t == 90] = 8
        remapped_mask[mask_t == 95] = 9
        remapped_mask[mask_t == 100] = 10

        mask_t = remapped_mask

        # Basic alignment check
        if torch.isnan(image_t).any() or torch.isinf(image_t).any():
            raise ValueError(f"NaN/Inf found in image {spectral_path}")

        sample = {
            "image": image_t,
            "mask": mask_t,
            "patch_id": row["patch_id"],
            "country": row["country"],
            "split": row["split"],
        }

        # Apply Augmentations (Train only usually)
        if self.augmentations:
            sample = self.augmentations(sample)

        # Normalize (Always)
        sample["image"] = normalize_image(sample["image"], self.mean, self.std)

        return sample
