import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from dvc.repo import Repo
from torch.utils.data import DataLoader

from landcover.datasets.dataset import LandCoverPatchDataset
from landcover.datasets.transforms import LandCoverAugmentations
from utils.logging import get_logger

logger = get_logger(__name__)


class LandCoverDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Land Cover Classification.
    Manages Train, Val, Test (IID), and OOD splits.
    """

    def __init__(
        self,
        index_path: str,
        norm_stats_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        cloud_frac_max: float = 0.20,
        test_apply_cloud_filter: bool = True,
        augment: bool = True,
        overfit_cfg: dict | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.index_path = index_path
        self.norm_stats_path = norm_stats_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cloud_frac_max = cloud_frac_max
        self.test_apply_cloud_filter = test_apply_cloud_filter
        self.augment = augment
        self.overfit_cfg = overfit_cfg or {}
        self.seed = seed

        self.train_ds: LandCoverPatchDataset | None = None
        self.val_ds: LandCoverPatchDataset | None = None
        self.test_ds: LandCoverPatchDataset | None = None
        self.ood_ds: LandCoverPatchDataset | None = None

    def prepare_data(self):
        """Ensure data is local before training starts."""
        # Check if basic data markers exist
        basic_data_exists = (
            Path(self.index_path).exists() and Path(self.norm_stats_path).exists()
        )
        patches_exist = Path("data/patches").exists()

        if basic_data_exists and patches_exist:
            logger.info("Main data targets already exist locally.")
            return

        # Stage names from dvc.yaml that produce the required data
        # 'assign_splits' -> index, 'compute_norm_stats' -> stats, 'patchify' -> patches
        targets = ["assign_splits", "compute_norm_stats", "patchify"]

        logger.info(f"Pulling data from DVC using stage names: {targets}")
        try:
            repo = Repo()
            repo.pull(targets=targets)
        except Exception as e:
            logger.error(f"Failed to pull data from DVC: {e}")
            logger.warning(
                "Proceeding assuming data might be partially available or manually managed."
            )

    @staticmethod
    def worker_init_fn(worker_id: int):
        """Ensure each worker has a different seed."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            # Train: Filtered, Augmented (unless debug mode)
            is_debug = bool(self.overfit_cfg)
            if is_debug:
                logger.info("Debug mode: augmentations disabled, using subset")
                augmentations = None
                subset_n = self.overfit_cfg.get("subset_n", 100)
            else:
                augmentations = LandCoverAugmentations() if self.augment else None
                subset_n = None

            self.train_ds = LandCoverPatchDataset(
                index_path=self.index_path,
                split="train",
                norm_stats_path=self.norm_stats_path,
                cloud_frac_max=self.cloud_frac_max,
                apply_cloud_filter=True,  # Always filter train
                augmentations=augmentations,
                debug_limit=subset_n,
            )

            # Val: Filtered, No Augmentations (also limited in debug mode)
            self.val_ds = LandCoverPatchDataset(
                index_path=self.index_path,
                split="val",
                norm_stats_path=self.norm_stats_path,
                cloud_frac_max=self.cloud_frac_max,
                apply_cloud_filter=True,  # Always filter val
                augmentations=None,
                debug_limit=subset_n if is_debug else None,
            )

        if (stage == "test" or stage is None) and not self.overfit_cfg:
            # Test (IID): Configurable Filtering
            self.test_ds = LandCoverPatchDataset(
                index_path=self.index_path,
                split="test",
                norm_stats_path=self.norm_stats_path,
                cloud_frac_max=self.cloud_frac_max,
                apply_cloud_filter=self.test_apply_cloud_filter,
                augmentations=None,
            )

            # OOD: No filtering, No Augmentations
            self.ood_ds = LandCoverPatchDataset(
                index_path=self.index_path,
                split="ood",
                norm_stats_path=self.norm_stats_path,
                apply_cloud_filter=False,  # Explicitly unfiltered
                augmentations=None,
            )

    def _get_loader_kwargs(self, shuffle: bool = False):
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "worker_init_fn": self.worker_init_fn,
        }
        if self.num_workers > 0:
            kwargs.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": 2,
                }
            )
        return kwargs

    def train_dataloader(self):
        is_debug = bool(self.overfit_cfg)
        return DataLoader(
            self.train_ds,
            drop_last=not is_debug,
            **self._get_loader_kwargs(shuffle=True),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            **self._get_loader_kwargs(shuffle=False),
        )

    def test_dataloader(self):
        # Return list of loaders: [IID_Test, OOD_Test]
        loaders = [
            DataLoader(
                self.test_ds,
                **self._get_loader_kwargs(shuffle=False),
            )
        ]
        if self.ood_ds is not None and len(self.ood_ds) > 0:
            loaders.append(
                DataLoader(
                    self.ood_ds,
                    **self._get_loader_kwargs(shuffle=False),
                )
            )
        return loaders

    def ood_dataloader(self):
        """Explicit accessor for OOD loader if needed separately"""
        return DataLoader(
            self.ood_ds,
            **self._get_loader_kwargs(shuffle=False),
        )
