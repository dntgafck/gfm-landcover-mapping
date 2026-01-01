import pytorch_lightning as pl
from torch.utils.data import DataLoader

from landcover.datasets.dataset import LandCoverPatchDataset
from landcover.datasets.transforms import LandCoverAugmentations


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
        self.seed = seed

        self.train_ds: LandCoverPatchDataset | None = None
        self.val_ds: LandCoverPatchDataset | None = None
        self.test_ds: LandCoverPatchDataset | None = None
        self.ood_ds: LandCoverPatchDataset | None = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            # Train: Filtered, Augmented
            augmentations = LandCoverAugmentations() if self.augment else None
            self.train_ds = LandCoverPatchDataset(
                index_path=self.index_path,
                split="train",
                norm_stats_path=self.norm_stats_path,
                cloud_frac_max=self.cloud_frac_max,
                apply_cloud_filter=True,  # Always filter train
                augmentations=augmentations,
            )

            # Val: Filtered, No Augmentations
            self.val_ds = LandCoverPatchDataset(
                index_path=self.index_path,
                split="val",
                norm_stats_path=self.norm_stats_path,
                cloud_frac_max=self.cloud_frac_max,
                apply_cloud_filter=True,  # Always filter val
                augmentations=None,
            )

        if stage == "test" or stage is None:
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

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        # Return list of loaders: [IID_Test, OOD_Test]
        # Or just IID? Usually Trainer treats list as separate dataloaders
        loaders = [
            DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        ]
        if self.ood_ds is not None and len(self.ood_ds) > 0:
            loaders.append(
                DataLoader(
                    self.ood_ds,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            )
        return loaders

    def ood_dataloader(self):
        """Explicit accessor for OOD loader if needed separately"""
        return DataLoader(
            self.ood_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )
