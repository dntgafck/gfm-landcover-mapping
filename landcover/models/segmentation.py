import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex


class LandCoverSegmentationModule(pl.LightningModule):
    """
    LightningModule for Land Cover Segmentation.
    Handles training, validation, testing, metrics, and optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 11,
        ignore_index: int = 0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # Metrics
        metrics = MetricCollection(
            {
                "mIoU": MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index),
                "macro_f1": MulticlassF1Score(
                    num_classes=num_classes, ignore_index=ignore_index, average="macro"
                ),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_iid_metrics = metrics.clone(prefix="test_iid/")
        self.test_ood_metrics = metrics.clone(prefix="test_ood/")

        # For per-class IoU (optional but recommended)
        self.test_iid_iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=ignore_index, average=None
        )
        self.test_ood_iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=ignore_index, average=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, batch_idx, prefix: str):
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)
        loss = self.criterion(logits, masks)

        return loss, logits, masks

    def training_step(self, batch, batch_idx):
        loss, logits, masks = self._shared_step(batch, batch_idx, "train")

        # Log training loss
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=masks.size(0)
        )

        # Update metrics
        self.train_metrics.update(logits, masks)

        return loss

    def on_train_epoch_end(self):
        # Log aggregated metrics
        output = self.train_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, logits, masks = self._shared_step(batch, batch_idx, "val")

        # Log validation loss
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=masks.size(0)
        )

        # Update metrics
        self.val_metrics.update(logits, masks)

        return loss

    def on_validation_epoch_end(self):
        # Log aggregated metrics
        output = self.val_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)
        loss = self.criterion(logits, masks)

        if dataloader_idx == 0:
            # IID Test
            self.test_iid_metrics.update(logits, masks)
            self.test_iid_iou_per_class.update(logits, masks)
            self.log(
                "test_iid/loss",
                loss,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
                batch_size=masks.size(0),
            )
        else:
            # OOD Test
            self.test_ood_metrics.update(logits, masks)
            self.test_ood_iou_per_class.update(logits, masks)
            self.log(
                "test_ood/loss",
                loss,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
                batch_size=masks.size(0),
            )

    def on_test_epoch_end(self):
        # Log IID metrics
        iid_metrics = self.test_iid_metrics.compute()
        self.log_dict(iid_metrics)
        self.test_iid_metrics.reset()

        # Log IID per-class IoU
        iid_per_class = self.test_iid_iou_per_class.compute()
        for i, iou in enumerate(iid_per_class):
            if i != self.ignore_index:
                self.log(f"test_iid/iou_class_{i}", iou)
        self.test_iid_iou_per_class.reset()

        # Log OOD metrics (if they were updated)
        try:
            ood_metrics = self.test_ood_metrics.compute()
            # Only log if we have actual OOD data (mIoU will be NaN if no updates)
            if not torch.isnan(ood_metrics["test_ood/mIoU"]):
                self.log_dict(ood_metrics)
            self.test_ood_metrics.reset()

            ood_per_class = self.test_ood_iou_per_class.compute()
            if not torch.isnan(ood_per_class).all():
                for i, iou in enumerate(ood_per_class):
                    if i != self.ignore_index:
                        self.log(f"test_ood/iou_class_{i}", iou)
            self.test_ood_iou_per_class.reset()
        except Exception:
            # If OOD was never used, compute() might fail or be empty
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
