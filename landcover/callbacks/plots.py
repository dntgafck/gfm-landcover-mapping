import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from utils.logging import get_logger

logger = get_logger(__name__)


class PlotLoggerCallback(pl.Callback):
    """
    Callback to save training plots to the plots directory.
    """

    def __init__(self, output_dir: str = "plots"):
        super().__init__()
        self.output_dir = output_dir
        self.metrics_history = {
            "train/loss_epoch": [],
            "val/loss": [],
            "val/mIoU": [],
        }
        self.epochs = []

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        self.epochs.append(trainer.current_epoch)

        # Collect metrics
        for key in self.metrics_history.keys():
            val = trainer.callback_metrics.get(key)
            if val is not None:
                self.metrics_history[key].append(val.item())
            else:
                # If metric is missing for some reason, append None or last value
                if len(self.metrics_history[key]) > 0:
                    self.metrics_history[key].append(self.metrics_history[key][-1])
                else:
                    self.metrics_history[key].append(0.0)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Generating training plots in {self.output_dir}...")

        # Plot 1: Training Loss
        self._plot_metric(
            self.epochs,
            self.metrics_history["train/loss_epoch"],
            "Training Loss",
            "Epoch",
            "Loss",
            os.path.join(self.output_dir, "train_loss.png"),
        )

        # Plot 2: Validation Loss
        self._plot_metric(
            self.epochs,
            self.metrics_history["val/loss"],
            "Validation Loss",
            "Epoch",
            "Loss",
            os.path.join(self.output_dir, "val_loss.png"),
        )

        # Plot 3: Validation mIoU
        self._plot_metric(
            self.epochs,
            self.metrics_history["val/mIoU"],
            "Validation mIoU",
            "Epoch",
            "mIoU",
            os.path.join(self.output_dir, "val_miou.png"),
        )

    def _plot_metric(self, x, y, title, xlabel, ylabel, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker="o", linestyle="-", color="b")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved plot to {save_path}")
