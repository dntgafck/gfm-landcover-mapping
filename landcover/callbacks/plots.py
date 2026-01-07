from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from matplotlib.figure import Figure
from pytorch_lightning.loggers import MLFlowLogger

from utils.logging import get_logger

logger = get_logger(__name__)


class PlotLoggerCallback(pl.Callback):
    """
    Callback to save training plots to the plots directory.

    Collects train metrics in on_train_epoch_end and
    val metrics in on_validation_epoch_end to avoid shifted curves.
    Only plots metrics that were actually observed (None values are skipped).
    Optionally logs plots as MLflow artifacts if an MLflow logger is active.

    Args:
        output_dir: Path to the directory where plots will be saved.
                   Typically run_dir/artifacts/plots/
    """

    def __init__(self, output_dir: str | Path = "plots"):
        super().__init__()
        self.output_dir = Path(output_dir)
        # Store None for missing metrics instead of 0.0
        self.metrics_history: dict[str, list[float | None]] = {
            "train/loss_epoch": [],
            "val/loss": [],
            "val/mIoU": [],
        }
        self.train_epochs: list[int] = []
        self.val_epochs: list[int] = []

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Collect train metrics only."""
        self.train_epochs.append(trainer.current_epoch)

        train_loss = trainer.callback_metrics.get("train/loss_epoch")
        if train_loss is not None:
            self.metrics_history["train/loss_epoch"].append(train_loss.item())
        else:
            self.metrics_history["train/loss_epoch"].append(None)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Collect val metrics only."""
        # Skip sanity check validation (before training starts)
        if trainer.sanity_checking:
            return

        self.val_epochs.append(trainer.current_epoch)

        for key in ["val/loss", "val/mIoU"]:
            val = trainer.callback_metrics.get(key)
            if val is not None:
                self.metrics_history[key].append(val.item())
            else:
                self.metrics_history[key].append(None)

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not trainer.is_global_zero:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating training plots in {self.output_dir}...")

        # Plot 1: Training Loss
        self._create_and_log_plot(
            self.train_epochs,
            self.metrics_history["train/loss_epoch"],
            "Training Loss",
            "Epoch",
            "Loss",
            self.output_dir / "train_loss.png",
        )

        # Plot 2: Validation Loss
        self._create_and_log_plot(
            self.val_epochs,
            self.metrics_history["val/loss"],
            "Validation Loss",
            "Epoch",
            "Loss",
            self.output_dir / "val_loss.png",
        )

        # Plot 3: Validation mIoU
        self._create_and_log_plot(
            self.val_epochs,
            self.metrics_history["val/mIoU"],
            "Validation mIoU",
            "Epoch",
            "mIoU",
            self.output_dir / "val_miou.png",
        )

    def _create_and_log_plot(
        self,
        epochs: list[int],
        values: list[float | None],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: Path,
    ) -> None:
        """Create a plot, save to disk, and log to MLflow if available."""
        fig = self._create_figure(epochs, values, title, xlabel, ylabel)
        if fig is None:
            return

        # Save to disk
        fig.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")
        plt.close(fig)

    def _create_figure(
        self,
        epochs: list[int],
        values: list[float | None],
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> Figure | None:
        """Create a matplotlib figure, filtering out None values."""
        # Filter out None values
        valid_data = [
            (e, v) for e, v in zip(epochs, values, strict=False) if v is not None
        ]

        if not valid_data:
            logger.warning(f"No data to plot for '{title}'. Skipping.")
            return None

        x, y = zip(*valid_data, strict=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, marker="o", linestyle="-")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return fig
