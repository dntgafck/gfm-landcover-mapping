"""Debug/overfit training entrypoint for sanity testing."""

import sys

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from landcover.callbacks.plots import PlotLoggerCallback
from landcover.training.common import (
    create_datamodule,
    create_model,
    create_module,
    load_or_compute_class_weights,
)
from utils.logging import get_logger

logger = get_logger(__name__)

# PASS/FAIL thresholds
MIOU_PASS_THRESHOLD = 0.85
MIOU_WARN_THRESHOLD = 0.70


def debug_train(cfg: DictConfig) -> None:
    """Run overfit-100 sanity test training.

    This trains on a small subset of data to verify the model can overfit,
    confirming the training pipeline is working correctly.
    """
    # Optimize for Tensor Cores
    torch.set_float32_matmul_precision("high")

    # 1. Seed
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Debug Training Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Create components with overfit mode enabled
    datamodule = create_datamodule(cfg, overfit_mode=True)
    model = create_model(cfg)
    class_weights = load_or_compute_class_weights(cfg, datamodule)
    segmentation_module = create_module(cfg, model, class_weights, overfit_mode=True)

    # 3. Minimal callbacks (no checkpointing/early stopping)
    callbacks = [
        PlotLoggerCallback(output_dir=cfg.callbacks.plots.output_dir),
    ]

    # 4. Trainer setup with overfit overrides
    trainer_kwargs = dict(cfg.trainer)
    trainer_kwargs.pop("export", None)

    # Apply overfit-specific overrides
    trainer_kwargs.update(
        {
            "max_steps": cfg.debug.overfit_max_steps,
            "max_epochs": -1,  # Use max_steps instead
            "limit_val_batches": 0,
            "num_sanity_val_steps": 0,
            "enable_checkpointing": False,
        }
    )
    logger.info("Overfit mode: trainer overrides applied.")

    # Simple CSV logger only
    loggers = [
        pl.loggers.CSVLogger("logs", name="landcover_overfit"),
    ]

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        logger=loggers,
    )

    # 5. Train
    logger.info("Starting overfit training...")
    trainer.fit(segmentation_module, datamodule=datamodule)

    # 6. Evaluate PASS/FAIL
    _evaluate_overfit_result(trainer)


def _evaluate_overfit_result(trainer: pl.Trainer) -> None:
    """Evaluate overfit training result and report PASS/FAIL/WARN status."""
    train_miou = trainer.callback_metrics.get("train/mIoU_epoch")
    if train_miou is None:
        train_miou = trainer.callback_metrics.get("train/mIoU")

    if train_miou is None:
        logger.warning("Could not find train/mIoU in metrics. PASS/FAIL skipped.")
        return

    train_miou = float(train_miou)

    if train_miou >= MIOU_PASS_THRESHOLD:
        status = "PASS"
        color = "\033[92m"  # Green
    elif train_miou >= MIOU_WARN_THRESHOLD:
        status = "WARN"
        color = "\033[93m"  # Yellow
    else:
        status = "FAIL"
        color = "\033[91m"  # Red

    reset = "\033[0m"
    print(f"\n{color}OVERFIT-100: {status} (train mIoU={train_miou:.4f}){reset}\n")

    if status == "FAIL":
        sys.exit(1)
