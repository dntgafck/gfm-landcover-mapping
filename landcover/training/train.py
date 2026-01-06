"""Regular training entrypoint."""

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from landcover.callbacks.plots import PlotLoggerCallback
from landcover.training.common import (
    create_datamodule,
    create_model,
    create_module,
    export_model,
    get_git_revision_hash,
    load_or_compute_class_weights,
)
from utils.logging import get_logger

logger = get_logger(__name__)


def train(cfg: DictConfig) -> None:
    """Run regular training pipeline with checkpointing, logging, and export."""
    # Optimize for Tensor Cores
    torch.set_float32_matmul_precision("high")

    # 1. Seed
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Create components
    datamodule = create_datamodule(cfg, overfit_mode=False)
    model = create_model(cfg)
    class_weights = load_or_compute_class_weights(cfg, datamodule)
    segmentation_module = create_module(cfg, model, class_weights, overfit_mode=False)

    # 3. Callbacks
    callbacks = [
        PlotLoggerCallback(output_dir=cfg.callbacks.plots.output_dir),
        EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
            mode=cfg.callbacks.early_stopping.mode,
        ),
        ModelCheckpoint(
            monitor=cfg.callbacks.model_checkpoint.monitor,
            save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
            mode=cfg.callbacks.model_checkpoint.mode,
            filename=cfg.callbacks.model_checkpoint.filename,
        ),
    ]

    # 4. Trainer setup
    trainer_kwargs = dict(cfg.trainer)
    # Remove 'export' key - not a Pytorch Lightning Trainer param
    trainer_kwargs.pop("export", None)

    # Loggers
    loggers = [
        pl.loggers.CSVLogger("logs", name="landcover_segmentation"),
    ]
    if cfg.logging.mlflow.enabled:
        loggers.append(
            pl.loggers.MLFlowLogger(
                tracking_uri=cfg.logging.mlflow.tracking_uri,
                experiment_name=cfg.logging.mlflow.experiment_name,
                run_name=cfg.logging.mlflow.run_name,
                tags={"git_commit": get_git_revision_hash()},
            )
        )

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        logger=loggers,
    )

    # 5. Train
    logger.info("Starting training...")
    trainer.fit(segmentation_module, datamodule=datamodule)

    # 6. Export best model
    export_model(cfg, trainer, model)

    # 7. Test
    logger.info("Starting testing...")
    trainer.test(ckpt_path="best", datamodule=datamodule)
