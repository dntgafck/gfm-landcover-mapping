"""Training entrypoint for both regular and debug training."""

from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger

from landcover.callbacks.plots import PlotLoggerCallback
from landcover.training.common import (
    create_datamodule,
    create_model,
    create_module,
    export_model,
    get_git_revision_hash,
    load_or_compute_class_weights,
    mirror_to_mlflow,
)
from utils.lineage import compute_lineage_metadata, get_lineage_tags, save_lineage
from utils.logging import get_logger
from utils.run_id import get_run_paths

logger = get_logger(__name__)


def train(
    cfg: DictConfig, run_id: str, run_dir: Path, debug_mode: bool = False
) -> None:
    """Run training pipeline with checkpointing, logging, and optional export.

    Args:
        cfg: Hydra configuration object
        run_id: Unique run identifier
        run_dir: Path to the run directory for all outputs
        debug_mode: If True, use reduced dataset from cfg.debug settings
    """
    mode_str = "debug" if debug_mode else "train"

    # Get standard paths within run directory
    run_paths = get_run_paths(run_dir)

    # Optimize for Tensor Cores
    torch.set_float32_matmul_precision("high")

    # 1. Seed
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"{mode_str.capitalize()} Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Create components (debug_mode controls dataset subsetting)
    datamodule = create_datamodule(cfg, debug_mode=debug_mode)
    model = create_model(cfg)
    class_weights = load_or_compute_class_weights(cfg, datamodule)
    segmentation_module = create_module(cfg, model, class_weights)

    # 3. Callbacks - all paths relative to run_dir
    callbacks = [
        PlotLoggerCallback(output_dir=run_paths["plots"]),
        EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
            mode=cfg.callbacks.early_stopping.mode,
        ),
        ModelCheckpoint(
            dirpath=run_paths["checkpoints"],
            monitor=cfg.callbacks.model_checkpoint.monitor,
            save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
            save_last=cfg.callbacks.model_checkpoint.get("save_last", True),
            mode=cfg.callbacks.model_checkpoint.mode,
            filename=cfg.callbacks.model_checkpoint.filename,
        ),
    ]

    # Add LearningRateMonitor for non-debug training
    if not debug_mode:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    # 4. Trainer setup
    trainer_kwargs = dict(cfg.trainer)
    # Remove non-Trainer params
    trainer_kwargs.pop("export", None)
    trainer_kwargs.pop("runs_root", None)

    # 5. Loggers - CSVLogger writes to run_dir/logs/
    loggers = [
        pl.loggers.CSVLogger(
            save_dir=str(run_paths["logs"]),
            name=None,
            version="",
        ),
    ]

    # MLflow logger (optional)
    mlflow_logger = None
    if cfg.logging.mlflow.enabled:
        tags = {
            "git_commit": get_git_revision_hash(),
            "run_dir": str(run_dir),
            "mode": mode_str,
        }
        mlflow_logger = MLFlowLogger(
            tracking_uri=cfg.logging.mlflow.tracking_uri,
            experiment_name=cfg.logging.mlflow.experiment_name,
            run_name=f"{cfg.logging.mlflow.run_name or run_id}",
            tags=tags,
        )
        loggers.append(mlflow_logger)

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        logger=loggers,
    )

    # Log full config as hyperparameters for reproducibility
    hparams = OmegaConf.to_container(cfg, resolve=True)
    trainer.logger.log_hyperparams(hparams)

    # 6. Compute and save lineage metadata
    lineage = compute_lineage_metadata(cfg, cfg.seed)
    lineage["mode"] = mode_str
    save_lineage(lineage, run_paths["lineage"])

    # Add lineage tags to MLflow if enabled
    if mlflow_logger is not None:
        lineage_tags = get_lineage_tags(lineage)
        lineage_tags["mode"] = mode_str
        for key, value in lineage_tags.items():
            mlflow_logger.experiment.set_tag(mlflow_logger.run_id, key, value)

    # 7. Train
    logger.info(f"Starting {mode_str} training...")
    trainer.fit(segmentation_module, datamodule=datamodule)

    # 8. Export model (skip in debug mode)
    export_cfg = dict(cfg.trainer).get("export", {})
    if export_cfg.get("enabled", True):
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        export_model(cfg, model, checkpoint_path, run_paths["export"])
    else:
        logger.info("Model export disabled in config. Skipping.")

    # 9. Mirror artifacts to MLflow if enabled
    if mlflow_logger is not None:
        mirror_to_mlflow(cfg, mlflow_logger, run_paths)

    # 10. Test (skip in debug mode - no test dataset)
    if not debug_mode:
        logger.info("Starting testing...")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    # Log final metrics
    final_metrics = trainer.callback_metrics
    logger.info(f"{mode_str.capitalize()} training complete. Final metrics:")
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {float(value):.4f}")

    logger.info(f"Training complete. All outputs saved to: {run_dir}")


def debug_train(cfg: DictConfig, run_id: str, run_dir: Path) -> None:
    """Run debug training on a reduced dataset subset.

    This is an alias for train(debug_mode=True).
    """
    train(cfg, run_id, run_dir, debug_mode=True)
