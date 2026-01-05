import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from landcover.callbacks.plots import PlotLoggerCallback
from landcover.datasets.datamodule import LandCoverDataModule
from landcover.models.segmentation import LandCoverSegmentationModule
from landcover.models.unet import UNetBaseline
from landcover.stats.class_weights import compute_class_weights
from utils.logging import get_logger

logger = get_logger(__name__)


def get_git_revision_hash() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    # Optimize for Tensor Cores - moved from file level
    torch.set_float32_matmul_precision("high")

    # 1. Seed
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Instantiate DataModule
    datamodule = LandCoverDataModule(
        index_path=cfg.data.index_path,
        norm_stats_path=cfg.data.norm_stats_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        cloud_frac_max=cfg.data.cloud_frac_max,
        test_apply_cloud_filter=cfg.data.test_apply_cloud_filter,
        augment=cfg.data.augment,
        overfit_cfg=cfg.debug,
        seed=cfg.seed,
    )

    # 3. Instantiate Model
    model = UNetBaseline(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        base_channels=cfg.model.base_channels,
        num_stages=cfg.model.num_stages,
        norm_type=cfg.model.norm_type,
        upsample_type=cfg.model.upsample_type,
    )

    # 4. Instantiate LightningModule
    module_cfg = cfg.module

    # Load class weights if path is provided and loss is weighted
    class_weights = None
    if module_cfg.loss.name in ["weighted_ce", "weighted_ce_dice"]:
        weights_path = module_cfg.loss.get("class_weights_path")

        import os

        if module_cfg.loss.get("compute_if_missing") and (
            not weights_path or not os.path.exists(weights_path)
        ):
            logger.info("Class weights missing or not specified. Computing...")
            # Ensure data is present!
            datamodule.prepare_data()
            # If path not specified, use a default
            weights_path = weights_path or "data/stats/class_weights.json"
            class_weights = compute_class_weights(
                index_path=cfg.data.index_path,
                output_path=weights_path,
                split_name="train",
                cloud_frac_max=cfg.data.cloud_frac_max,
                num_classes=cfg.model.num_classes,
                ignore_index=module_cfg.ignore_index,
                min_weight=module_cfg.loss.get("min_weight", 0.25),
                max_weight=module_cfg.loss.get("max_weight", 4.0),
            )
        elif weights_path and os.path.exists(weights_path):
            import json

            logger.info(f"Loading class weights from {weights_path}")
            with open(weights_path) as f:
                weights_data = json.load(f)
                class_weights = weights_data["class_weights"]

    # Overfit Mode Overrides
    scheduler_type = module_cfg.scheduler.name
    lr = module_cfg.lr
    if cfg.debug.overfit_100:
        logger.info(
            f"Overfit mode: forcing constant LR={cfg.debug.overfit_lr} (scheduler=none)"
        )
        scheduler_type = "none"
        lr = cfg.debug.overfit_lr

    segmentation_module = LandCoverSegmentationModule(
        model=model,
        num_classes=cfg.model.num_classes,
        ignore_index=module_cfg.ignore_index,
        lr=lr,
        weight_decay=module_cfg.weight_decay,
        loss_type=module_cfg.loss.name,
        dice_weight=module_cfg.loss.get("dice_weight", 0.2),
        class_weights=class_weights,
        scheduler_type=scheduler_type,
        scheduler_step_size=module_cfg.scheduler.get("step_size", 20),
        scheduler_gamma=module_cfg.scheduler.get("gamma", 0.1),
        T_max=module_cfg.scheduler.get("T_max", cfg.trainer.max_epochs),
    )

    # 5. Callbacks
    callbacks = [
        PlotLoggerCallback(output_dir=cfg.callbacks.plots.output_dir),
    ]

    # Only add EarlyStopping and ModelCheckpoint if not in overfit mode
    if not cfg.debug.overfit_100:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.callbacks.early_stopping.monitor,
                patience=cfg.callbacks.early_stopping.patience,
                mode=cfg.callbacks.early_stopping.mode,
            )
        )
        callbacks.append(
            ModelCheckpoint(
                monitor=cfg.callbacks.model_checkpoint.monitor,
                save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
                mode=cfg.callbacks.model_checkpoint.mode,
                filename=cfg.callbacks.model_checkpoint.filename,
            )
        )

    # 6. Trainer setup
    trainer_kwargs = dict(cfg.trainer)
    # Remove 'export' key from trainer_kwargs as it's not a Pytorch Lightning Trainer param
    export_cfg = trainer_kwargs.pop("export", {})

    if cfg.debug.overfit_100:
        logger.info("Overfit mode: applying trainer overrides.")
        trainer_kwargs.update(
            {
                "max_steps": cfg.debug.overfit_max_steps,
                "max_epochs": -1,  # Use max_steps
                "limit_val_batches": 0,
                "num_sanity_val_steps": 0,
                "enable_checkpointing": False,
            }
        )

    # Loggers
    loggers = [
        pl.loggers.CSVLogger("logs", name="landcover_segmentation"),
    ]
    if cfg.logging.mlflow.enabled and not cfg.debug.overfit_100:
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

    # 7. Run Train & Test
    logger.info("Starting training...")
    trainer.fit(segmentation_module, datamodule=datamodule)

    # PASS/FAIL Reporting for Overfit Mode
    if cfg.debug.overfit_100:
        train_miou = trainer.callback_metrics.get("train/mIoU_epoch")
        if train_miou is None:
            train_miou = trainer.callback_metrics.get("train/mIoU")

        if train_miou is not None:
            train_miou = float(train_miou)
            if train_miou >= 0.85:
                status = "PASS"
                color = "\033[92m"  # Green
            elif train_miou >= 0.70:
                status = "WARN"
                color = "\033[93m"  # Yellow
            else:
                status = "FAIL"
                color = "\033[91m"  # Red

            reset = "\033[0m"
            print(
                f"\n{color}OVERFIT-100: {status} (train mIoU={train_miou:.4f}){reset}\n"
            )

            if status == "FAIL":
                import sys

                sys.exit(1)
        else:
            logger.warning("Could not find train/mIoU in metrics. PASS/FAIL skipped.")

        # Skip export and testing in overfit mode
        return

    # 8. Model Export
    artifacts_dir = export_cfg.get("artifacts_dir", "artifacts")
    onnx_filename = export_cfg.get("onnx_filename", "model.onnx")

    logger.info(f"Exporting model to {artifacts_dir}...")
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Loading best model from {best_model_path} for export.")
        # Load best model and move to CPU for export
        export_module = LandCoverSegmentationModule.load_from_checkpoint(
            best_model_path, model=model
        ).cpu()
        export_module.eval()

        # Create export directory
        import os
        import shutil

        os.makedirs(artifacts_dir, exist_ok=True)

        # Export to ONNX - force CPU input
        try:
            onnx_path = os.path.join(artifacts_dir, onnx_filename)
            dummy_input = torch.randn(1, cfg.model.in_channels, 256, 256).cpu()

            with torch.no_grad():
                export_module.to_onnx(
                    onnx_path,
                    input_sample=dummy_input,
                    export_params=True,
                    opset_version=17,  # Safer for newer torch
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                )
            logger.info(f"Model exported to {onnx_path}")
        except Exception as e:
            logger.warning(f"Failed to export to ONNX: {e}")

        # Copy norm stats
        shutil.copy(
            cfg.data.norm_stats_path, os.path.join(artifacts_dir, "norm_stats.json")
        )

        # Save config
        with open(os.path.join(artifacts_dir, "model_config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        logger.info("Export metadata saved.")
    else:
        logger.warning("No best model checkpoint found. Skipping export.")

    logger.info("Starting testing...")
    # Test using the best checkpoint explicitly
    # This will use the data loaders from datamodule (iid and ood if present)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    train()
