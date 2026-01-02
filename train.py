import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from landcover.datasets.datamodule import LandCoverDataModule
from landcover.models.segmentation import LandCoverSegmentationModule
from landcover.models.unet import UNetBaseline
from utils.logging import get_logger

logger = get_logger(__name__)

# Optimize for Tensor Cores
torch.set_float32_matmul_precision("high")


@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
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
    segmentation_module = LandCoverSegmentationModule(
        model=model,
        num_classes=cfg.model.num_classes,
        ignore_index=cfg.module.ignore_index,
        lr=cfg.module.lr,
        weight_decay=cfg.module.weight_decay,
        scheduler_step_size=cfg.module.scheduler_step_size,
        scheduler_gamma=cfg.module.scheduler_gamma,
    )

    # 5. Callbacks
    callbacks = [
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

    # 6. Trainer setup
    # Handle debug run overrides
    trainer_kwargs = dict(cfg.trainer)
    if cfg.debug:
        logger.info("Debug mode enabled: using minimal settings.")
        trainer_kwargs.update(
            {
                "limit_train_batches": 5,
                "limit_val_batches": 2,
                "limit_test_batches": 2,
                "max_epochs": 1,
                "num_sanity_val_steps": 0,
            }
        )
        # Override num_workers for local debug
        datamodule.num_workers = 0

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        logger=pl.loggers.CSVLogger("logs", name="landcover_segmentation"),
    )

    # 7. Run Train & Test
    logger.info("Starting training...")
    trainer.fit(segmentation_module, datamodule=datamodule)

    # 8. Model Export
    logger.info(f"Exporting model to {cfg.export_dir}...")
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Loading best model from {best_model_path}")
        # Load best model
        export_module = LandCoverSegmentationModule.load_from_checkpoint(
            best_model_path, model=model
        )
        export_module.eval()

        # Create export directory
        import os
        import shutil

        os.makedirs(cfg.export_dir, exist_ok=True)

        # Export to ONNX
        try:
            onnx_path = os.path.join(cfg.export_dir, cfg.onnx_filename)
            dummy_input = torch.randn(1, cfg.model.in_channels, 256, 256)
            export_module.to_onnx(onnx_path, input_sample=dummy_input, export_params=True)
            logger.info(f"Model exported to {onnx_path}")
        except Exception as e:
            logger.warning(f"Failed to export to ONNX: {e}")
            logger.info(
                "The model data (checkpoint, stats, config) is still saved in the export directory."
            )

        # Copy norm stats
        shutil.copy(cfg.data.norm_stats_path, os.path.join(cfg.export_dir, "norm_stats.json"))

        # Save config
        with open(os.path.join(cfg.export_dir, "model_config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        logger.info("Export metadata saved.")
    else:
        logger.warning("No best model checkpoint found. Skipping export.")

    logger.info("Starting testing...")
    # Test using the best checkpoint
    trainer.test(segmentation_module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    train()
