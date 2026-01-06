"""Shared utilities for training and debug training."""

import subprocess

import torch
from omegaconf import DictConfig

from landcover.datasets.datamodule import LandCoverDataModule
from landcover.models.segmentation import LandCoverSegmentationModule
from landcover.models.unet import UNetBaseline
from landcover.stats.class_weights import compute_class_weights
from utils.logging import get_logger

logger = get_logger(__name__)


def get_git_revision_hash() -> str:
    """Get the current git revision hash."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def create_datamodule(
    cfg: DictConfig, overfit_mode: bool = False
) -> LandCoverDataModule:
    """Create and return a LandCoverDataModule based on configuration."""
    return LandCoverDataModule(
        index_path=cfg.data.index_path,
        norm_stats_path=cfg.data.norm_stats_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        cloud_frac_max=cfg.data.cloud_frac_max,
        test_apply_cloud_filter=cfg.data.test_apply_cloud_filter,
        augment=cfg.data.augment if not overfit_mode else False,
        overfit_cfg=cfg.debug if overfit_mode else None,
        seed=cfg.seed,
    )


def create_model(cfg: DictConfig) -> UNetBaseline:
    """Create and return a UNetBaseline model based on configuration."""
    return UNetBaseline(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        base_channels=cfg.model.base_channels,
        num_stages=cfg.model.num_stages,
        norm_type=cfg.model.norm_type,
        upsample_type=cfg.model.upsample_type,
    )


def load_or_compute_class_weights(
    cfg: DictConfig, datamodule: LandCoverDataModule
) -> list | None:
    """Load class weights from file or compute them if needed."""
    import json
    import os

    module_cfg = cfg.module
    if module_cfg.loss.name not in ["weighted_ce", "weighted_ce_dice"]:
        return None

    weights_path = module_cfg.loss.get("class_weights_path")

    if module_cfg.loss.get("compute_if_missing") and (
        not weights_path or not os.path.exists(weights_path)
    ):
        logger.info("Class weights missing or not specified. Computing...")
        datamodule.prepare_data()
        weights_path = weights_path or "data/stats/class_weights.json"
        return compute_class_weights(
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
        logger.info(f"Loading class weights from {weights_path}")
        with open(weights_path) as f:
            weights_data = json.load(f)
            return weights_data["class_weights"]

    return None


def create_module(
    cfg: DictConfig,
    model: UNetBaseline,
    class_weights: list | None = None,
    overfit_mode: bool = False,
) -> LandCoverSegmentationModule:
    """Create and return a LandCoverSegmentationModule."""
    module_cfg = cfg.module

    # Determine scheduler and LR based on mode
    if overfit_mode:
        scheduler_type = "none"
        lr = cfg.debug.overfit_lr
        logger.info(f"Overfit mode: forcing constant LR={lr} (scheduler=none)")
    else:
        scheduler_type = module_cfg.scheduler.name
        lr = module_cfg.lr

    return LandCoverSegmentationModule(
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


def export_model(
    cfg: DictConfig,
    trainer,
    model: UNetBaseline,
) -> None:
    """Export the best model to ONNX format with metadata."""
    import os
    import shutil

    from omegaconf import OmegaConf

    export_cfg = dict(cfg.trainer).get("export", {})
    artifacts_dir = export_cfg.get("artifacts_dir", "artifacts")
    onnx_filename = export_cfg.get("onnx_filename", "model.onnx")

    logger.info(f"Exporting model to {artifacts_dir}...")
    best_model_path = trainer.checkpoint_callback.best_model_path

    if not best_model_path:
        logger.warning("No best model checkpoint found. Skipping export.")
        return

    logger.info(f"Loading best model from {best_model_path} for export.")
    export_module = LandCoverSegmentationModule.load_from_checkpoint(
        best_model_path, model=model
    ).cpu()
    export_module.eval()

    os.makedirs(artifacts_dir, exist_ok=True)

    # Export to ONNX
    try:
        onnx_path = os.path.join(artifacts_dir, onnx_filename)
        dummy_input = torch.randn(1, cfg.model.in_channels, 256, 256).cpu()

        with torch.no_grad():
            export_module.to_onnx(
                onnx_path,
                input_sample=dummy_input,
                export_params=True,
                opset_version=17,
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
