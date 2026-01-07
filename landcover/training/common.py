"""Shared utilities for training and debug training."""

import subprocess
from pathlib import Path

import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger

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


def create_datamodule(cfg: DictConfig, debug_mode: bool = False) -> LandCoverDataModule:
    """Create and return a LandCoverDataModule based on configuration."""
    return LandCoverDataModule(
        index_path=cfg.data.index_path,
        norm_stats_path=cfg.data.norm_stats_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        cloud_frac_max=cfg.data.cloud_frac_max,
        test_apply_cloud_filter=cfg.data.test_apply_cloud_filter,
        augment=cfg.data.augment if not debug_mode else False,
        overfit_cfg=cfg.debug if debug_mode else None,
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
    from pathlib import Path

    module_cfg = cfg.module
    if module_cfg.loss.name not in ["weighted_ce", "weighted_ce_dice"]:
        return None

    weights_path = module_cfg.loss.get("class_weights_path")

    if module_cfg.loss.get("compute_if_missing") and (
        not weights_path or not Path(weights_path).exists()
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
    elif weights_path and Path(weights_path).exists():
        logger.info(f"Loading class weights from {weights_path}")
        with open(weights_path) as f:
            weights_data = json.load(f)
            return weights_data["class_weights"]

    return None


def create_module(
    cfg: DictConfig,
    model: UNetBaseline,
    class_weights: list | None = None,
) -> LandCoverSegmentationModule:
    """Create and return a LandCoverSegmentationModule."""
    module_cfg = cfg.module

    return LandCoverSegmentationModule(
        model=model,
        num_classes=cfg.model.num_classes,
        ignore_index=module_cfg.ignore_index,
        lr=module_cfg.lr,
        weight_decay=module_cfg.weight_decay,
        loss_type=module_cfg.loss.name,
        dice_weight=module_cfg.loss.get("dice_weight", 0.2),
        class_weights=class_weights,
        scheduler_type=module_cfg.scheduler.name,
        scheduler_step_size=module_cfg.scheduler.get("step_size", 20),
        scheduler_gamma=module_cfg.scheduler.get("gamma", 0.1),
        T_max=module_cfg.scheduler.get("T_max", cfg.trainer.max_epochs),
    )


def mirror_to_mlflow(
    cfg: DictConfig,
    mlflow_logger: MLFlowLogger,
    run_paths: dict[str, Path],
) -> None:
    """Mirror local artifacts to MLflow.

    Args:
        cfg: Configuration object
        mlflow_logger: MLflow logger instance
        run_paths: Dictionary of run directory paths
    """
    mlflow_cfg = cfg.logging.mlflow

    # Mirror config
    if mlflow_cfg.get("mirror_config", True):
        config_path = run_paths["config_yaml"]
        if config_path.exists():
            mlflow_logger.experiment.log_artifact(
                run_id=mlflow_logger.run_id,
                local_path=str(config_path),
                artifact_path="config",
            )
            logger.info("Mirrored config.yaml to MLflow")

    # Mirror plots
    if mlflow_cfg.get("mirror_plots", True):
        plots_dir = run_paths["plots"]
        if plots_dir.exists():
            for plot_file in plots_dir.glob("*.png"):
                mlflow_logger.experiment.log_artifact(
                    run_id=mlflow_logger.run_id,
                    local_path=str(plot_file),
                    artifact_path="plots",
                )
            logger.info("Mirrored plots to MLflow")

    # Mirror best checkpoint (optional, can be large)
    if mlflow_cfg.get("mirror_checkpoint", False):
        best_ckpt = run_paths["checkpoints"] / "best.ckpt"
        if best_ckpt.exists():
            mlflow_logger.experiment.log_artifact(
                run_id=mlflow_logger.run_id,
                local_path=str(best_ckpt),
                artifact_path="checkpoints",
            )
            logger.info("Mirrored best.ckpt to MLflow")

    # Mirror lineage
    lineage_path = run_paths["lineage"]
    if lineage_path.exists():
        mlflow_logger.experiment.log_artifact(
            run_id=mlflow_logger.run_id, local_path=str(lineage_path)
        )
        logger.info("Mirrored lineage.json to MLflow")


def resolve_checkpoint_path(run_dir: Path, checkpoint_type: str = "best") -> Path:
    """Resolve checkpoint path based on run_dir and checkpoint type.

    Args:
        run_dir: Path to the run directory
        checkpoint_type: Type of checkpoint to resolve ('best' or 'last')

    Returns:
        Path to the checkpoint file

    Raises:
        ValueError: If checkpoint_type is not 'best' or 'last'
        FileNotFoundError: If the checkpoint file does not exist
    """
    checkpoints_dir = Path(run_dir) / "checkpoints"

    if checkpoint_type == "best":
        checkpoint_path = checkpoints_dir / "best.ckpt"
    elif checkpoint_type == "last":
        checkpoint_path = checkpoints_dir / "last.ckpt"
    else:
        raise ValueError(
            f"Unknown checkpoint type: {checkpoint_type}. Use 'best' or 'last'."
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def export_model(
    cfg: DictConfig,
    model: UNetBaseline,
    checkpoint_path: "str | Path",
    export_dir: Path,
    norm_stats_path: "str | Path | None" = None,
) -> None:
    """Export a model to ONNX format with metadata.

    Args:
        cfg: Hydra configuration object
        model: The UNet model architecture (weights will be loaded from checkpoint)
        checkpoint_path: Path to the checkpoint file to export
        export_dir: Directory to save exports
        norm_stats_path: Path to norm_stats.json (default: from cfg.data.norm_stats_path)
    """
    import shutil
    from pathlib import Path

    from omegaconf import OmegaConf

    export_cfg = dict(cfg.trainer).get("export", {})

    artifacts_dir = Path(export_dir)
    onnx_filename = export_cfg.get("onnx_filename", "model.onnx")
    checkpoint_path = Path(checkpoint_path)

    logger.info(f"Exporting model to {artifacts_dir}...")

    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}. Skipping export.")
        return

    logger.info(f"Loading model from {checkpoint_path} for export.")
    export_module = LandCoverSegmentationModule.load_from_checkpoint(
        checkpoint_path, model=model
    ).cpu()
    export_module.eval()

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    try:
        onnx_path = artifacts_dir / onnx_filename
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

    # Copy norm stats (use provided path or fall back to config)
    stats_source = (
        Path(norm_stats_path) if norm_stats_path else Path(cfg.data.norm_stats_path)
    )
    shutil.copy(stats_source, artifacts_dir / "norm_stats.json")

    # Save inference config (subset of full config relevant for inference)
    inference_config = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "data": {
            "norm_stats_path": "norm_stats.json",  # Relative to export dir
            "in_channels": cfg.model.in_channels,
            "num_classes": cfg.model.num_classes,
        },
    }
    with open(artifacts_dir / "inference_config.yaml", "w") as f:
        import yaml

        yaml.dump(inference_config, f, default_flow_style=False)

    logger.info(f"Export complete. Files saved to {artifacts_dir}")
