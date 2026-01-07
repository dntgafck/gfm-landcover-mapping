"""TensorRT conversion functions for ONNX models."""

from pathlib import Path

import torch
from omegaconf import DictConfig

from landcover.models.segmentation import LandCoverSegmentationModule
from utils.logging import get_logger

logger = get_logger(__name__)


def _export_to_tensorrt(
    export_module: LandCoverSegmentationModule,
    onnx_path: Path,
    artifacts_dir: Path,
    tensorrt_cfg: dict,
    cfg: DictConfig,
) -> None:
    """Convert ONNX model to TensorRT engine.

    Args:
        export_module: The PyTorch Lightning module
        onnx_path: Path to the ONNX model
        artifacts_dir: Directory to save the TensorRT engine
        tensorrt_cfg: TensorRT configuration from Hydra config
        cfg: Full Hydra configuration
    """
    try:
        # Try torch-tensorrt first (PyTorch Lightning integration)
        import torch_tensorrt

        engine_filename = tensorrt_cfg.get("engine_filename", "model.engine")
        engine_path = artifacts_dir / engine_filename
        precision = tensorrt_cfg.get("precision", "fp16")
        max_batch_size = tensorrt_cfg.get("max_batch_size", 1)
        workspace_size = tensorrt_cfg.get("workspace_size", 1073741824)  # 1GB

        logger.info("Converting to TensorRT using torch-tensorrt...")

        # Prepare input sample
        input_sample = torch.randn(1, cfg.model.in_channels, 256, 256).cpu()

        # Compile with TensorRT
        trt_model = torch_tensorrt.compile(
            export_module,
            inputs=[input_sample],
            enabled_precisions=(
                {torch.float, torch.half} if precision == "fp16" else {torch.float}
            ),
            max_batch_size=max_batch_size,
            workspace_size=workspace_size,
            use_fp16=(precision == "fp16"),
            debug_mode=tensorrt_cfg.get("verbose", False),
        )

        # Save the TensorRT engine
        torch_tensorrt.save(trt_model, str(engine_path))

        logger.info(f"TensorRT engine saved to {engine_path}")

    except ImportError:
        logger.warning(
            "torch-tensorrt not available, trying ONNX-TensorRT conversion..."
        )
        _export_to_tensorrt_onnx(onnx_path, artifacts_dir, tensorrt_cfg)
    except Exception as e:
        logger.warning(f"torch-tensorrt conversion failed: {e}")
        logger.info("Falling back to ONNX-TensorRT conversion...")
        _export_to_tensorrt_onnx(onnx_path, artifacts_dir, tensorrt_cfg)


def _export_to_tensorrt_onnx(
    onnx_path: Path, artifacts_dir: Path, tensorrt_cfg: dict
) -> None:
    """Convert ONNX model to TensorRT engine using onnx-tensorrt.

    Args:
        onnx_path: Path to the ONNX model
        artifacts_dir: Directory to save the TensorRT engine
        tensorrt_cfg: TensorRT configuration from Hydra config
    """
    try:
        import onnx
        import tensorrt as trt

        engine_filename = tensorrt_cfg.get("engine_filename", "model.engine")
        engine_path = artifacts_dir / engine_filename

        logger.info("Converting ONNX to TensorRT engine...")

        # Create TensorRT logger
        logger_t = trt.Logger(trt.Logger.WARNING)

        # Create builder and config
        builder = trt.Builder(logger_t)
        config = builder.create_builder_config()
        config.max_workspace_size = tensorrt_cfg.get(
            "workspace_size", 1073741824
        )  # 1GB

        # Set precision
        precision = tensorrt_cfg.get("precision", "fp16")
        if precision == "fp16":
            config.set_flag(trt.BuilderConfig.FP16)
        elif precision == "int8":
            config.set_flag(trt.BuilderConfig.INT8)

        # Build engine
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger_t)

        # Load ONNX model
        with open(onnx_path, "rb") as f:
            onnx_model = onnx.load_from_string(f.read())
            parser.parse(onnx_model.SerializeToString())

        # Build engine
        engine = builder.build_engine(network, config)

        if engine:
            # Save engine
            with open(engine_path, "wb") as f:
                f.write(engine.serialize())
            logger.info(f"TensorRT engine saved to {engine_path}")
        else:
            logger.error("Failed to build TensorRT engine")

    except ImportError:
        logger.warning(
            "ONNX-TensorRT conversion failed: tensorrt or onnx packages not available"
        )
    except Exception as e:
        logger.warning(f"ONNX-TensorRT conversion failed: {e}")
