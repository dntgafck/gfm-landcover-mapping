"""FastAPI application for inference server."""

import io
import json
import zipfile
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from omegaconf import DictConfig

from inference.catalog import TileCatalog
from inference.dvc_sync import sync_inference_data
from inference.engine import InferenceEngine, compute_histogram
from inference.model_loader import load_model, load_norm_stats
from inference.render import (
    image_to_bytes,
    render_class_map,
    render_side_by_side,
)
from inference.schemas import (
    CountryListResponse,
    HealthResponse,
    InferRequest,
    InferStats,
    ModelInfoResponse,
    TileListResponse,
    TileMetaResponse,
)
from inference.utils import TimingAccumulator, generate_inference_id
from utils.logging import get_logger

logger = get_logger(__name__)

# Global state (initialized at startup)
_state: dict[str, Any] = {
    "config": None,
    "session": None,
    "model_path": None,
    "catalog": None,
    "engine": None,
    "mean": None,
    "std": None,
    "initialized": False,
}


def initialize_server(cfg: DictConfig) -> None:
    """Initialize server components.

    Args:
        cfg: Hydra configuration
    """
    logger.info("Initializing inference server...")

    # Store config
    _state["config"] = cfg

    # Sync inference data via DVC
    logger.info("Syncing inference data...")
    sync_inference_data(cfg)

    # Load ONNX model
    logger.info("Loading ONNX model...")
    session, model_path, artifacts_dir = load_model(cfg)
    _state["session"] = session
    _state["model_path"] = model_path

    # Load normalization stats (from MLflow artifacts or local path)
    logger.info("Loading normalization stats...")
    if artifacts_dir is not None:
        # MLflow source: use downloaded artifacts
        norm_stats_path = artifacts_dir / "norm_stats.json"
    elif cfg.model.source == "local":
        # Local source: look next to ONNX file or use config path
        sibling_path = model_path.parent / "norm_stats.json"
        if sibling_path.exists():
            norm_stats_path = sibling_path
        else:
            norm_stats_path = cfg.model.norm_stats_path
    else:
        norm_stats_path = cfg.model.norm_stats_path

    mean, std = load_norm_stats(norm_stats_path)
    _state["mean"] = mean
    _state["std"] = std

    # Build tile catalog
    logger.info("Building tile catalog...")
    catalog = TileCatalog(cfg)
    _state["catalog"] = catalog

    # Create inference engine
    logger.info("Creating inference engine...")
    engine = InferenceEngine(session, catalog, mean, std, cfg)
    _state["engine"] = engine

    _state["initialized"] = True
    logger.info(
        f"Server initialized: {catalog.num_countries} countries, "
        f"{catalog.num_tiles} tiles"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""
    # Startup: initialization happens in create_app before uvicorn starts
    yield
    # Shutdown
    logger.info("Shutting down inference server...")


def create_app(cfg: DictConfig | None = None) -> FastAPI:
    """Create FastAPI application.

    Args:
        cfg: Optional Hydra configuration (if None, must call initialize_server later)

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Land Cover Inference Server",
        description="ONNX-based inference server for land cover segmentation",
        version="1.0.0",
        lifespan=lifespan,
    )

    if cfg is not None:
        initialize_server(cfg)

    # Register routes
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if _state["initialized"] else "initializing",
            model_loaded=_state["session"] is not None,
            data_ready=_state["catalog"] is not None,
        )

    @app.get("/model", response_model=ModelInfoResponse)
    async def model_info():
        """Get model information."""
        if not _state["initialized"]:
            raise HTTPException(status_code=503, detail="Server not initialized")

        cfg = _state["config"]
        session = _state["session"]

        return ModelInfoResponse(
            source=cfg.model.source,
            model_uri_or_path=str(_state["model_path"]),
            providers=list(cfg.runtime.providers),
            input_names=[i.name for i in session.get_inputs()],
            output_names=[o.name for o in session.get_outputs()],
        )

    @app.get("/countries", response_model=CountryListResponse)
    async def list_countries():
        """List available countries."""
        if not _state["initialized"]:
            raise HTTPException(status_code=503, detail="Server not initialized")

        catalog: TileCatalog = _state["catalog"]
        return CountryListResponse(countries=catalog.get_countries())

    # Define Query parameters at module level to avoid B008
    CountryQuery = Query(..., description="Country code")

    @app.get("/tiles", response_model=TileListResponse)
    async def list_tiles(country: str = CountryQuery):
        """List tiles for a country."""
        if not _state["initialized"]:
            raise HTTPException(status_code=503, detail="Server not initialized")

        catalog: TileCatalog = _state["catalog"]
        try:
            tiles = catalog.get_tiles(country)
            return TileListResponse(country=country, tiles=tiles)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Country not found: {country}")

    @app.get("/tiles/{country}/{tile_id}/meta", response_model=TileMetaResponse)
    async def tile_metadata(country: str, tile_id: str):
        """Get metadata for a specific tile."""
        if not _state["initialized"]:
            raise HTTPException(status_code=503, detail="Server not initialized")

        catalog: TileCatalog = _state["catalog"]
        try:
            meta = catalog.get_tile_meta(country, tile_id)
            return TileMetaResponse(**meta)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/infer")
    async def run_inference(request: InferRequest):
        """Run inference on a tile window.

        Returns a ZIP file containing:
        - pred.png: Prediction visualization
        - label.png: Ground truth visualization (if available)
        - compare.png: Side-by-side comparison (if label available)
        - stats.json: Inference statistics
        """
        if not _state["initialized"]:
            raise HTTPException(status_code=503, detail="Server not initialized")

        engine: InferenceEngine = _state["engine"]
        timings = TimingAccumulator()

        try:
            # Run inference
            with timings.time("total"):
                predictions, labels, window_info = engine.run_inference(
                    request, timings
                )

            # Generate inference ID
            inference_id = generate_inference_id()

            # Compute histograms
            with timings.time("histograms"):
                pred_histogram = compute_histogram(predictions)
                label_histogram = (
                    compute_histogram(labels) if labels is not None else None
                )

            # Create stats object
            stats = InferStats(
                inference_id=inference_id,
                timings=timings.get_timings(),
                window_info=window_info,
                pred_histogram=pred_histogram,
                label_histogram=label_histogram,
            )

            # Create ZIP file in memory
            with timings.time("rendering"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    # Add prediction PNG
                    if request.include_pred:
                        pred_img = render_class_map(predictions)
                        zip_file.writestr("pred.png", image_to_bytes(pred_img))

                    # Add label PNG (if available)
                    if request.include_label and labels is not None:
                        label_img = render_class_map(labels)
                        zip_file.writestr("label.png", image_to_bytes(label_img))

                    # Add comparison PNG (if both available)
                    if request.include_compare and labels is not None:
                        compare_img = render_side_by_side(predictions, labels)
                        zip_file.writestr("compare.png", image_to_bytes(compare_img))

                    # Add stats JSON
                    stats_json = stats.model_dump_json(indent=2)
                    zip_file.writestr("stats.json", stats_json)

            zip_buffer.seek(0)

            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{inference_id}.zip"'
                },
            )

        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception(f"Inference error: {e}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    @app.get("/infer/stats/{inference_id}")
    async def get_stats(inference_id: str):
        """Get stats for a previous inference (stub - returns 404).

        Note: Stats are returned inline with inference results.
        This endpoint is a placeholder for future result caching.
        """
        raise HTTPException(
            status_code=404,
            detail="Stats caching not implemented. Stats are included in /infer response.",
        )

    return app
