"""Pydantic request/response models for inference API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool = False
    data_ready: bool = False


class ModelInfoResponse(BaseModel):
    """Model information response."""

    source: str
    model_uri_or_path: str
    providers: list[str]
    input_names: list[str]
    output_names: list[str]


class TileMetaResponse(BaseModel):
    """Tile metadata response."""

    tile_id: str
    country: str
    width: int
    height: int
    crs: str
    bounds: list[float]  # [minx, miny, maxx, maxy]
    band_count: int


class InferRequest(BaseModel):
    """Inference request body."""

    country: str = Field(..., description="Country code (e.g., 'DEU')")
    tile_id: str = Field(..., description="Tile identifier")
    row_off: int = Field(0, ge=0, description="Row offset for window")
    col_off: int = Field(0, ge=0, description="Column offset for window")
    height: int = Field(256, gt=0, description="Window height in pixels")
    width: int = Field(256, gt=0, description="Window width in pixels")

    # Optional overrides
    patch_size: int | None = Field(None, gt=0, description="Override patch size")
    stride: int | None = Field(None, gt=0, description="Override stride")
    batch_size: int | None = Field(None, gt=0, description="Override batch size")

    # Output toggles
    include_pred: bool = Field(True, description="Include prediction PNG")
    include_label: bool = Field(True, description="Include ground-truth PNG")
    include_compare: bool = Field(True, description="Include side-by-side PNG")


class WindowInfo(BaseModel):
    """Information about the inference window."""

    row_off: int
    col_off: int
    height: int
    width: int
    actual_height: int
    actual_width: int


class ClassHistogram(BaseModel):
    """Class distribution histogram."""

    class_id: int
    class_name: str
    pixel_count: int
    fraction: float


class InferStats(BaseModel):
    """Inference statistics returned as JSON."""

    inference_id: str
    timings: dict[str, float]  # e.g., {"preprocess": 0.1, "inference": 0.5, ...}
    window_info: WindowInfo
    pred_histogram: list[ClassHistogram]
    label_histogram: list[ClassHistogram] | None


class CountryListResponse(BaseModel):
    """List of available countries."""

    countries: list[str]


class TileListResponse(BaseModel):
    """List of tiles for a country."""

    country: str
    tiles: list[str]
