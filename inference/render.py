"""PNG rendering with ESA WorldCover colormap."""

import io
from pathlib import Path

import numpy as np
from PIL import Image

from utils.logging import get_logger

logger = get_logger(__name__)

# ESA WorldCover color palette (RGB values for classes 0-10)
# Colors match the official ESA WorldCover visualization
WORLDCOVER_PALETTE = {
    0: (0, 100, 0),  # Tree cover - dark green
    1: (255, 187, 34),  # Shrubland - orange
    2: (255, 255, 76),  # Grassland - yellow
    3: (240, 150, 255),  # Cropland - pink
    4: (250, 0, 0),  # Built-up - red
    5: (180, 180, 180),  # Bare / sparse vegetation - gray
    6: (240, 240, 240),  # Snow and ice - white
    7: (0, 100, 200),  # Permanent water bodies - blue
    8: (0, 150, 160),  # Herbaceous wetland - teal
    9: (0, 207, 117),  # Mangroves - green
    10: (250, 230, 160),  # Moss and lichen - beige
    255: (0, 0, 0),  # No data / ignore - black
}


def create_colormap_lut() -> np.ndarray:
    """Create a 256x3 lookup table for fast colormap application.

    Returns:
        Lookup table array of shape [256, 3]
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for class_id, color in WORLDCOVER_PALETTE.items():
        if class_id < 256:
            lut[class_id] = color
    return lut


# Pre-computed lookup table
_COLORMAP_LUT = create_colormap_lut()


def apply_colormap(array: np.ndarray) -> np.ndarray:
    """Apply WorldCover colormap to class ID array.

    Args:
        array: Class ID array [H, W] with values 0-10 (and 255 for ignore)

    Returns:
        RGB array [H, W, 3]
    """
    # Use lookup table for fast conversion
    return _COLORMAP_LUT[array]


def render_class_map(array: np.ndarray, with_legend: bool = True) -> Image.Image:
    """Render class prediction or label array as PIL Image.

    Args:
        array: Class ID array [H, W]
        with_legend: Whether to include legend on the right side

    Returns:
        PIL Image with colormap applied (and optional legend)
    """
    rgb_array = apply_colormap(array.astype(np.uint8))
    map_img = Image.fromarray(rgb_array, mode="RGB")

    if not with_legend:
        return map_img

    # Create legend and composite
    legend = create_legend_image(width=180, item_height=22)

    # Create canvas with space for legend on right
    h, w = array.shape
    legend_w = legend.width
    canvas = Image.new(
        "RGB", (w + legend_w + 10, max(h, legend.height)), (255, 255, 255)
    )

    # Paste map and legend
    canvas.paste(map_img, (0, 0))
    canvas.paste(legend, (w + 10, 5))

    return canvas


def render_side_by_side(
    pred: np.ndarray,
    label: np.ndarray,
    gap: int = 4,
    gap_color: tuple[int, int, int] = (255, 255, 255),
    with_legend: bool = True,
) -> Image.Image:
    """Render prediction and label side by side.

    Args:
        pred: Prediction class array [H, W]
        label: Label class array [H, W]
        gap: Gap width between images in pixels
        gap_color: RGB color for the gap
        with_legend: Whether to include legend on the right side

    Returns:
        PIL Image with both visualizations (and optional legend)
    """
    pred_rgb = apply_colormap(pred.astype(np.uint8))
    label_rgb = apply_colormap(label.astype(np.uint8))

    h, w, _ = pred_rgb.shape

    # Create combined image with gap
    combined = np.full((h, w * 2 + gap, 3), gap_color, dtype=np.uint8)
    combined[:, :w] = pred_rgb
    combined[:, w + gap :] = label_rgb

    map_img = Image.fromarray(combined, mode="RGB")

    if not with_legend:
        return map_img

    # Add legend
    legend = create_legend_image(width=180, item_height=22)
    canvas_w = w * 2 + gap + legend.width + 10
    canvas = Image.new("RGB", (canvas_w, max(h, legend.height)), (255, 255, 255))
    canvas.paste(map_img, (0, 0))
    canvas.paste(legend, (w * 2 + gap + 10, 5))

    return canvas


def save_png(image: Image.Image, path: Path) -> None:
    """Save PIL Image as PNG file.

    Args:
        image: PIL Image to save
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    logger.debug(f"Saved PNG: {path}")


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes.

    Args:
        image: PIL Image
        format: Image format (default: PNG)

    Returns:
        Image bytes
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def create_legend_image(
    width: int = 200,
    item_height: int = 24,
    font_size: int = 12,
) -> Image.Image:
    """Create a legend image for the colormap.

    Args:
        width: Legend width in pixels
        item_height: Height of each legend item
        font_size: Font size for labels

    Returns:
        PIL Image with legend
    """
    from PIL import ImageDraw

    num_classes = 11
    height = num_classes * item_height + 10

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    class_names = [
        "Tree cover",
        "Shrubland",
        "Grassland",
        "Cropland",
        "Built-up",
        "Bare/sparse veg.",
        "Snow and ice",
        "Water bodies",
        "Herbaceous wetland",
        "Mangroves",
        "Moss and lichen",
    ]

    for i, name in enumerate(class_names):
        y = 5 + i * item_height
        color = WORLDCOVER_PALETTE[i]

        # Draw color box
        draw.rectangle([10, y, 30, y + item_height - 4], fill=color, outline=(0, 0, 0))

        # Draw label
        draw.text((40, y + 2), name, fill=(0, 0, 0))

    return img
