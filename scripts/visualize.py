from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def read_stack(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read()  # (bands, H, W)
        return arr, ds


def stretch_uint8(img, p2=2, p98=98):
    """Percentile stretch to 0..1 for display."""
    img = img.astype(np.float32)
    lo = np.percentile(img[np.isfinite(img)], p2)
    hi = np.percentile(img[np.isfinite(img)], p98)
    if hi <= lo:
        return np.clip(img, 0, 1)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def main(tile_dir: str):
    td = Path(tile_dir)

    spectral_path = td / "spectral.tif"
    scl_path = td / "scl.tif"
    mask_path = td / "mask.tif"
    labels_path = td / "labels.tif"  # optional

    if not spectral_path.exists():
        raise FileNotFoundError(f"Missing {spectral_path}")

    spectral, _ = read_stack(spectral_path)  # expected: [B02,B03,B04,B08]
    b02, b03, b04, b08 = spectral

    # RGB: (R,G,B) = (B04,B03,B02)
    rgb = np.stack([b04, b03, b02], axis=-1)
    rgb_disp = stretch_uint8(rgb)

    # NDVI
    ndvi = (b08 - b04) / (b08 + b04 + 1e-6)

    # Masks
    scl = None
    if scl_path.exists():
        scl, _ = read_stack(scl_path)
        scl = scl[0]

    datamask = None
    if mask_path.exists():
        datamask, _ = read_stack(mask_path)
        datamask = datamask[0]

    labels = None
    if labels_path.exists():
        labels, _ = read_stack(labels_path)
        labels = labels[0]

    # Plot
    plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(rgb_disp)
    ax1.set_title("RGB (B04,B03,B02) – stretched")
    ax1.axis("off")

    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(ndvi, vmin=-1, vmax=1)
    ax2.set_title("NDVI (B08,B04)")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(2, 2, 3)
    if scl is not None:
        im3 = ax3.imshow(scl)
        ax3.set_title("SCL (Scene Classification Layer)")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    else:
        ax3.text(0.5, 0.5, "No scl.tif found", ha="center", va="center")
        ax3.set_title("SCL")
    ax3.axis("off")

    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(rgb_disp)
    title = "RGB + overlays"
    if datamask is not None:
        # show invalid pixels in red
        invalid = datamask == 0
        overlay = np.zeros((*invalid.shape, 4), dtype=np.float32)
        overlay[invalid] = [1, 0, 0, 0.35]
        ax4.imshow(overlay)
        title += " (red=invalid dataMask)"
    if labels is not None:
        # overlay labels lightly
        ax4.imshow(labels, alpha=0.25)
        title += " + labels (alpha)"
    ax4.set_title(title)
    ax4.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
