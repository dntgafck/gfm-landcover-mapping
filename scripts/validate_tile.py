import sys
from pathlib import Path

import numpy as np
import rasterio


def describe(path: Path):
    """Print basic raster metadata and statistics."""
    with rasterio.open(path) as ds:
        arr = ds.read()
        print(f"\n== {path.name} ==")
        print(f"  Shape:     {arr.shape}")
        print(f"  Dtype:     {ds.dtypes[0]}")
        print(f"  CRS:       {ds.crs}")
        print(f"  Transform: {ds.transform}")
        print(f"  Res:       {ds.res}")

        # Stats per band
        for b in range(arr.shape[0]):
            band = arr[b]
            finite = np.isfinite(band)
            if not finite.any():
                print(f"  Band {b+1}: all non-finite")
                continue
            v = band[finite]
            print(
                f"  Band {b+1}: min={v.min():.4f} p50={np.percentile(v,50):.4f} max={v.max():.4f}"
            )
        return ds.crs, ds.transform, ds.width, ds.height, ds.bounds


def check_alignment(ref_meta, target_path, name):
    """Assert that target raster aligns with reference metadata."""
    with rasterio.open(target_path) as ds:
        ref_crs, ref_transform, ref_w, ref_h, _ = ref_meta

        ok = True
        if ds.crs != ref_crs:
            print(f"[FAIL] {name} CRS mismatch: {ds.crs} != {ref_crs}")
            ok = False
        if ds.transform != ref_transform:
            print(f"[FAIL] {name} Transform mismatch")
            ok = False
        if ds.width != ref_w or ds.height != ref_h:
            print(
                f"[FAIL] {name} Shape mismatch: {(ds.height, ds.width)} != {(ref_h, ref_w)}"
            )
            ok = False

        if ok:
            print(f"[OK] {name} is perfectly aligned with reference.")
        return ok


def check_categorical_values(path: Path, expected=None):
    with rasterio.open(path) as ds:
        data = ds.read(1)
        vals = np.unique(data)
        if expected:
            extra = set(vals) - set(expected)
            if extra:
                print(
                    f"[WARN] {path.name} has unexpected values: {sorted(list(extra))}"
                )
            else:
                print(f"[OK] {path.name} values are subset of expected.")
        else:
            print(f"[INFO] {path.name} unique values: {vals.tolist()}")


def validate_tile(
    spectral_path: Path,
    labels_path: Path | None = None,
    scl_path: Path | None = None,
    mask_path: Path | None = None,
):
    print("Validating Tile Components...")

    if not spectral_path.exists():
        print(f"[ERROR] Reference spectral.tif not found: {spectral_path}")
        return

    # 1. Authority Check
    ref_meta = describe(spectral_path)

    # 2. Alignment Checks
    components = [("labels", labels_path), ("scl", scl_path), ("mask", mask_path)]

    for name, path in components:
        if path and path.exists():
            check_alignment(ref_meta, path, name)
            if name == "labels":
                check_categorical_values(path)
            elif name == "mask":
                check_categorical_values(path, expected=[0, 1])
        elif path:
            print(f"[MISS] {name} not found at {path}")

    # 3. Semantic Checks
    with rasterio.open(spectral_path) as ds:
        # Assumes B04 (Red) is band 3, B08 (NIR) is band 4 for Sentinel-2
        if ds.count >= 4:
            # We don't strictly know band order from indices, but typically it is 2,3,4,8?
            # User script previously assumed b02, b03, b04, b08
            bands = ds.read()
            # If 4 bands, assume [B02, B03, B04, B08]
            b04 = bands[2]
            b08 = bands[3]
            ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
            finite = ndvi[np.isfinite(ndvi)]
            if finite.size > 0:
                print("\n[NDVI] (assuming bands 3=Red, 4=NIR):")
                print(
                    f"  min={finite.min():.3f} "
                    f"p50={np.percentile(finite, 50):.3f} "
                    f"max={finite.max():.3f}"
                )


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_tile.py <tile_path_or_iso_and_key>")
        print("Example: python scripts/validate_tile.py data/imagery/ESP/189b...")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # Smart discovery
    # data/imagery/<ISO>/<KEY>/spectral.tif
    # data/labels/<ISO>/<KEY>/labels.tif

    if "data/imagery" in str(input_path):
        imagery_dir = input_path if input_path.is_dir() else input_path.parent
        iso = imagery_dir.parent.name
        key = imagery_dir.name
        labels_dir = Path("data/labels") / iso / key
    elif "data/labels" in str(input_path):
        labels_dir = input_path if input_path.is_dir() else input_path.parent
        iso = labels_dir.parent.name
        key = labels_dir.name
        imagery_dir = Path("data/imagery") / iso / key
    else:
        # Assume path is a direct tile directory
        imagery_dir = input_path
        labels_dir = input_path  # Fallback to same dir

    validate_tile(
        spectral_path=imagery_dir / "spectral.tif",
        labels_path=labels_dir / "labels.tif",
        scl_path=imagery_dir / "scl.tif",
        mask_path=imagery_dir / "mask.tif",
    )


if __name__ == "__main__":
    main()
