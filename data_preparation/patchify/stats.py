from typing import Any

import numpy as np


def compute_valid_frac(mask: np.ndarray) -> float:
    """
    Computes fraction of valid pixels from data mask.
    Assumes mask == 1 is valid.
    """
    return float(np.mean(mask == 1))


def compute_cloud_frac(scl: np.ndarray, cloud_set: list[int]) -> float:
    """
    Computes fraction of pixels in the cloud set from SCL.
    """
    return float(np.mean(np.isin(scl, cloud_set)))


def compute_label_stats(
    labels: np.ndarray, ignore_values: list[int] | None = None
) -> dict[str, Any]:
    """
    Computes unique classes, dominant class and its fraction.
    """
    if ignore_values is None:
        ignore_values = []

    mask = ~np.isin(labels, ignore_values)
    valid_labels = labels[mask]

    if valid_labels.size == 0:
        return {"unique_classes": 0, "dominant_class": None, "dominant_frac": 0.0}

    unique, counts = np.unique(valid_labels, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(unique, counts, strict=False)}
    idx = np.argmax(counts)

    return {
        "unique_classes": len(unique),
        "dominant_class": int(unique[idx]),
        "dominant_frac": float(counts[idx] / labels.size),
        "class_counts": class_counts,
    }
