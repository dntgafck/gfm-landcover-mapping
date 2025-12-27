import numpy as np

from data_preparation.patchify.stats import (
    compute_cloud_frac,
    compute_label_stats,
    compute_valid_frac,
    is_usable,
)


def test_compute_valid_frac():
    mask = np.array([[1, 1], [0, 1]])
    assert compute_valid_frac(mask) == 0.75


def test_compute_cloud_frac():
    scl = np.array([[3, 4], [8, 4]])
    cloud_set = [3, 8]
    assert compute_cloud_frac(scl, cloud_set) == 0.5


def test_compute_label_stats():
    labels = np.array([[10, 10], [20, 10]])
    stats = compute_label_stats(labels)
    assert stats["unique_classes"] == 2
    assert stats["dominant_class"] == 10
    assert stats["dominant_frac"] == 0.75


def test_compute_label_stats_with_ignore():
    labels = np.array([[0, 10], [10, 20]])
    stats = compute_label_stats(labels, ignore_values=[0])
    assert stats["unique_classes"] == 2
    # Dominant among all pixels (denominator is labels.size = 4)
    # Counts: 10 (2), 20 (1)
    assert stats["dominant_class"] == 10
    assert stats["dominant_frac"] == 0.5


def test_is_usable():
    assert is_usable(0.95, 0.05) is True
    assert is_usable(0.85, 0.05) is False
    assert is_usable(0.95, 0.15) is False
