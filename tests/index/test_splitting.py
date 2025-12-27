import pytest

from data_preparation.index.hash_split import (
    assign_split,
    get_stable_hash_float,
    validate_fractions,
)


def test_stable_hashing():
    key = "tile_001"
    seed = 42
    u1 = get_stable_hash_float(key, seed)
    u2 = get_stable_hash_float(key, seed)
    assert u1 == u2
    assert 0 <= u1 < 1

    # Different seed should change hash
    u3 = get_stable_hash_float(key, 43)
    assert u1 != u3

    # Different key should change hash
    u4 = get_stable_hash_float("tile_002", seed)
    assert u1 != u4


def test_assign_split():
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    assert assign_split(0.1, fractions) == "train"
    assert assign_split(0.69, fractions) == "train"
    assert assign_split(0.71, fractions) == "val"
    assert assign_split(0.84, fractions) == "val"
    assert assign_split(0.86, fractions) == "test"
    assert assign_split(0.99, fractions) == "test"


def test_validate_fractions():
    validate_fractions({"train": 0.6, "val": 0.2, "test": 0.2})

    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_fractions({"train": 0.5, "val": 0.1, "test": 0.1})

    with pytest.raises(ValueError, match="non-negative"):
        validate_fractions({"train": 1.1, "val": -0.1, "test": 0.0})


def test_group_consistency():
    # Simulate assigning splits to multiple groups and ensure no leakage
    seed = 100
    fractions = {"train": 0.6, "val": 0.2, "test": 0.2}

    group_id = "FRA"
    u = get_stable_hash_float(group_id, seed)
    split = assign_split(u, fractions)

    # Re-running the logic for same group/seed MUST give same split
    u_new = get_stable_hash_float(group_id, seed)
    split_new = assign_split(u_new, fractions)
    assert split == split_new
