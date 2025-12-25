import pandas as pd

from scripts.load_data.processors import TileSampler


def allocate_budget(*args, **kwargs):
    return TileSampler.allocate_budget(*args, **kwargs)


def test_allocate_budget_basic():
    # Setup
    groups = [
        ("ITA", pd.DataFrame({"id": range(100)})),
        ("ESP", pd.DataFrame({"id": range(100)})),
    ]
    total_budget = 100
    weight_power = 1.0
    min_per_country = 10

    # Execute
    allocation = allocate_budget(groups, total_budget, weight_power, min_per_country)

    # Assert
    assert sum(allocation.values()) == total_budget
    assert allocation["ITA"] == 50
    assert allocation["ESP"] == 50


def test_allocate_budget_min_constraints():
    groups = [
        ("ITA", pd.DataFrame({"id": range(200)})),
        ("VAT", pd.DataFrame({"id": range(5)})),  # Very small
    ]
    total_budget = 100
    weight_power = 1.0
    min_per_country = 50

    allocation = allocate_budget(groups, total_budget, weight_power, min_per_country)

    assert sum(allocation.values()) == total_budget
    assert allocation["VAT"] == 5
    assert allocation["ITA"] == 95


def test_allocate_budget_weight_power():
    groups = [
        ("ITA", pd.DataFrame({"id": range(100)})),
        ("ESP", pd.DataFrame({"id": range(400)})),
    ]
    total_budget = 100
    weight_power = 0.5  # sqrt(100)=10, sqrt(400)=20. Ratio 1:2
    min_per_country = 10

    allocation = allocate_budget(groups, total_budget, weight_power, min_per_country)

    # ITA: 100 * (10/30) = 33.33 -> 33
    # ESP: 100 * (20/30) = 66.66 -> 66
    # Total 99. Diff 1. Residual ITA=0.33, ESP=0.66. ESP gets +1 -> 67.
    assert sum(allocation.values()) == total_budget
    assert allocation["ITA"] == 33
    assert allocation["ESP"] == 67


def test_allocate_budget_zero_budget():
    groups = [
        ("ITA", pd.DataFrame({"id": range(100)})),
    ]
    total_budget = 0
    weight_power = 1.0
    min_per_country = 10

    allocation = allocate_budget(groups, total_budget, weight_power, min_per_country)
    assert sum(allocation.values()) == 0
    assert allocation["ITA"] == 0
