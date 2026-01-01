import pandas as pd
import pytest

from utils.sampling import RandomSampler, StratifiedSampler


@pytest.fixture
def dummy_df():
    data = {
        "id": range(100),
        "category": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,
        "value": range(100),
    }
    return pd.DataFrame(data)


class TestRandomSampler:
    def test_sample_subset(self, dummy_df):
        sampler = RandomSampler(seed=42)
        sampled = sampler.sample(dummy_df, total_samples=10)
        assert len(sampled) == 10
        assert len(sampled["id"].unique()) == 10

    def test_sample_all(self, dummy_df):
        sampler = RandomSampler(seed=42)
        # Request more than available
        sampled = sampler.sample(dummy_df, total_samples=200)
        assert len(sampled) == 100

    def test_reproducibility(self, dummy_df):
        s1 = RandomSampler(seed=123)
        s2 = RandomSampler(seed=123)

        df1 = s1.sample(dummy_df, 10)
        df2 = s2.sample(dummy_df, 10)

        pd.testing.assert_frame_equal(df1, df2)

    def test_empty_df(self):
        df = pd.DataFrame({"a": []})
        sampler = RandomSampler()
        sampled = sampler.sample(df, 10)
        assert len(sampled) == 0


class TestStratifiedSampler:
    def test_stratified_counts(self, dummy_df):
        # A: 50, B: 30, C: 20 -> Total 100
        # Request 10 samples. Proportional: A=5, B=3, C=2
        sampler = StratifiedSampler(stratify_by="category", seed=42)
        sampled = sampler.sample(dummy_df, total_samples=10)

        assert len(sampled) == 10
        counts = sampled["category"].value_counts()
        assert counts["A"] == 5
        assert counts["B"] == 3
        assert counts["C"] == 2

    def test_min_per_strata(self, dummy_df):
        # Request 10 samples, but min 3 per strata.
        # 3 strata * 3 = 9 samples guaranteed. 1 left to allocate.
        # Proportions: A(0.5)->5, B(0.3)->3, C(0.2)->2.
        # With min 3: A->3 (needs 5 ideally), B->3, C->3 (needs 2 ideally).
        # Algo should boost C to 3. A might drop or just global constraints apply.

        sampler = StratifiedSampler(stratify_by="category", min_per_strata=3, seed=42)
        sampled = sampler.sample(dummy_df, total_samples=10)

        assert len(sampled) == 10  # Total budget is hard constraint
        counts = sampled["category"].value_counts()
        assert counts["A"] >= 3
        assert counts["B"] >= 3
        assert counts["C"] >= 3
        # Sum is 10. A=4, B=3, C=3 or similar.

    def test_missing_column(self, dummy_df):
        sampler = StratifiedSampler(stratify_by="non_existent", seed=42)
        # Should fallback to global sampling
        sampled = sampler.sample(dummy_df, total_samples=10)
        assert len(sampled) == 10
        # Temporary column should not be in output
        assert "non_existent" not in sampled.columns
        assert "_global" not in sampled.columns

    def test_weight_power(self, dummy_df):
        # High weight power favors larger groups more aggressively?
        # Or if weight_power=0, all groups equal weight?
        # Let's test weight_power=0 -> Equal allocation if possible

        sampler = StratifiedSampler(stratify_by="category", weight_power=0.0, seed=42)
        # 3 groups. 10 samples. 10/3 = 3.33...
        # Should be roughly equal: 3, 3, 4
        sampled = sampler.sample(dummy_df, total_samples=10)
        assert len(sampled) == 10
        counts = sampled["category"].value_counts()
        # With proportional (power=1), A gets 5.
        # With equal weighting (power=0), A should get close to 1/3 ~ 3 or 4.
        assert 3 <= counts["A"] <= 4

    def test_empty_df(self):
        df = pd.DataFrame({"category": [], "val": []})
        sampler = StratifiedSampler(stratify_by="category")
        sampled = sampler.sample(df, 10)
        assert len(sampled) == 0
