import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Sampler(ABC):
    """
    Abstract base class for sampling strategies.
    """

    @abstractmethod
    def sample(self, df: pd.DataFrame, total_samples: int) -> pd.DataFrame:
        """
        Selects a subset of rows from the DataFrame.

        Args:
            df: Input DataFrame.
            total_samples: Desired number of samples.

        Returns:
            A DataFrame containing the sampled rows.
        """
        pass


class RandomSampler(Sampler):
    """
    Strategy for simple random sampling.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def sample(self, df: pd.DataFrame, total_samples: int) -> pd.DataFrame:
        if len(df) == 0:
            return df

        if len(df) <= total_samples:
            return df

        return df.sample(n=total_samples, random_state=self.seed)


class StratifiedSampler(Sampler):
    """
    Strategy for stratified sampling ensuring minimum representation per group.
    """

    def __init__(
        self,
        stratify_by: str,
        min_per_strata: int = 0,
        weight_power: float = 1.0,
        seed: int = 42,
    ):
        self.stratify_by = stratify_by
        self.min_per_strata = min_per_strata
        self.weight_power = weight_power
        self.seed = seed

    def sample(self, df: pd.DataFrame, total_samples: int) -> pd.DataFrame:
        if len(df) == 0:
            return df

        # If total samples requested is more than available, return everything
        # BUT existing logic in sample_stratified checked this differently (via allocate_budget).
        # allocate_budget handles cases where total_budget might be different.
        # But generally if total_samples >= len(df), we usually just want everything.
        # However, StratifiedSampler might want to enforce distribution?
        # The original code:
        # if n_samples >= len(group): take all
        # So if total_samples is huge, it takes all.

        stratify_col = self.stratify_by
        # Handle missing column
        if stratify_col not in df.columns:
            logger.warning(
                f"Stratification column '{stratify_col}' not in DataFrame. "
                "treating as single group."
            )
            # Create a temporary view
            df_view = df.copy(deep=False)
            df_view["_global"] = "global"
            stratify_col = "_global"
            groups = df_view.groupby(stratify_col)
        else:
            groups = df.groupby(stratify_col)

        allocation = self._allocate_budget(groups, total_samples)

        logger.info(
            f"Stratified Allocation (Total: {total_samples}, Min/Strata: {self.min_per_strata}):"
        )
        for k, v in allocation.items():
            logger.info(f"  {k}: {v}")

        rng = np.random.default_rng(self.seed)
        sampled_list = []

        # Iterate groups in sorted order for determinism
        sorted_keys = sorted(allocation.keys())

        for key in sorted_keys:
            n_samples = allocation[key]
            group = groups.get_group(key)

            if n_samples >= len(group):
                sampled_list.append(group)
            else:
                indices = group.index.to_numpy()
                chosen_indices = rng.choice(indices, size=n_samples, replace=False)
                # Sort indices to maintain relative order
                chosen_indices.sort()
                sampled_list.append(group.loc[chosen_indices])

        if not sampled_list:
            return df.iloc[0:0]

        result = pd.concat(sampled_list)

        # Cleanup temporary column if used
        if "_global" in result.columns and self.stratify_by not in df.columns:
            # Logic match: if we used _global because original was missing
            if stratify_col == "_global":
                result = result.drop(columns=["_global"])

        return result

    def _allocate_budget(self, groups, total_budget):
        """
        Allocate budget per stratum (group) based on population size.
        """
        # 1. Calculate weights
        stats = []

        # Handle GroupBy object directly or iterator
        if hasattr(groups, "__iter__") and not isinstance(groups, pd.core.groupby.GroupBy):
            group_iter = groups
        else:
            group_iter = groups

        for key, df in group_iter:
            n_c = len(df)
            weight = n_c**self.weight_power
            stats.append({"key": key, "n_c": n_c, "weight": weight})

        stats_df = pd.DataFrame(stats)
        if stats_df.empty:
            return {}
        if total_budget == 0:
            return dict(zip(stats_df["key"], [0] * len(stats_df), strict=False))

        total_weight = stats_df["weight"].sum()

        # 2. Initial allocation
        if total_weight == 0:
            stats_df["allocated"] = 0
        else:
            stats_df["allocated"] = np.floor(
                total_budget * stats_df["weight"] / total_weight
            ).astype(int)

        # 3. Enforce min/max constraints
        def apply_constraints(row):
            target = row["allocated"]
            # Floor is min needed, but cannot exceed population
            floor_val = min(row["n_c"], self.min_per_strata)
            target = max(target, floor_val)
            target = min(target, row["n_c"])
            return int(target)

        stats_df["allocated"] = stats_df.apply(apply_constraints, axis=1)

        # 4. Adjust to match total_budget exactly
        current_total = stats_df["allocated"].sum()
        diff = total_budget - current_total

        if diff == 0:
            return dict(zip(stats_df["key"], stats_df["allocated"], strict=False))

        stats_df = stats_df.sort_values("key")

        if diff > 0:
            # Need to ADD samples (under-allocated due to rounding down)
            while diff > 0:
                mask = stats_df["allocated"] < stats_df["n_c"]
                if not mask.any():
                    break

                stats_df["ideal"] = total_budget * stats_df["weight"] / total_weight
                stats_df["residual"] = stats_df["ideal"] - stats_df["allocated"]

                # Give to those with highest residual (most under-represented relative to ideal)
                best_idx = (
                    stats_df.loc[mask]
                    .sort_values(by=["residual", "key"], ascending=[False, True])
                    .index[0]
                )

                stats_df.at[best_idx, "allocated"] += 1
                diff -= 1
            return dict(zip(stats_df["key"], stats_df["allocated"], strict=False))

        # Need to REMOVE samples (over-allocated due to min constraints)
        while diff < 0:

            def get_floor(row):
                return min(row["n_c"], self.min_per_strata)

            stats_df["floor"] = stats_df.apply(get_floor, axis=1)
            # Can only remove from those above their floor
            candidates = stats_df[stats_df["allocated"] > stats_df["floor"]]

            if candidates.empty:
                logger.warning("Could not reduce count to total_budget due to minimum constraints.")
                break

            stats_df["ideal"] = total_budget * stats_df["weight"] / total_weight
            stats_df["residual"] = stats_df["ideal"] - stats_df["allocated"]

            # Remove from those with lowest residual (most over-represented)
            best_idx = (
                stats_df.loc[candidates.index]
                .sort_values(by=["residual", "key"], ascending=[True, True])
                .index[0]
            )

            stats_df.at[best_idx, "allocated"] -= 1
            diff += 1

        return dict(zip(stats_df["key"], stats_df["allocated"], strict=False))
