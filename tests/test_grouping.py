"""Unit tests for hierarchical grouping construction."""

from __future__ import annotations

import pandas as pd

from floodbhm.features.grouping import (
    build_bhm_category,
    merge_small_groups_by_distance,
)


def test_build_bhm_category(synthetic_peaks: pd.DataFrame):
    df = build_bhm_category(synthetic_peaks)
    assert "BHM_Category" in df.columns
    # All labels are concatenations of the 4 source columns
    sample = df.iloc[0]
    expected = (
        f"{sample['Natural_Managed']}_{sample['HUC02']}_"
        f"{sample['koppen_category']}_{sample['area_hydro_fine']}"
    )
    assert sample["BHM_Category"] == expected


def test_build_bhm_category_missing_col():
    import pytest

    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(ValueError):
        build_bhm_category(df)


def test_merge_small_groups_reduces_unique_labels(synthetic_peaks: pd.DataFrame):
    df = build_bhm_category(synthetic_peaks)
    n_before = df["BHM_Category"].nunique()
    merged = merge_small_groups_by_distance(df, min_count=10, max_radius_km=1e9)
    n_after = merged["BHM_Category"].nunique()
    assert n_after <= n_before


def test_merge_with_no_small_groups_is_noop(synthetic_peaks: pd.DataFrame):
    df = build_bhm_category(synthetic_peaks)
    # min_count=1 means no group is "small"
    merged = merge_small_groups_by_distance(df, min_count=1)
    assert merged["BHM_Category"].equals(df["BHM_Category"])
