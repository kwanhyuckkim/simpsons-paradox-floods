"""Hierarchical grouping for BHM_Category construction.

Builds the 4-tuple grouping ``Natural_Managed × HUC02 × Köppen × area_bin`` and
merges small groups (fewer than ``min_count`` members) with the nearest larger
group by haversine distance over gauge centroids.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["build_bhm_category", "merge_small_groups_by_distance"]


EARTH_RADIUS_KM = 6371.0


def build_bhm_category(
    df: pd.DataFrame,
    *,
    natural_managed_col: str = "Natural_Managed",
    huc_col: str = "HUC02",
    koppen_col: str = "koppen_category",
    area_bin_col: str = "area_hydro_fine",
    out_col: str = "BHM_Category",
) -> pd.DataFrame:
    """Compose the 4-tuple ``BHM_Category`` label.

    Args:
        df: DataFrame containing the four group columns.
        natural_managed_col: Column for managed/natural flag.
        huc_col: USGS HUC02 column.
        koppen_col: Köppen-Geiger class column.
        area_bin_col: Area-bin column (e.g., ``pd.qcut`` result).
        out_col: Name of output column.

    Returns:
        DataFrame with ``out_col`` added.
    """
    missing = [
        c
        for c in (natural_managed_col, huc_col, koppen_col, area_bin_col)
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"build_bhm_category: missing columns {missing}")
    out = df.copy()
    out[out_col] = (
        out[natural_managed_col].astype(str)
        + "_"
        + out[huc_col].astype(str)
        + "_"
        + out[koppen_col].astype(str)
        + "_"
        + out[area_bin_col].astype(str)
    )
    return out


def merge_small_groups_by_distance(
    df: pd.DataFrame,
    *,
    group_col: str = "BHM_Category",
    lat_col: str = "LAT_GAGE",
    lng_col: str = "LNG_GAGE",
    min_count: int = 10,
    max_radius_km: float = 150.0,
) -> pd.DataFrame:
    """Merge groups with fewer than ``min_count`` members into the nearest larger group.

    Uses scikit-learn ``BallTree`` with haversine metric on gauge centroids
    (group centroid = mean lat/lng of member rows). A small group is merged into
    the closest larger group within ``max_radius_km``. Groups with no nearby
    larger neighbor are kept as-is (still small).

    Args:
        df: DataFrame with ``group_col``, ``lat_col``, ``lng_col``.
        group_col: Column to merge.
        lat_col: Latitude column (degrees).
        lng_col: Longitude column (degrees).
        min_count: Minimum members; smaller groups get merged.
        max_radius_km: Maximum allowed merge distance (km).

    Returns:
        DataFrame with merged ``group_col``.
    """
    from sklearn.neighbors import BallTree

    if df.empty:
        return df.copy()

    counts = df[group_col].value_counts()
    small = counts[counts < min_count].index.tolist()
    large = counts[counts >= min_count].index.tolist()

    if not small or not large:
        return df.copy()

    centroids = df.groupby(group_col).agg(
        lat=(lat_col, "mean"), lng=(lng_col, "mean")
    )
    large_centroids = centroids.loc[large]
    tree = BallTree(np.radians(large_centroids[["lat", "lng"]].to_numpy()), metric="haversine")

    new_label = {}
    for small_grp in small:
        sc = centroids.loc[small_grp, ["lat", "lng"]].to_numpy().reshape(1, 2)
        dist_rad, idx = tree.query(np.radians(sc), k=1)
        dist_km = float(dist_rad[0, 0]) * EARTH_RADIUS_KM
        if dist_km <= max_radius_km:
            new_label[small_grp] = large_centroids.index[int(idx[0, 0])]

    out = df.copy()
    out[group_col] = out[group_col].replace(new_label)
    return out
