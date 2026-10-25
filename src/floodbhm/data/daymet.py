"""Daymet and PRISM loaders.

I normally start from zonal-statistics parquet files computed ahead of time (they
will go on Zenodo). To redo the zonal stats from the raw rasters, use the
exactextract CLI; apply_exactextract is a small wrapper around it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["load_zonal_parquet"]


def load_zonal_parquet(path: Path) -> pd.DataFrame:
    """Load a pre-computed zonal-stats parquet keyed by ``(GAGE_ID, date)``.

    Args:
        path: Path to the parquet.

    Returns:
        DataFrame with the expected schema (varies by variable).
    """
    df = pd.read_parquet(path)
    if "GAGE_ID" not in df.columns:
        raise ValueError(f"{path} missing GAGE_ID column")
    return df
