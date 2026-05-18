"""Global ISA (gISA) and Köppen-Geiger classification loaders.

Pre-computed zonal statistics live on Zenodo. ``load_isa_timeseries`` returns
one row per ``(GAGE_ID, year)`` with mean ISA fraction.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["load_isa_timeseries", "load_koppen_majority"]


def load_isa_timeseries(path: Path) -> pd.DataFrame:
    """Load gISA zonal-statistics parquet (mean ISA per gauge per year)."""
    df = pd.read_parquet(path)
    required = {"GAGE_ID", "year", "isa_mean"}
    if not required.issubset(df.columns):
        raise ValueError(f"isa parquet missing columns: {required - set(df.columns)}")
    return df


def load_koppen_majority(path: Path) -> pd.DataFrame:
    """Load Köppen-Geiger majority class per gauge."""
    df = pd.read_parquet(path)
    if "GAGE_ID" not in df.columns:
        raise ValueError("koppen parquet must contain GAGE_ID")
    return df
