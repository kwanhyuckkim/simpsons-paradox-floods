"""Annual peak streamflow event extraction paired with peak precipitation.

The function :func:`extract_annual_peaks` walks each gauge's daily streamflow
record, applies ``scipy.signal.find_peaks`` with a percentile-based height
threshold, computes a basin-area-binned time-of-concentration window, and pairs
each streamflow peak with the dominant precipitation event inside that window.

Peaks with no preceding precipitation (5-day cumulative < 5 mm) are tagged as
*dry-day* events and reported separately so the caller can decide whether to
exclude them.
"""

from __future__ import annotations

from collections.abc import Iterable
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from floodbhm.features.time_of_concentration import time_of_concentration

__all__ = ["PeakExtractionResult", "extract_annual_peaks"]


class PeakExtractionResult:
    """Container for peak extraction outputs."""

    def __init__(self, wet: pd.DataFrame, dry: pd.DataFrame):
        self.wet = wet
        self.dry = dry

    def __repr__(self) -> str:
        return f"PeakExtractionResult(wet={len(self.wet)} rows, dry={len(self.dry)} rows)"


def _extract_one_gauge(args: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Worker function for one gauge. Returns (wet_peaks, dry_peaks)."""
    gage_id, df_gauge, percent, dry_threshold_mm = args
    df_gauge = df_gauge.sort_values("date").reset_index(drop=True)
    q = df_gauge["streamflow"].to_numpy()
    p = df_gauge["prcp"].to_numpy()

    if len(q) < 100:
        return pd.DataFrame(), pd.DataFrame()

    height = float(np.nanpercentile(q, percent))
    peak_idx, _ = find_peaks(q, height=height, distance=10)

    area_km2 = float(df_gauge["AREA"].iloc[0]) if "AREA" in df_gauge.columns else 100.0
    slope = (
        float(df_gauge["SLOPE_PCT"].iloc[0]) / 100.0 if "SLOPE_PCT" in df_gauge.columns else 0.01
    )
    slope = max(slope, 1e-4)
    t_c_hours = time_of_concentration(area_km2=area_km2, slope=slope)
    t_c_days = max(1, int(np.ceil(t_c_hours / 24.0)))

    wet_rows = []
    dry_rows = []
    for i in peak_idx:
        start = max(0, i - t_c_days - 1)
        window_p = p[start : i + 1]
        if len(window_p) == 0:
            continue
        cum_5d = float(np.nansum(p[max(0, i - 4) : i + 1]))
        peak_p_value = float(window_p.max()) if window_p.size else 0.0

        row = {
            "GAGE_ID": gage_id,
            "date": df_gauge["date"].iloc[i],
            "streamflow": float(q[i]),
            "prcp": peak_p_value,
            "cum_ppt": cum_5d,
            "T_c": t_c_hours,
            "AREA": area_km2,
        }
        if cum_5d < dry_threshold_mm:
            dry_rows.append(row)
        else:
            wet_rows.append(row)

    return pd.DataFrame(wet_rows), pd.DataFrame(dry_rows)


def extract_annual_peaks(
    df: pd.DataFrame,
    *,
    percent: float = 90.0,
    dry_threshold_mm: float = 5.0,
    n_workers: int = 1,
) -> PeakExtractionResult:
    """Extract one annual peak event per ``(GAGE_ID, year)``.

    Args:
        df: Long-format DataFrame with columns
            ``GAGE_ID``, ``date``, ``streamflow``, ``prcp``, ``AREA``,
            ``SLOPE_PCT``.
        percent: Percentile threshold for ``find_peaks`` height (default 90).
        dry_threshold_mm: 5-day cumulative precipitation below which a peak is
            tagged as a dry-day event (default 5 mm).
        n_workers: Number of parallel worker processes. Use 1 for debugging,
            48 for full HPC node.

    Returns:
        :class:`PeakExtractionResult` with ``wet`` and ``dry`` DataFrames.

    Example:
        >>> result = extract_annual_peaks(df, percent=90.0, n_workers=4)
        >>> annual = result.wet.loc[result.wet.groupby(['GAGE_ID', 'year'])['streamflow'].idxmax()]
    """
    required = {"GAGE_ID", "date", "streamflow", "prcp"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"df missing required columns: {missing}")

    groups: Iterable[tuple] = (
        (gid, gdf, percent, dry_threshold_mm) for gid, gdf in df.groupby("GAGE_ID")
    )

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_extract_one_gauge, list(groups))
    else:
        results = [_extract_one_gauge(args) for args in groups]

    wet_dfs = [w for w, _ in results if not w.empty]
    dry_dfs = [d for _, d in results if not d.empty]

    wet = pd.concat(wet_dfs, ignore_index=True) if wet_dfs else pd.DataFrame()
    dry = pd.concat(dry_dfs, ignore_index=True) if dry_dfs else pd.DataFrame()

    if not wet.empty:
        wet["year"] = pd.to_datetime(wet["date"]).dt.year

    return PeakExtractionResult(wet=wet, dry=dry)
