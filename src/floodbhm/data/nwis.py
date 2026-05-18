"""USGS NWIS daily streamflow loader via the ``dataretrieval`` package."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

__all__ = ["load_nwis_daily"]


def load_nwis_daily(
    gage_ids: Iterable[str],
    *,
    start: str = "1985-01-01",
    end: str = "2019-12-31",
    parameter_code: str = "00060",
) -> pd.DataFrame:
    """Download USGS NWIS daily values for the given gauges.

    Args:
        gage_ids: Iterable of USGS site IDs (8-digit strings).
        start: ISO date string.
        end: ISO date string.
        parameter_code: NWIS parameter (default ``00060`` = daily mean discharge).

    Returns:
        Long-format DataFrame with columns ``[GAGE_ID, date, streamflow]``.
        ``streamflow`` is in cubic feet per second (cfs) — convert downstream
        if needed.

    Note:
        Requires network access. Errors per-gauge are logged but do not halt
        the loop — caller should check the returned ``GAGE_ID`` set.
    """
    from dataretrieval import nwis

    rows = []
    for gid in gage_ids:
        try:
            df, _ = nwis.get_record(
                sites=gid, service="dv", start=start, end=end, parameterCd=parameter_code
            )
        except Exception as e:  # noqa: BLE001
            print(f"NWIS fetch failed for {gid}: {e}")
            continue
        if df.empty:
            continue
        df = df.reset_index()
        col = f"{parameter_code}_Mean"
        if col not in df.columns:
            continue
        df_clean = pd.DataFrame(
            {
                "GAGE_ID": gid,
                "date": pd.to_datetime(df["datetime"]),
                "streamflow": df[col].astype(float),
            }
        )
        rows.append(df_clean)
    if not rows:
        return pd.DataFrame(columns=["GAGE_ID", "date", "streamflow"])
    return pd.concat(rows, ignore_index=True)
