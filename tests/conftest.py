"""Shared pytest fixtures for floodbhm tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_peaks() -> pd.DataFrame:
    """A small synthetic peaks DataFrame for unit-test use.

    100 fake gauges × 30 fake years with realistic-ish ranges.
    """
    rng = np.random.default_rng(seed=42)
    n_gauges = 100
    n_years = 30
    rows = []
    for g in range(n_gauges):
        area = rng.uniform(10.0, 5000.0)
        for yr in range(n_years):
            isa = rng.uniform(0.0, 0.5)
            prcp = rng.uniform(20.0, 200.0)
            q = 0.001 * prcp * area * (1.0 + isa) + rng.normal(0.0, 5.0)
            q = max(q, 0.1)
            rows.append(
                {
                    "GAGE_ID": f"{g:08d}",
                    "year": 1990 + yr,
                    "streamflow": q,
                    "prcp_mean": prcp,
                    "isa_mean": isa,
                    "AREA": area,
                    "BFI_AVE": rng.uniform(0.1, 0.9),
                    "SLOPE_PCT": rng.uniform(0.1, 10.0),
                    "LAT_GAGE": rng.uniform(25.0, 49.0),
                    "LNG_GAGE": rng.uniform(-125.0, -67.0),
                    "Natural_Managed": "Natural" if rng.random() < 0.7 else "Managed",
                    "HUC02": f"{rng.integers(1, 19):02d}",
                    "koppen_category": rng.choice(["Cfa", "Dfa", "Dfb", "BSk", "Csa"]),
                    "area_hydro_fine": rng.integers(0, 10),
                }
            )
    return pd.DataFrame(rows)
