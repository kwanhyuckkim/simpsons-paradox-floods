"""5-minute toy example: full pipeline on synthetic data.

Demonstrates:
    1. peak extraction from synthetic daily series
    2. BHM_Category construction
    3. plot_simpsons_paradox

Run:
    python examples/quickstart.py

Generates ``results/quickstart/simpsons_demo.png``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from floodbhm.features.grouping import build_bhm_category
from floodbhm.viz.simpsons import plot_simpsons_paradox


def synthetic_annual_peaks(n_gauges: int = 50, n_years: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate a fake dataset that exhibits Simpson's paradox.

    The within-gauge slope of log(streamflow) vs log(isa_mean) is positive,
    but pooled across gauges the slope is negative because high-ISA gauges
    happen to have smaller area.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_gauges):
        # High-ISA gauges are smaller (this creates Simpson's paradox)
        isa_baseline = rng.uniform(0.01, 0.50)
        area = max(10.0, 5000.0 * (1.0 - isa_baseline) + rng.normal(0.0, 200.0))
        for yr in range(n_years):
            isa = max(0.01, isa_baseline + rng.normal(0.0, 0.05))
            prcp = rng.uniform(20.0, 200.0)
            # Within-gauge: positive elasticity isa^0.5
            q = 0.0005 * prcp * area * (isa**0.5) * 100.0 * (1.0 + rng.normal(0.0, 0.1))
            q = max(q, 0.1)
            rows.append(
                {
                    "GAGE_ID": f"{g:08d}",
                    "year": 1990 + yr,
                    "streamflow": q,
                    "isa_mean": isa,
                    "AREA": area,
                    "Natural_Managed": "Natural" if rng.random() < 0.7 else "Managed",
                    "HUC02": f"{rng.integers(1, 19):02d}",
                    "koppen_category": str(rng.choice(["Cfa", "Dfa", "BSk"])),
                    "area_hydro_fine": int(rng.integers(0, 10)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = Path("results/quickstart")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = synthetic_annual_peaks()
    df = build_bhm_category(df)
    print(f"Synthetic peaks: {len(df)} rows, {df['GAGE_ID'].nunique()} gauges")
    print(f"Unique BHM_Category labels: {df['BHM_Category'].nunique()}")

    fig, stats = plot_simpsons_paradox(
        df,
        x="isa_mean",
        y="streamflow",
        group_col="GAGE_ID",
        xlabel="Impervious Surface Area (fraction)",
        ylabel="Annual Peak Streamflow",
        title="Simpson's Paradox: pooled OLS vs per-gauge slopes",
    )
    fig_path = out_dir / "simpsons_demo.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved {fig_path}")
    print(f"  pooled slope          : {stats['pooled_slope']:+.3f}")
    print(f"  median within-gauge   : {stats['median_within_slope']:+.3f}")
    print(f"  % positive within     : {stats['pct_positive_slopes'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
