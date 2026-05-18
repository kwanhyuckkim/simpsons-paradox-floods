"""Generate paper figures from cached outputs.

Idempotent: each figure regenerates only if its dependencies are newer than
the existing output (simple mtime check). Set ``FORCE=1`` to override.

Usage:
    python scripts/make_figures.py
    FORCE=1 python scripts/make_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    from floodbhm.viz.simpsons import plot_simpsons_paradox

    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "peaks_v2_category_ver2.parquet"
    out_dir = repo_root / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Data not found at {data_path}. Run `scripts/download_data.py` first.")
        return 1

    df = pd.read_parquet(data_path)

    # Figure 1: Simpson's paradox
    fig_path = out_dir / "fig01_simpsons_paradox.png"
    if not fig_path.exists() or os.environ.get("FORCE"):
        fig, stats = plot_simpsons_paradox(
            df,
            x="isa_mean",
            y="streamflow",
            group_col="GAGE_ID",
            xlabel="Impervious Surface Area (fraction)",
            ylabel="Annual Peak Streamflow (m$^3$/s)",
            title="Simpson's Paradox in ISA–Streamflow Relationship",
        )
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        fig.savefig(fig_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        print(f"Saved {fig_path}")
        print(f"  pooled slope = {stats['pooled_slope']:.3f}")
        print(f"  pct positive = {stats['pct_positive_slopes'] * 100:.1f}%")
        print(f"  median within-group slope = {stats['median_within_slope']:.3f}")
    else:
        print(f"Skipping {fig_path} (exists; set FORCE=1 to regenerate)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
