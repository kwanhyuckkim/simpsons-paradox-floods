"""Simpson's paradox figure: pooled OLS vs group-specific slopes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

__all__ = ["plot_simpsons_paradox"]


def plot_simpsons_paradox(
    df: pd.DataFrame,
    *,
    x: str = "isa_mean",
    y: str = "streamflow",
    group_col: str = "GAGE_ID",
    min_n_per_group: int = 5,
    figsize: tuple[float, float] = (12, 5),
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, dict]:
    """Two-panel figure: pooled OLS slope vs distribution of per-group slopes.

    Panel A shows the pooled OLS regression of log(y) on log(x) over all rows.
    Panel B shows a histogram of per-group OLS slopes, with vertical references
    at 0 and the pooled slope.

    Args:
        df: DataFrame with positive ``x`` and ``y`` columns.
        x: Predictor column (positive values for log scale).
        y: Response column (positive).
        group_col: Column to stratify by (e.g., ``GAGE_ID``).
        min_n_per_group: Minimum group size to fit a slope.
        figsize: Matplotlib figure size.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        title: Optional figure title.

    Returns:
        ``(fig, stats)`` where ``stats`` includes the pooled slope and the
        fraction of groups with positive slope.
    """
    mask = (df[x] > 0) & (df[y] > 0)
    sub = df.loc[mask, [x, y, group_col]].copy()

    Xp = np.log(sub[x].to_numpy()).reshape(-1, 1)
    yp = np.log(sub[y].to_numpy())
    pooled = LinearRegression().fit(Xp, yp)
    pooled_slope = float(pooled.coef_[0])

    slopes: list[float] = []
    for _, grp in sub.groupby(group_col):
        if len(grp) < min_n_per_group:
            continue
        Xi = np.log(grp[x].to_numpy()).reshape(-1, 1)
        yi = np.log(grp[y].to_numpy())
        m = LinearRegression().fit(Xi, yi)
        slopes.append(float(m.coef_[0]))

    pct_positive = float(np.mean(np.array(slopes) > 0)) if slopes else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax_a = axes[0]
    ax_a.scatter(sub[x], sub[y], s=4, alpha=0.2, color="gray")
    xx = np.array([sub[x].min(), sub[x].max()])
    yy = np.exp(pooled.intercept_ + pooled_slope * np.log(xx))
    ax_a.plot(xx, yy, color="red", lw=2.0, label=f"pooled slope = {pooled_slope:.2f}")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel(xlabel or x)
    ax_a.set_ylabel(ylabel or y)
    ax_a.set_title("Pooled regression")
    ax_a.legend()
    ax_a.grid(True, which="both", ls=":", alpha=0.5)

    ax_b = axes[1]
    if slopes:
        ax_b.hist(slopes, bins=30, color="steelblue", edgecolor="black", alpha=0.85)
        ax_b.axvline(0.0, color="black", ls="--", lw=1.0, label="zero")
        ax_b.axvline(
            pooled_slope, color="red", ls="-", lw=1.5, label=f"pooled = {pooled_slope:.2f}"
        )
        ax_b.set_xlabel(f"Per-{group_col} log-log slope")
        ax_b.set_ylabel("Count")
        ax_b.set_title(f"Within-group slopes  ({pct_positive * 100:.0f}% positive)")
        ax_b.legend()
        ax_b.grid(True, ls=":", alpha=0.5)
    else:
        ax_b.text(0.5, 0.5, "No groups with ≥ min_n", ha="center", va="center")

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    return fig, {
        "pooled_slope": pooled_slope,
        "n_groups_fit": len(slopes),
        "pct_positive_slopes": pct_positive,
        "median_within_slope": float(np.median(slopes)) if slopes else float("nan"),
    }
