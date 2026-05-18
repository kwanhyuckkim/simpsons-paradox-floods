"""Hydrologic and prediction-interval metrics.

All metrics return scalar floats. Input arrays must be 1-D numpy arrays or
``pandas.Series`` of identical length. ``NaN`` values are not handled
implicitly — pass cleaned data.

Reference for hydrologic metrics: Gupta et al. (2009),
"Decomposition of the mean squared error and NSE performance criteria",
*J. Hydrol.* 377, 80–91.

Reference for interval score: Gneiting & Raftery (2007),
"Strictly proper scoring rules, prediction, and estimation",
*J. Am. Stat. Assoc.* 102, 359–378.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "adjusted_r_squared",
    "kling_gupta_efficiency",
    "nash_sutcliffe_efficiency",
    "nmpiw_iqr",
    "pbias",
    "picp",
    "smape",
    "winkler_interval_score",
]


def _as_1d(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D input, got shape {arr.shape}")
    return arr


def smape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-9) -> float:
    """Symmetric mean absolute percentage error (in percent, range 0-200).

    SMAPE = mean( 200 * |yhat - y| / (|y| + |yhat| + eps) )

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        eps: Small constant guarding against division by zero when both
            ``|y|`` and ``|yhat|`` are zero.

    Returns:
        SMAPE in percent (0 = perfect, 200 = worst).

    Example:
        >>> smape([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        0.0
    """
    y_true_arr = _as_1d(y_true)
    y_pred_arr = _as_1d(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    num = np.abs(y_pred_arr - y_true_arr)
    den = np.abs(y_true_arr) + np.abs(y_pred_arr) + eps
    return float(200.0 * np.mean(num / den))


def adjusted_r_squared(y_true: ArrayLike, y_pred: ArrayLike, n_features: int) -> float:
    """Adjusted R-squared accounting for model degrees of freedom.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        n_features: Number of predictor features in the model (must be < n_samples).

    Returns:
        Adjusted R² (can be negative; 1 = perfect fit).
    """
    y_true_arr = _as_1d(y_true)
    y_pred_arr = _as_1d(y_pred)
    n_samples = y_true_arr.size
    if n_samples <= n_features + 1:
        raise ValueError(f"n_samples ({n_samples}) must exceed n_features+1 ({n_features + 1})")
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    r2 = 1.0 - ss_res / ss_tot
    adj = 1.0 - (1.0 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return float(adj)


def nash_sutcliffe_efficiency(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Nash-Sutcliffe efficiency.

    NSE ∈ (-∞, 1], with 1 = perfect, 0 = no better than predicting the mean.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        NSE.
    """
    y_true_arr = _as_1d(y_true)
    y_pred_arr = _as_1d(y_pred)
    num = float(np.sum((y_pred_arr - y_true_arr) ** 2))
    den = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    if den == 0.0:
        return float("nan")
    return float(1.0 - num / den)


def kling_gupta_efficiency(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Kling-Gupta efficiency (decomposition-based diagnostic).

    KGE = 1 - sqrt((r-1)² + (alpha-1)² + (beta-1)²), where r is Pearson correlation,
    alpha = sigma_pred/sigma_true (relative variability), beta = mean_pred/mean_true
    (bias ratio). Range (-∞, 1], with 1 = perfect.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        KGE.
    """
    y_true_arr = _as_1d(y_true)
    y_pred_arr = _as_1d(y_pred)
    mu_t = float(np.mean(y_true_arr))
    mu_p = float(np.mean(y_pred_arr))
    sd_t = float(np.std(y_true_arr, ddof=0))
    sd_p = float(np.std(y_pred_arr, ddof=0))
    if mu_t == 0.0 or sd_t == 0.0:
        return float("nan")
    r = float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1])
    alpha = sd_p / sd_t
    beta = mu_p / mu_t
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def pbias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Percent bias.

    PBIAS = 100 * sum(yhat - y) / sum(y). Positive = overestimation,
    negative = underestimation. Perfect = 0.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        PBIAS in percent.
    """
    y_true_arr = _as_1d(y_true)
    y_pred_arr = _as_1d(y_pred)
    den = float(np.sum(y_true_arr))
    if den == 0.0:
        return float("nan")
    return float(100.0 * np.sum(y_pred_arr - y_true_arr) / den)


def picp(y_true: ArrayLike, y_lower: ArrayLike, y_upper: ArrayLike) -> float:
    """Prediction Interval Coverage Probability.

    PICP = mean( I(y_lower <= y <= y_upper) ). Should match the nominal coverage
    (e.g., 0.90 for a 90% prediction interval).

    Args:
        y_true: Observed values.
        y_lower: Lower quantile predictions.
        y_upper: Upper quantile predictions.

    Returns:
        Fraction of observations inside the interval, in [0, 1].
    """
    y = _as_1d(y_true)
    lo = _as_1d(y_lower)
    hi = _as_1d(y_upper)
    inside = (y >= lo) & (y <= hi)
    return float(np.mean(inside))


def nmpiw_iqr(y_true: ArrayLike, y_lower: ArrayLike, y_upper: ArrayLike) -> float:
    """Normalized Mean Prediction Interval Width, normalized by the observed IQR.

    NMPIW = mean(y_upper - y_lower) / IQR(y_true). Smaller is sharper.
    Robust to extreme tails (uses IQR instead of range).

    Args:
        y_true: Observed values.
        y_lower: Lower quantile predictions.
        y_upper: Upper quantile predictions.

    Returns:
        Width ratio (unitless).
    """
    y = _as_1d(y_true)
    lo = _as_1d(y_lower)
    hi = _as_1d(y_upper)
    widths = hi - lo
    mpiw = float(np.mean(widths))
    q1, q3 = np.percentile(y, [25, 75])
    iqr = float(q3 - q1)
    if iqr == 0.0:
        return float("nan")
    return float(mpiw / iqr)


def winkler_interval_score(
    y_true: ArrayLike,
    y_lower: ArrayLike,
    y_upper: ArrayLike,
    alpha: float = 0.10,
) -> float:
    """Winkler interval score (a.k.a. interval score IS_alpha).

    IS = (upper - lower) + (2/alpha) * max(0, lower - y) + (2/alpha) * max(0, y - upper)

    The score rewards sharp intervals that contain the observation. Smaller is better.

    Args:
        y_true: Observed values.
        y_lower: Lower quantile predictions (alpha/2 quantile).
        y_upper: Upper quantile predictions (1-alpha/2 quantile).
        alpha: Total miscoverage probability (e.g., 0.10 for 90% interval).

    Returns:
        Mean interval score across samples.

    Reference:
        Gneiting & Raftery 2007, Eq. 43.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    y = _as_1d(y_true)
    lo = _as_1d(y_lower)
    hi = _as_1d(y_upper)
    widths = hi - lo
    lower_pen = np.maximum(lo - y, 0.0)
    upper_pen = np.maximum(y - hi, 0.0)
    score = widths + (2.0 / alpha) * (lower_pen + upper_pen)
    return float(np.mean(score))
