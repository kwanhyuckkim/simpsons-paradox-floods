"""Stability-weighted Recursive Feature Elimination using QRF importances.

At each iteration we fit a Quantile Random Forest under ``GroupKFold(GAGE_ID)``,
collect per-fold MDI importances, and drop the feature with the lowest
composite score:

.. math::

    c_j = \\mu_j \\cdot \\left(1 - \\min(s_j, t_{stab}) / t_{stab}\\right),
    \\quad s_j = \\sigma_j / \\mu_j

The stability term penalizes features whose importance varies across folds.

Differences vs the original ``rfe.py``:

- ``RandomForestQuantileRegressor.predict()`` is now called with explicit
  ``quantiles=[0.5]`` rather than the default (whose semantics changed across
  ``quantile-forest`` versions).
- Resume logic uses a JSON manifest rather than appending to a plain text file,
  reducing parsing errors.
- The optimal-feature recovery step (``KeyError: 'var'`` in the original) is
  fixed by using consistent column names throughout.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["RFEResult", "rfe_with_stability"]


@dataclass
class RFEResult:
    """Per-iteration RFE record."""

    history: pd.DataFrame
    optimal_n_features: int
    optimal_features: list[str]


def _composite_score(
    mean_importance: pd.Series,
    std_importance: pd.Series,
    stability_threshold: float,
) -> pd.Series:
    """Combine mean importance with cross-fold stability."""
    stability = std_importance / mean_importance.replace(0.0, 1e-10)
    stability = stability.fillna(1.0)
    capped = np.minimum(stability, stability_threshold)
    return mean_importance * (1.0 - capped / stability_threshold)


def rfe_with_stability(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = 10,
    n_estimators: int = 1000,
    stability_threshold: float = 0.3,
    progress_path: Path | None = None,
    random_state: int = 42,
) -> RFEResult:
    """Run stability-weighted RFE end-to-end.

    Args:
        X: Predictor matrix (rows = samples, columns = features).
        y: Response vector.
        groups: Group label per row for ``GroupKFold`` (e.g., ``GAGE_ID``).
        n_splits: Number of CV folds.
        n_estimators: Trees per QRF.
        stability_threshold: Cap on the ``std/mean`` ratio used in the score.
        progress_path: Optional JSON path for resume after interruption.
        random_state: Seed.

    Returns:
        :class:`RFEResult` with a history DataFrame and the optimal feature set
        (chosen by maximum validation adjusted R²).
    """
    from quantile_forest import RandomForestQuantileRegressor
    from sklearn.model_selection import GroupKFold

    from floodbhm.eval.metrics import adjusted_r_squared, smape

    removed: list[str] = []
    if progress_path is not None and progress_path.exists():
        manifest = json.loads(progress_path.read_text())
        removed = [f for f in manifest.get("removed_features", []) if f in X.columns]

    X_curr = X.drop(columns=removed)
    history_rows: list[dict] = []

    gkf = GroupKFold(n_splits=n_splits)

    while X_curr.shape[1] >= 1:
        n_features = X_curr.shape[1]
        fold_train: list[dict] = []
        fold_valid: list[dict] = []
        fold_importances: list[pd.Series] = []

        for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(X_curr, groups=groups), 1):
            X_tr = X_curr.iloc[tr_idx]
            X_va = X_curr.iloc[va_idx]
            y_tr = y.iloc[tr_idx]
            y_va = y.iloc[va_idx]

            model = RandomForestQuantileRegressor(
                n_estimators=n_estimators,
                max_features="sqrt",
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)

            # Always request median explicitly to be version-stable.
            y_pred_tr = model.predict(X_tr, quantiles=[0.5]).squeeze()
            y_pred_va = model.predict(X_va, quantiles=[0.5]).squeeze()

            fold_train.append(
                {
                    "smape": smape(y_tr, y_pred_tr),
                    "adj_r2": adjusted_r_squared(y_tr, y_pred_tr, n_features),
                }
            )
            fold_valid.append(
                {
                    "smape": smape(y_va, y_pred_va),
                    "adj_r2": adjusted_r_squared(y_va, y_pred_va, n_features),
                }
            )
            fold_importances.append(pd.Series(model.feature_importances_, index=X_curr.columns))

        mean_train_adj = float(np.mean([m["adj_r2"] for m in fold_train]))
        mean_valid_adj = float(np.mean([m["adj_r2"] for m in fold_valid]))
        mean_train_smape = float(np.mean([m["smape"] for m in fold_train]))
        mean_valid_smape = float(np.mean([m["smape"] for m in fold_valid]))

        imp_df = pd.DataFrame(fold_importances)
        mean_imp = imp_df.mean()
        std_imp = imp_df.std()
        score = _composite_score(mean_imp, std_imp, stability_threshold)

        feat_to_remove = str(score.idxmin())
        history_rows.append(
            {
                "n_features": n_features,
                "mean_valid_adj_r2": mean_valid_adj,
                "mean_valid_smape": mean_valid_smape,
                "mean_train_adj_r2": mean_train_adj,
                "mean_train_smape": mean_train_smape,
                "removed_feature": feat_to_remove,
                "removed_mean_importance": float(mean_imp[feat_to_remove]),
                "removed_stability": float(std_imp[feat_to_remove] / max(mean_imp[feat_to_remove], 1e-10)),
            }
        )

        removed.append(feat_to_remove)
        X_curr = X_curr.drop(columns=[feat_to_remove])

        if progress_path is not None:
            progress_path.write_text(json.dumps({"removed_features": removed}, indent=2))

    history = pd.DataFrame(history_rows)
    # Optimal is the iteration with the highest mean validation adjusted R²
    optimal_row = history.loc[history["mean_valid_adj_r2"].idxmax()]
    optimal_n = int(optimal_row["n_features"])
    # Features still present at that iteration = all minus features removed
    # in earlier iterations
    removed_before_optimum = history.loc[
        history["n_features"] > optimal_n, "removed_feature"
    ].tolist()
    optimal_features = [f for f in X.columns if f not in removed_before_optimum]

    return RFEResult(
        history=history,
        optimal_n_features=optimal_n,
        optimal_features=optimal_features,
    )
