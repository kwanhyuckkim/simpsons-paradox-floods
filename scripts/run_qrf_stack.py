"""CLI entry point for the final QRF stacking layer.

Loads BHM β-features and GP spatial features, trains the stacked QRF, evaluates
on test set, writes predictions + prediction-interval quality metrics.

Usage:
    python scripts/run_qrf_stack.py
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="stack", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from floodbhm.eval.metrics import (
        adjusted_r_squared,
        nash_sutcliffe_efficiency,
        nmpiw_iqr,
        pbias,
        picp,
        smape,
        winkler_interval_score,
    )
    from floodbhm.models.stacking import (
        build_stacked_features,
        extract_bhm_coefficients,
        fit_qrf_stack,
    )

    df = pd.read_parquet(cfg.data.path)
    idata = az.from_netcdf(cfg.bhm.posterior_path)

    bhm_coefs = extract_bhm_coefficients(idata, covariates=list(cfg.model.covariates))
    X = build_stacked_features(df, bhm_coefs, group_col="BHM_Category")

    train_mask = df[cfg.data.split_col] == "train"
    test_mask = df[cfg.data.split_col] == "test"

    feature_cols = [c for c in X.columns if c != cfg.model.target]
    X_train = X.loc[train_mask, feature_cols]
    y_train = X.loc[train_mask, cfg.model.target]
    X_test = X.loc[test_mask, feature_cols]
    y_test = X.loc[test_mask, cfg.model.target]

    qrf = fit_qrf_stack(
        X_train,
        y_train,
        quantiles=tuple(cfg.model.quantiles),
        n_estimators=cfg.model.n_estimators,
        random_state=cfg.model.random_state,
    )

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(qrf, out_dir / "qrf_stack.joblib")

    preds = qrf.predict(X_test, quantiles=list(cfg.model.quantiles))
    median_idx = list(cfg.model.quantiles).index(0.5)
    lo_idx = list(cfg.model.quantiles).index(0.05)
    hi_idx = list(cfg.model.quantiles).index(0.95)
    y_med = preds[:, median_idx]
    y_lo = preds[:, lo_idx]
    y_hi = preds[:, hi_idx]

    n_features = X_train.shape[1]
    metrics_summary = {
        "smape": smape(y_test, y_med),
        "adj_r2": adjusted_r_squared(y_test, y_med, n_features=n_features),
        "nse": nash_sutcliffe_efficiency(y_test, y_med),
        "pbias": pbias(y_test, y_med),
        "picp_90": picp(y_test, y_lo, y_hi),
        "nmpiw_iqr_90": nmpiw_iqr(y_test, y_lo, y_hi),
        "winkler_is_90": winkler_interval_score(y_test, y_lo, y_hi, alpha=0.10),
    }
    pd.DataFrame([metrics_summary]).to_csv(out_dir / "test_metrics.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "y_obs": y_test.values,
            "y_pred_median": y_med,
            "y_pred_05": y_lo,
            "y_pred_95": y_hi,
        }
    )
    pred_df.to_parquet(out_dir / "test_predictions.parquet")

    print("Test metrics:")
    for k, v in metrics_summary.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
