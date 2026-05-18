"""QRF stacking layer on top of BHM β-features and GP spatial features.

Given an InferenceData posterior and a set of trained spatial GPs, this module
materializes the per-group BHM beta coefficients as new feature columns and
trains a final :class:`RandomForestQuantileRegressor` on the augmented feature
matrix.

The stacking layer outputs predictions at quantiles
``[0.05, 0.25, 0.5, 0.75, 0.95]`` for prediction-interval coverage analysis
(PICP, NMPIW, Winkler IS_α).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import arviz as az

__all__ = ["BHMCoefficients", "build_stacked_features", "extract_bhm_coefficients", "fit_qrf_stack"]


@dataclass
class BHMCoefficients:
    """Group-specific random slopes extracted from a BHM posterior."""

    by_covariate: dict[str, pd.Series]
    by_group: pd.DataFrame

    def materialize(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """Return df with ``Beta_<cov>`` columns joined on ``group_col``."""
        beta_df = self.by_group.copy()
        out = df.merge(
            beta_df.add_prefix("Beta_").reset_index().rename(columns={"index": group_col}),
            on=group_col,
            how="left",
        )
        return out


def extract_bhm_coefficients(
    idata: az.InferenceData,
    covariates: Sequence[str],
    group_dim: str = "BHM_Category__factor_dim",
) -> BHMCoefficients:
    """Extract posterior-mean group-specific slopes from a Bambi InferenceData.

    Args:
        idata: Posterior with random-slope variables named ``<cov>|<group>``.
        covariates: Covariates whose group-varying slopes to extract.
        group_dim: Name of the group dimension in the posterior.

    Returns:
        :class:`BHMCoefficients`.
    """
    by_cov: dict[str, pd.Series] = {}
    for cov in covariates:
        var_name = f"{cov}|{group_dim.split('__')[0]}"
        try:
            samples = idata.posterior[var_name].mean(dim=["chain", "draw"])
        except KeyError:
            continue
        s = pd.Series(samples.values, index=samples[group_dim].values, name=cov)
        by_cov[cov] = s

    by_group = pd.concat(by_cov, axis=1) if by_cov else pd.DataFrame()
    return BHMCoefficients(by_covariate=by_cov, by_group=by_group)


def build_stacked_features(
    df: pd.DataFrame,
    bhm_coefs: BHMCoefficients,
    spatial_gp_features: dict[str, np.ndarray] | None = None,
    *,
    group_col: str = "BHM_Category",
) -> pd.DataFrame:
    """Augment ``df`` with BHM β-features and GP spatial features.

    Args:
        df: Base feature matrix (must contain ``group_col``).
        bhm_coefs: Output of :func:`extract_bhm_coefficients`.
        spatial_gp_features: Mapping ``var_name -> array of length len(df)``.
        group_col: Column to merge on.

    Returns:
        Augmented DataFrame.
    """
    out = bhm_coefs.materialize(df, group_col)
    if spatial_gp_features:
        for name, values in spatial_gp_features.items():
            if len(values) != len(out):
                raise ValueError(
                    f"GP feature '{name}' length {len(values)} != df length {len(out)}"
                )
            out[f"{name}_spatial_gp"] = values
    return out


def fit_qrf_stack(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    n_estimators: int = 1000,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> Any:
    """Train the final stacking QRF.

    Args:
        X: Stacked feature matrix.
        y: Response vector.
        quantiles: Quantile levels to predict at evaluation time.
        n_estimators: Forest size.
        min_samples_leaf: Tree leaf size.
        random_state: Seed.

    Returns:
        A trained ``RandomForestQuantileRegressor`` with the ``quantiles_``
        attribute attached for downstream prediction at the requested levels.
    """
    from quantile_forest import RandomForestQuantileRegressor

    model = RandomForestQuantileRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=min_samples_leaf,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    model.quantiles_ = list(quantiles)
    return model
