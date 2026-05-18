"""Posterior predictive checks for Bambi models.

A thin wrapper around ``arviz.plot_ppc`` and a Bayesian p-value computation
that we report alongside R-hat / ESS in every BHM fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import arviz as az

__all__ = ["bayesian_p_value", "plot_ppc"]


def bayesian_p_value(idata: az.InferenceData, var_name: str = "streamflow") -> float:
    """Bayesian p-value for the discrepancy statistic ``T(y) = var(y)``.

    A value near 0.5 indicates good calibration. Values near 0 or 1 signal
    systematic over- or under-dispersion in the posterior predictive.

    Args:
        idata: Posterior InferenceData with ``posterior_predictive`` group.
        var_name: Variable name in the posterior predictive.

    Returns:
        Bayesian p-value in [0, 1].
    """
    pp = idata.posterior_predictive[var_name].values  # (chain, draw, n_obs)
    obs = idata.observed_data[var_name].values
    t_obs = float(np.var(obs))
    pp_flat = pp.reshape(-1, pp.shape[-1])
    t_rep = np.var(pp_flat, axis=1)
    return float(np.mean(t_rep >= t_obs))


def plot_ppc(idata: az.InferenceData, **kwargs) -> object:
    """Render arviz posterior predictive check plot.

    Thin wrapper for convenience; pass-through ``kwargs`` to ``az.plot_ppc``.
    """
    import arviz as az

    return az.plot_ppc(idata, **kwargs)
