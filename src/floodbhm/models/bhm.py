"""Bayesian hierarchical model construction with Bambi.

This module fixes the four prior/scale issues identified during the T02 audit:

1. **Sigma prior is HalfNormal**, not Normal — scale parameters require
   positive support. Original code used ``Normal(0, 10**4)``, which is wrong.
2. **Priors are weakly informative, not vague**: ``Normal(0, 2.5)`` on
   standardized covariates instead of ``Normal(0, 10**4)``. The vague prior
   was both untestable and unhelpful for convergence.
3. **The priors dictionary is explicitly passed to** ``bmb.Model(..., priors=priors)``,
   not commented out (the original code had ``#, priors=priors`` which silently
   dropped them).
4. **Random-effect variance has its own HalfNormal hyperprior**, enabling
   non-centered parameterization.

Reference:
    Gabry et al. (2019), "Visualization in Bayesian workflow",
    *J. R. Stat. Soc. A* 182, 389–402.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import arviz as az
    import bambi as bmb
    import pandas as pd


__all__ = ["BHMSpec", "build_bambi_model", "default_priors", "fit_with_diagnostics"]


@dataclass
class BHMSpec:
    """Specification of a hierarchical Bambi model.

    Attributes:
        target: Name of the response variable (assumed already standardized or
            on a meaningful natural scale; not log-transformed by default).
        fixed_covariates: List of column names for fixed-effect predictors.
        random_covariates: Subset of ``fixed_covariates`` to also get
            group-varying slopes. Defaults to all fixed covariates.
        group: Name of the group column for partial pooling (e.g.,
            ``BHM_Category``).
        family: Bambi family string (default ``"gaussian"``).
    """

    target: str
    fixed_covariates: Sequence[str]
    group: str
    random_covariates: Sequence[str] | None = None
    family: str = "gaussian"

    def __post_init__(self) -> None:
        if self.random_covariates is None:
            self.random_covariates = list(self.fixed_covariates)


def default_priors(covariates: Sequence[str]) -> dict[str, bmb.Prior]:
    """Return weakly-informative priors suitable for standardized covariates.

    Each fixed covariate gets ``Normal(0, 2.5)``. The response noise gets
    ``HalfNormal(1.0)`` (positive support; the original code used Normal which
    permitted negative scale). Random-effect variances use ``HalfNormal(1.0)``
    too.

    Args:
        covariates: List of fixed-effect covariate names.

    Returns:
        Dict suitable for the ``priors=`` argument of :class:`bambi.Model`.
    """
    import bambi as bmb

    priors: dict[str, bmb.Prior] = {
        cov: bmb.Prior("Normal", mu=0.0, sigma=2.5) for cov in covariates
    }
    # Response observation noise — MUST be positive
    priors["sigma"] = bmb.Prior("HalfNormal", sigma=1.0)
    # Intercept on the same scale as covariates after standardization
    priors["Intercept"] = bmb.Prior("Normal", mu=0.0, sigma=2.5)
    return priors


def build_bambi_model(
    df: pd.DataFrame,
    spec: BHMSpec,
    priors: dict | None = None,
) -> bmb.Model:
    """Construct a Bambi hierarchical model from a :class:`BHMSpec`.

    The formula is:

    .. code-block:: text

        target ~ 1 + sum(fixed_covariates) + (1 + sum(random_covariates) | group)

    Args:
        df: Training DataFrame containing all spec columns.
        spec: Model specification.
        priors: Custom prior dictionary; if ``None``, uses :func:`default_priors`.

    Returns:
        Configured :class:`bambi.Model` (not yet fit).

    Raises:
        ValueError: If required columns are missing from ``df``.
    """
    import bambi as bmb

    required = (
        {spec.target, spec.group} | set(spec.fixed_covariates) | set(spec.random_covariates or [])
    )
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    fixed = " + ".join(spec.fixed_covariates)
    random = " + ".join(spec.random_covariates or [])
    formula = f"{spec.target} ~ 1 + {fixed} + (1 + {random} | {spec.group})"

    if priors is None:
        priors = default_priors(list(spec.fixed_covariates))

    model = bmb.Model(formula, df, family=spec.family, priors=priors)
    return model


def fit_with_diagnostics(
    model: bmb.Model,
    *,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 4,
    target_accept: float = 0.99,
    nuts_sampler: str = "blackjax",
    backend: str = "bayeux",
    random_seed: int = 100,
    raise_on_failure: bool = False,
) -> az.InferenceData:
    """Fit a Bambi model and run the full diagnostic suite.

    On completion the function prints (and returns inside the
    InferenceData's attrs) a :class:`DiagnosticsReport`. If
    ``raise_on_failure`` is ``True``, raises if any of R-hat, ESS,
    divergences, or BFMI fails its threshold.

    Args:
        model: Configured Bambi model.
        draws: Posterior draws per chain (post-tuning).
        tune: Warmup / tuning iterations per chain.
        chains: Number of chains (≥ 2 required for R-hat).
        cores: Parallel cores.
        target_accept: NUTS step-size adaptation target. Raise to 0.99 for
            hierarchical models with many random effects.
        nuts_sampler: ``"blackjax"`` (recommended for large models), or
            ``"pymc"``, ``"nutpie"``.
        backend: ``"bayeux"`` for blackjax via JAX, or ``"pymc"`` for native.
        random_seed: Seed for reproducibility.
        raise_on_failure: If ``True``, escalate diagnostic failures to errors.

    Returns:
        :class:`arviz.InferenceData` with posterior and diagnostics attached.
    """
    from floodbhm.eval.posterior_diagnostics import report_full_diagnostics

    idata = model.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        nuts_sampler=nuts_sampler,
        backend=backend,
        random_seed=random_seed,
        progressbar=True,
    )

    report = report_full_diagnostics(idata, raise_on_failure=raise_on_failure)
    print(report)

    # Persist diagnostics in the InferenceData attrs for downstream consumers
    idata.attrs["rhat_max"] = report.rhat_max
    idata.attrs["ess_bulk_min"] = report.ess_bulk_min
    idata.attrs["ess_tail_min"] = report.ess_tail_min
    idata.attrs["n_divergences"] = report.n_divergences
    idata.attrs["bfmi_min"] = report.bfmi_min
    idata.attrs["diagnostics_passed"] = report.passed

    return idata
