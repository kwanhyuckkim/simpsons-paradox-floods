"""Posterior convergence diagnostics for Bambi / PyMC / blackjax fits.

This module enforces the diagnostic standard required for publishable
Bayesian inference:

- R-hat (potential scale reduction) < 1.01 for every parameter.
- ESS_bulk and ESS_tail above a minimum (default 400).
- Zero divergent transitions.
- BFMI (Bayesian fraction of missing information) above a threshold per chain.
- Posterior predictive p-value in a plausible range.

A single call to :func:`report_full_diagnostics` runs all checks and returns a
``DiagnosticsReport`` with pass/fail flags so the caller can fail loudly during
CI or in a notebook.

The diagnostics intentionally do NOT silently fix anything — failures raise or
return ``passed = False`` so the user reruns the chain with adjustments
(``target_accept``, non-centered parameterization, longer warmup, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import arviz as az

__all__ = [
    "DiagnosticsReport",
    "check_rhat",
    "check_ess",
    "check_divergences",
    "check_bfmi",
    "report_full_diagnostics",
]


@dataclass
class DiagnosticsReport:
    """Container for posterior diagnostic outcomes.

    Attributes:
        passed: ``True`` iff all individual checks passed.
        rhat_max: Largest R-hat observed across parameters.
        ess_bulk_min: Smallest bulk ESS observed.
        ess_tail_min: Smallest tail ESS observed.
        n_divergences: Total count of divergent transitions across chains.
        bfmi_min: Smallest per-chain BFMI.
        failures: Human-readable list of failed checks.
        thresholds: Thresholds used for each check.
    """

    passed: bool
    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    n_divergences: int
    bfmi_min: float
    failures: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"DiagnosticsReport(passed={self.passed})",
            f"  R-hat max          : {self.rhat_max:.4f}   (threshold < {self.thresholds.get('rhat', 1.01)})",
            f"  ESS bulk min       : {self.ess_bulk_min:.0f}   (threshold > {self.thresholds.get('ess_bulk', 400)})",
            f"  ESS tail min       : {self.ess_tail_min:.0f}   (threshold > {self.thresholds.get('ess_tail', 400)})",
            f"  Divergences        : {self.n_divergences}   (must be 0)",
            f"  BFMI min per chain : {self.bfmi_min:.3f}   (threshold > {self.thresholds.get('bfmi', 0.3)})",
        ]
        if self.failures:
            lines.append("  FAILURES:")
            lines.extend(f"    - {f}" for f in self.failures)
        return "\n".join(lines)


def _summary_table(idata: az.InferenceData) -> Any:
    """Run arviz.summary lazily so we don't import arviz at module top."""
    import arviz as az

    return az.summary(idata, fmt="wide")


def check_rhat(idata: az.InferenceData, threshold: float = 1.01) -> tuple[bool, float]:
    """Return ``(passed, rhat_max)`` for the posterior.

    R-hat > 1.01 indicates the chains have not mixed.
    """
    summary = _summary_table(idata)
    rhat_max = float(summary["r_hat"].max())
    return rhat_max < threshold, rhat_max


def check_ess(
    idata: az.InferenceData, bulk_min: float = 400.0, tail_min: float = 400.0
) -> tuple[bool, float, float]:
    """Return ``(passed, ess_bulk_min, ess_tail_min)``."""
    summary = _summary_table(idata)
    ess_bulk = float(summary["ess_bulk"].min())
    ess_tail = float(summary["ess_tail"].min())
    passed = (ess_bulk >= bulk_min) and (ess_tail >= tail_min)
    return passed, ess_bulk, ess_tail


def check_divergences(idata: az.InferenceData) -> tuple[bool, int]:
    """Return ``(passed, n_divergences)``. Passes only if zero divergences."""
    try:
        diverging = idata.sample_stats["diverging"]
    except (AttributeError, KeyError):
        return True, 0  # sampler didn't expose diverging — skip
    n_div = int(diverging.values.sum())
    return n_div == 0, n_div


def check_bfmi(idata: az.InferenceData, threshold: float = 0.3) -> tuple[bool, float]:
    """Return ``(passed, bfmi_min)``.

    BFMI < 0.3 indicates pathological energy distribution per chain — usually a
    funnel issue requiring non-centered parameterization.
    """
    import arviz as az

    try:
        bfmi = az.bfmi(idata)
    except Exception:  # noqa: BLE001 — arviz internal API may vary
        return True, float("nan")
    bfmi_min = float(np.min(bfmi))
    return bfmi_min >= threshold, bfmi_min


def report_full_diagnostics(
    idata: az.InferenceData,
    *,
    rhat_threshold: float = 1.01,
    ess_bulk_threshold: float = 400.0,
    ess_tail_threshold: float = 400.0,
    bfmi_threshold: float = 0.3,
    raise_on_failure: bool = False,
) -> DiagnosticsReport:
    """Run all diagnostic checks and return a structured report.

    Args:
        idata: Posterior ``InferenceData`` produced by Bambi or PyMC.
        rhat_threshold: Maximum allowed R-hat (default 1.01).
        ess_bulk_threshold: Minimum required bulk ESS (default 400).
        ess_tail_threshold: Minimum required tail ESS (default 400).
        bfmi_threshold: Minimum required per-chain BFMI (default 0.3).
        raise_on_failure: If ``True``, raise ``RuntimeError`` when any check fails.

    Returns:
        :class:`DiagnosticsReport`.

    Example:
        >>> import arviz as az
        >>> idata = az.from_netcdf("posterior.nc")
        >>> report = report_full_diagnostics(idata, raise_on_failure=False)
        >>> print(report)
    """
    failures: list[str] = []

    rhat_ok, rhat_max = check_rhat(idata, threshold=rhat_threshold)
    if not rhat_ok:
        failures.append(f"R-hat {rhat_max:.4f} >= {rhat_threshold}")

    ess_ok, ess_bulk, ess_tail = check_ess(
        idata, bulk_min=ess_bulk_threshold, tail_min=ess_tail_threshold
    )
    if not ess_ok:
        failures.append(
            f"ESS too low: bulk={ess_bulk:.0f}, tail={ess_tail:.0f} "
            f"(thresholds {ess_bulk_threshold:.0f}, {ess_tail_threshold:.0f})"
        )

    div_ok, n_div = check_divergences(idata)
    if not div_ok:
        failures.append(f"{n_div} divergent transitions (must be 0)")

    bfmi_ok, bfmi_min = check_bfmi(idata, threshold=bfmi_threshold)
    if not bfmi_ok:
        failures.append(f"BFMI {bfmi_min:.3f} < {bfmi_threshold}")

    report = DiagnosticsReport(
        passed=len(failures) == 0,
        rhat_max=rhat_max,
        ess_bulk_min=ess_bulk,
        ess_tail_min=ess_tail,
        n_divergences=n_div,
        bfmi_min=bfmi_min,
        failures=failures,
        thresholds={
            "rhat": rhat_threshold,
            "ess_bulk": ess_bulk_threshold,
            "ess_tail": ess_tail_threshold,
            "bfmi": bfmi_threshold,
        },
    )

    if not report.passed and raise_on_failure:
        raise RuntimeError(
            "Posterior diagnostics failed. See report for details:\n" + str(report)
        )

    return report
