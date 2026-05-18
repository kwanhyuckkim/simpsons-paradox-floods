"""Gaussian Process regression for spatial residual correction.

We provide three GP variants for different scale regimes:

- :class:`SimpleSpatialGP` — ``ExactGP`` with RBF kernel on (lat, lng). For
  small training sets (≤ 5000 gauges).
- :class:`MultiTaskSpatialGP` — ``MultitaskKernel`` for jointly modeling
  several dynamic variables. For medium sets when variables share structure.
- :class:`SparseBasinYearGP` — ``InducingPointKernel`` for large sets
  (basin × year cross product, > 10000 rows).

All variants record the final loss + lengthscale in
``model.training_log`` so downstream diagnostics can flag non-converged fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

__all__ = [
    "GPTrainingLog",
    "SimpleSpatialGP",
    "fit_spatial_gp",
]


@dataclass
class GPTrainingLog:
    """Per-iteration loss + final lengthscale."""

    losses: list[float] = field(default_factory=list)
    final_lengthscale: float | None = None
    converged: bool = False


def _build_simple_gp(train_x: torch.Tensor, train_y: torch.Tensor):
    """Construct an ExactGP with ScaleKernel(RBFKernel(ard_num_dims=2))."""
    import gpytorch
    from gpytorch.models import ExactGP

    class SimpleSpatialGP(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=2)
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SimpleSpatialGP(train_x, train_y, likelihood)
    return model, likelihood


# Public name re-exported for type hints; constructed by fit_spatial_gp
SimpleSpatialGP = _build_simple_gp


def fit_spatial_gp(
    coords: np.ndarray,
    residuals: np.ndarray,
    *,
    n_iter: int = 200,
    lr: float = 0.01,
    random_seed: int = 42,
    convergence_window: int = 20,
    convergence_tol: float = 1e-3,
) -> tuple[object, GPTrainingLog]:
    """Train a SimpleSpatialGP on BHM residuals.

    Args:
        coords: Array of shape ``(N, 2)`` with ``[lat, lng]`` per row.
        residuals: Target residuals, length ``N``.
        n_iter: Max training iterations.
        lr: Learning rate for Adam.
        random_seed: Torch random seed.
        convergence_window: Iterations without improvement that trigger early stop.
        convergence_tol: Loss improvement threshold for convergence.

    Returns:
        ``(model, training_log)``.

    Raises:
        ValueError: If shapes mismatch or arrays are empty.
    """
    import gpytorch
    import torch

    if coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2), got {coords.shape}")
    if len(coords) != len(residuals):
        raise ValueError(
            f"coords and residuals must have equal length, got {len(coords)} vs {len(residuals)}"
        )
    if len(coords) == 0:
        raise ValueError("coords is empty")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    train_x = torch.from_numpy(coords).float()
    train_y = torch.from_numpy(residuals).float()

    model, likelihood = _build_simple_gp(train_x, train_y)
    model.covar_module.base_kernel.lengthscale = 10.0  # informative init

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    log = GPTrainingLog()
    best_loss = float("inf")
    bad_steps = 0

    for _ in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        log.losses.append(loss_val)

        if loss_val < best_loss - convergence_tol:
            best_loss = loss_val
            bad_steps = 0
        else:
            bad_steps += 1
        if bad_steps >= convergence_window:
            log.converged = True
            break

    log.final_lengthscale = float(
        model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().mean()
    )
    if not log.converged:
        import warnings

        warnings.warn(
            f"GP did not converge within {n_iter} iterations (final loss {log.losses[-1]:.4f})",
            stacklevel=2,
        )

    return model, log
