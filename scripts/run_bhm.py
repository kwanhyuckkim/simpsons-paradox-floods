"""CLI entry point for BHM training with full diagnostics.

Usage:
    python scripts/run_bhm.py
    python scripts/run_bhm.py sampler=nuts_blackjax model=bhm_4tuple prior=weak

On HPC, prefer ``sbatch scripts/slurm/bhm.sbatch``.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="bhm", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from floodbhm.features.grouping import build_bhm_category, merge_small_groups_by_distance
    from floodbhm.models.bhm import BHMSpec, build_bambi_model, default_priors, fit_with_diagnostics

    df = pd.read_parquet(cfg.data.path)
    df = build_bhm_category(df)
    df = merge_small_groups_by_distance(
        df, min_count=cfg.model.min_group_count, max_radius_km=cfg.model.max_merge_radius_km
    )

    covariates = list(cfg.model.covariates)
    spec = BHMSpec(
        target=cfg.model.target,
        fixed_covariates=covariates,
        random_covariates=list(cfg.model.random_covariates),
        group="BHM_Category",
        family=cfg.model.family,
    )

    priors = default_priors(covariates)
    model = build_bambi_model(df, spec, priors=priors)

    idata = fit_with_diagnostics(
        model,
        draws=cfg.sampler.draws,
        tune=cfg.sampler.tune,
        chains=cfg.sampler.chains,
        cores=cfg.sampler.cores,
        target_accept=cfg.sampler.target_accept,
        nuts_sampler=cfg.sampler.nuts_sampler,
        backend=cfg.sampler.backend,
        random_seed=cfg.sampler.random_seed,
        raise_on_failure=cfg.diagnostics.raise_on_failure,
    )

    out_path = Path(cfg.output.path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(out_path)
    print(f"Posterior saved to {out_path}")


if __name__ == "__main__":
    main()
