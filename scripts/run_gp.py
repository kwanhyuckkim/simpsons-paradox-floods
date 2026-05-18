"""CLI entry point for spatial GP residual modeling.

For each dynamic covariate, fits a SimpleSpatialGP on BHM residuals and writes
the resulting ``{var}_spatial_gp`` feature column.

Usage:
    python scripts/run_gp.py
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="gp", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from floodbhm.models.gp import fit_spatial_gp

    df = pd.read_parquet(cfg.data.path)
    coords = df[[cfg.coords.lat, cfg.coords.lng]].to_numpy()

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_columns: dict[str, np.ndarray] = {}
    for var in cfg.gp.variables:
        residuals = df[f"residual_{var}"].to_numpy()
        model, log = fit_spatial_gp(
            coords=coords,
            residuals=residuals,
            n_iter=cfg.gp.n_iter,
            lr=cfg.gp.learning_rate,
            random_seed=cfg.gp.random_seed,
        )
        import torch

        with torch.no_grad():
            pred = model(torch.from_numpy(coords).float()).mean.numpy()
        new_columns[f"{var}_spatial_gp"] = pred

        log_path = out_dir / f"gp_{var}_log.json"
        log_path.write_text(
            __import__("json").dumps(
                {
                    "var": var,
                    "final_loss": log.losses[-1] if log.losses else None,
                    "final_lengthscale": log.final_lengthscale,
                    "converged": log.converged,
                    "n_iter": len(log.losses),
                }
            )
        )

    augmented = df.copy()
    for k, v in new_columns.items():
        augmented[k] = v

    augmented.to_parquet(out_dir / "gp_features.parquet")
    print(f"GP features written to {out_dir / 'gp_features.parquet'}")


if __name__ == "__main__":
    main()
