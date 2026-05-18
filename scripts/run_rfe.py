"""CLI entry point for stability-weighted RFE.

Usage:
    python scripts/run_rfe.py
    python scripts/run_rfe.py model.n_splits=5 data.path=path/to/peaks.parquet
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="rfe", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from floodbhm.models.rfe import rfe_with_stability

    df = pd.read_parquet(cfg.data.path)
    X = df[list(cfg.features.candidates)]
    y = df[cfg.target]
    groups = df[cfg.data.group_col]

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "rfe_progress.json"

    result = rfe_with_stability(
        X=X,
        y=y,
        groups=groups,
        n_splits=cfg.model.n_splits,
        n_estimators=cfg.model.n_estimators,
        stability_threshold=cfg.model.stability_threshold,
        progress_path=progress_path,
        random_state=cfg.model.random_state,
    )

    result.history.to_csv(out_dir / "rfe_history.csv", index=False)
    (out_dir / "optimal_features.txt").write_text("\n".join(result.optimal_features))
    print(f"Optimal feature count: {result.optimal_n_features}")
    print(f"Saved RFE history to {out_dir}")


if __name__ == "__main__":
    main()
