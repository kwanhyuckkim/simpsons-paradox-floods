# Notebooks

These notebooks reproduce every figure in the paper one-to-one. They consume
the package code in `src/floodbhm/` — they are *demonstrations*, not the
implementation. To re-run, install the package in editable mode
(`pip install -e .`) and execute notebooks in numeric order.

## Execution order

| Notebook | Topic | Estimated time | Key output |
|---|---|---|---|
| `00_data_overview.ipynb` | Study area map, EDA, Simpson's paradox preview | 5 min | `Fig1_studyarea.pdf` |
| `01_rfe.ipynb` | Stability-weighted RFE | 30 min | `rfe_history.csv`, selected feature set |
| `02_bhm_fit.ipynb` | Bambi + blackjax BHM training | (HPC: ~3 h) | Posterior `.nc` |
| `02b_bhm_diagnostics.ipynb` | R-hat, ESS, divergences, PPC | 5 min | Diagnostic report tables/plots |
| `03_gp_residuals.ipynb` | Spatial GP per variable | 1 h | `{var}_spatial_gp` features |
| `04_qrf_stacking.ipynb` | Final QRF on stacked features | 15 min | Test predictions + PI metrics |
| `05_simpsons_paradox.ipynb` | Killer figure of paper | 5 min | `Fig2_simpsons.pdf` |
| `06_ablation.ipynb` | BHM-only / GP-only / full stack | 30 min | Ablation table |
| `07_sota_comparison.ipynb` | LSTM (NeuralHydrology) baseline | 2 h | Comparison table |

## Conventions

- Each notebook begins by stamping the git SHA and Python version (`!git rev-parse HEAD`).
- Notebook outputs are stripped before commit (`nbstripout` pre-commit hook).
- All file paths come from `configs/data/*.yaml` (no hardcoded absolute paths).
- Long-running cells write artifacts to `results/` so re-execution is incremental.

## Status

Notebooks are added as they are ported from the original 32-MB development
notebook. The CI smoke-tests them with synthetic data fixtures.
