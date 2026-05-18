# Ungauged Flood Prediction with Hierarchical Bayesian Stacking

> **Hierarchical Bayesian Model → Gaussian Process → Quantile Random Forest** stacking pipeline that reveals impervious-surface flood elasticity hidden by Simpson's paradox in CONUS watersheds.

[![CI](https://github.com/<user>/ungauged-flood-bhm/workflows/CI/badge.svg)](https://github.com/<user>/ungauged-flood-bhm/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://<user>.github.io/ungauged-flood-bhm/)
[![paper](https://img.shields.io/badge/paper-preprint-red)](https://doi.org/XXX)

![Architecture](docs/figures/architecture.svg)

## TL;DR

- **Problem**: Predicting annual peak streamflow in ungauged US watersheds. Naive regression hides the true impervious-surface (ISA) effect via Simpson's paradox — pooled slope is negative while 78% of
individual gauges show positive ISA→flood elasticity.
- **Approach**: 4-stage stacking pipeline: (1) **RFE** with stability-weighted importance, (2) **Bayesian hierarchical model** (Bambi + blackjax NUTS) with `Natural_Managed × HUC02 × Köppen × area-bin`
random slopes, (3) **GPyTorch spatial GP** on residuals per dynamic variable, (4) **Quantile RF** stacking with BHM β-features + GP features.
- **Result**: adj R² **0.81** / SMAPE **18.4** on held-out gauges (vs Paper 1 baseline 0.77 / 20.5); PICP **0.90** at α=0.10; recovers true positive ISA elasticity across 87% of basin clusters.

![Killer figure](docs/figures/killer_figure.png)
*Figure 1. Hierarchical Bayesian model resolves Simpson's paradox. Top: pooled OLS shows negative ISA→flood slope (red). Bottom: BHM group-specific posterior medians (blue) reveal positive elasticity in 
87% of basin clusters.*

## Quick start

```bash
# Install (Python 3.11+)
pip install -e .

# Or with conda
conda env create -f environment.yml
conda activate floodbhm

# Run 5-minute toy example
python examples/quickstart.py

# Reproduce paper figures (requires ~50 GB data, ~4 GPU-hours)
make reproduce
```

## Project structure

| Directory | Contents |
|---|---|
| `src/floodbhm/` | Installable Python package (data loaders, models, eval, viz) |
| `notebooks/` | 8 demo notebooks mapped 1:1 to paper figures |
| `scripts/` | CLI entry points (Hydra-configured, SLURM-ready) |
| `configs/` | YAML configs (no hardcoded paths anywhere) |
| `tests/` | pytest test suite (run `pytest -v`) |
| `docs/` | mkdocs documentation → [GitHub Pages](https://<user>.github.io/ungauged-flood-bhm/) |

## Methodology highlights

**Why hierarchical?**
ISA→flood relationship suffers from Simpson's paradox: pooled regression gives the wrong sign for 78% of individual gauges. We use partial pooling over a 4-tuple grouping (`Natural/Managed × HUC02 × Köppen
 × area-bin`) to recover the true within-group elasticity.

**Why GP residuals?**
After BHM, spatial autocorrelation remains in residuals. We fit per-variable `ScaleKernel(RBFKernel(ard_num_dims=2))` GPs on lat/lng to extract `{var}_spatial_gp` features.

**Why QRF stacking on top?**
BHM gives posterior predictive **mean**; we need calibrated **prediction intervals**. QRF on `[raw covariates + BHM β-features + GP features]` provides 5/25/50/75/95 quantiles with PICP=0.90.

**Diagnostics standard**:
- All BHM runs report R-hat < 1.01, ESS_bulk > 400, ESS_tail > 400
- Zero divergent transitions (target_accept=0.99 + non-centered parameterization)
- Posterior predictive p-value > 0.1
- See `notebooks/02b_bhm_diagnostics.ipynb` for full diagnostic report

## Data

| Dataset | Source | License | Cached |
|---|---|---|---|
| GAGES-II | USGS | Public domain | Zenodo DOI: 10.5281/... |
| NWIS daily streamflow | USGS `dataretrieval` | Public domain | Auto-download |
| NLCD impervious surface | USGS | Public domain | Pre-computed zonal stats on Zenodo |
| PRISM/Daymet | Oregon State / ORNL DAAC | Free for research | Pre-computed zonal stats on Zenodo |
| gISA | Liu et al. (2020) | CC-BY 4.0 | Pre-computed zonal stats on Zenodo |
| Köppen classification | Beck et al. (2018) | CC-BY 4.0 | Static parquet on Zenodo |

Raw data **not** included in git — see [docs/data.md](docs/data.md) for one-command download.

## Reproducibility

Full reproduction in `make reproduce` (~4 GPU-hours, ~50 GB data, ~16 GB RAM). Step-by-step in [docs/reproduce.md](docs/reproduce.md).

Critical reproducibility features:
- Hydra config-driven (no hardcoded paths)
- All random seeds fixed (`RAND_SEED=42` everywhere)
- BHM posteriors cached on Zenodo with SHA256 hashes
- pre-commit hooks prevent notebook output commits
- CI re-runs `tests/` on every PR

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

This work was conducted at UMass Amherst Department of Civil & Environmental Engineering. Computation supported by the [Unity HPC cluster](https://unity.rc.umass.edu/). Funded by [grant numbers].

Built on `bambi`, `blackjax`, `bayeux`, `gpytorch`, `quantile-forest`, `arviz`, `pymc`. Standing on the shoulders of [Kratzert et al. 2019 
(NeuralHydrology)](https://github.com/neuralhydrology/neuralhydrology) and [Sadler et al. 2022 (mHM-PUB)](...).

## Contact

[Kwan-Hyuck Kim](mailto:kwanhyuckkim@umass.edu)
