# Ungauged Flood Prediction with Hierarchical Bayesian Stacking

**Hierarchical Bayesian Model → Gaussian Process → Quantile Random Forest** stacking pipeline that learns how impervious surface area shapes annual peak streamflow once the data is split into hydrologically meaningful clusters — exposing the within-group elasticity that complete pooling hides through Simpson's paradox.

---

**Status**: research code, methodology preview (work in progress)
**License**: MIT
**Python**: 3.11+
**Documentation**: https://kwanhyuckkim.github.io/simpsons-paradox-floods/

---

## TL;DR

- **Problem** — Predict the annual peak streamflow for watersheds where no gauge is installed, using watershed characteristics, land cover, and meteorology alone. The relationship between impervious surface area (ISA) and flood peaks is the core scientific interest.
- **Twist** — A single regression over all gauges returns a counterintuitive sign for the ISA effect. The fix is not a bigger model; it is a model that respects between-gauge structure.
- **Approach** — A four-stage pipeline that selects features by stability, partial-pools the ISA effect across hydrologic clusters, captures leftover spatial structure with Gaussian processes, and finally stacks everything into a Quantile Random Forest for calibrated prediction intervals.

## Quick start

```bash
# Install (Python 3.11+)
pip install -e .

# Or with conda
conda env create -f environment.yml
conda activate floodbhm

# Run a small synthetic demo (no data download needed)
python examples/quickstart.py

# Full reproduction (requires data download; see docs/data.md)
make reproduce
```

## Project structure

| Path | Contents |
|---|---|
| `src/floodbhm/` | Installable Python package — data loaders, features, models, eval, viz |
| `scripts/` | CLI entry points: `run_rfe`, `run_bhm`, `run_gp`, `run_qrf_stack`, `make_figures`; SLURM template under `scripts/slurm/` |
| `configs/` | Hydra YAML configs for data / model / prior / sampler (no hardcoded paths anywhere) |
| `tests/` | pytest suite — metrics, time-of-concentration, grouping, smoke imports |
| `docs/` | mkdocs Material site — methodology, data, reproduce, API reference |
| `examples/quickstart.py` | Five-minute synthetic-data walkthrough |
| `notebooks/` | Demo notebooks (added incrementally; see `notebooks/README.md`) |
| `Makefile` | Common targets: `install`, `dev`, `test`, `lint`, `reproduce`, `docs` |

The package source layout follows modern Python conventions (`src/` layout, `pyproject.toml` with `hatch` + version from VCS, `uv` for installs).

---

## Why this approach

The simplest way to study how impervious surface area affects floods is to pool all watersheds, fit a regression, and read off the slope. When we do that here, the slope comes out **with the wrong sign for the majority of individual watersheds**. The watersheds with the most impervious surface tend to be different in many other respects — they are smaller, more urban, in different climate zones, sometimes regulated by dams — and those confounders dominate the pooled fit.

This is **Simpson's paradox**: the within-group association can be the opposite of the between-group association. Increasing model capacity does not fix it; the fix is structural. We need a model that lets each cluster of watersheds carry its own intercept and its own slope, while still borrowing strength from neighbors.

That structural change is the spine of this project. The other three stages exist to make the hierarchical model usable in practice: feature selection so the model is identifiable, spatial residual modeling to clean up what the hierarchy misses, and quantile stacking to turn point predictions into proper prediction intervals.

## Methodology

The pipeline has four stages. Each stage produces a cacheable artifact (parquet / NetCDF / pickle) so downstream stages do not need to re-fit upstream models.

### Stage 1 — Stability-weighted Recursive Feature Elimination

We start with about thirty candidate covariates: meteorology (precipitation, temperature, snow water equivalent, solar radiation, daylength, aridity index), watershed shape (area, compactness, stream density, basin slope, elevation, latitude, longitude), soil (clay / silt / sand / organic matter / available water content / rock depth), and human disturbance (NID dam storage, maximum release discharge).

Plain backward elimination by importance is unstable — a feature that scores high in one fold can score low in another, and the elimination order matters. We weight each feature's mean importance by its cross-fold stability:

$$
c_j = \mu_j \cdot \left(1 - \min(s_j, 0.3) / 0.3\right),
\qquad s_j = \sigma_j / \mu_j
$$

where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of feature $j$'s MDI importance across ten `GroupKFold(GAGE_ID)` folds. The feature with the smallest composite score is dropped, and we repeat until one feature remains. The optimal feature count is the iteration with the highest validation adjusted $R^2$. Implementation: [`src/floodbhm/models/rfe.py`](src/floodbhm/models/rfe.py).

### Stage 2 — Bayesian Hierarchical Model

We split each watershed into a four-tuple cluster label,

$$
\text{BHM\_Category} = (\text{Natural/Managed}) \times \text{HUC02} \times \text{Köppen} \times \text{area-bin},
$$

so that watersheds in the same cluster share basin type, hydrologic region, climate, and scale. Clusters with fewer than ten members are merged with the nearest larger cluster by haversine distance on gauge centroids ([`features/grouping.py`](src/floodbhm/features/grouping.py)), so that partial pooling has enough data per cell.

The model itself is written in Bambi's R-style formula language:

```
streamflow_specific ~ 1 + sum(fixed_covariates)
                       + (1 + sum(random_covariates) | BHM_Category)
```

Each fixed covariate gets a population-level slope; the variables most relevant to within-cluster heterogeneity (precipitation, ISA, antecedent moisture, cumulative precipitation, aridity index) additionally get cluster-specific slopes. This is what "partial pooling" means in practice — the cluster slopes are pulled toward the population mean, and the strength of that shrinkage is learned from the data, not chosen by hand.

**Priors are weakly informative on standardized covariates**:

| Parameter | Prior |
|---|---|
| Fixed-effect coefficients | $\mathrm{Normal}(0, 2.5)$ |
| Response noise $\sigma$ | $\mathrm{HalfNormal}(1.0)$ — **positive support** |
| Random-intercept scale | $\mathrm{HalfNormal}(1.0)$ |
| Random-slope scale | $\mathrm{HalfNormal}(1.0)$ |

The HalfNormal on $\sigma$ is important and easy to get wrong; a Normal prior on a scale parameter places half its mass on negative numbers, which silently breaks the model. Implementation: [`models/bhm.py::default_priors`](src/floodbhm/models/bhm.py).

**Sampler**: blackjax NUTS via Bayeux, four chains, two thousand draws each, `target_accept = 0.99` to keep step sizes small enough for the hierarchical funnel. We rely on Bambi's non-centered parameterization to keep the geometry well-behaved.

### Stage 3 — Spatial Gaussian Process on residuals

Hierarchical pooling captures the structure that follows our four chosen group axes, but spatial autocorrelation will remain in the residuals — nearby gauges respond similarly to precipitation in ways the cluster label cannot encode. For each *dynamic* covariate (precipitation, temperature, daylength, solar radiation, vapor pressure, snow water equivalent, antecedent moisture, cumulative precipitation, ISA, aridity index) we:

1. Compute the BHM residual at every training gauge,
2. Fit a GPyTorch `ExactGP` with `ScaleKernel(RBFKernel(ard_num_dims=2))` on the `(LAT_GAGE, LNG_GAGE)` coordinates,
3. Predict the smoothed residual at every train and test gauge,
4. Store the prediction as a new column `{var}_spatial_gp`.

The result is a small set of geographically smoothed correction features rather than one big spatial map of residuals. Each kernel has its own learned length scale; training records the final loss and length scale so non-converged fits can be flagged downstream. For training sets above ten thousand rows we switch to `InducingPointKernel` with five hundred inducing points. Implementation: [`models/gp.py`](src/floodbhm/models/gp.py).

### Stage 4 — Quantile Random Forest stacking

The BHM gives us posterior predictive means, but downstream users care about prediction intervals. We materialize the posterior into per-cluster slope features (`Beta_<cov>` columns, one per group × covariate) and concatenate them with the GP residual features and the original RFE-selected covariates. The full feature matrix then goes into a `RandomForestQuantileRegressor` that produces quantile estimates at $\{0.05, 0.25, 0.5, 0.75, 0.95\}$. Implementation: [`models/stacking.py`](src/floodbhm/models/stacking.py).

The reason for this final layer, rather than reporting BHM posterior predictive quantiles directly, is calibration. The BHM's likelihood is Gaussian on a transformed scale and tends to be over-confident on extreme events; the QRF makes no distributional assumption and adapts the interval width to local data density.

### Validation discipline

Two layers of evaluation, in this order:

1. **Held-out gauges by stratified area split.** We bin gauges into ten area quantiles (`pd.qcut`), split 80 / 20 within each bin on unique area values, and use the same partition throughout the pipeline. Cross-validation inside the training set is `GroupKFold(groups=GAGE_ID)` so that no gauge contributes to both a training and a validation fold.
2. **Leave-one-cohort-out (LOCO).** For the Bayesian side, holding out gauges does not test generalization to truly unfamiliar regimes if other gauges in the same `BHM_Category` are in the training set. We additionally hold out *entire clusters* in turn and measure how predictions degrade.

Metrics: SMAPE, adjusted $R^2$, Nash-Sutcliffe efficiency, Kling-Gupta efficiency, percent bias for point estimates; PICP, NMPIW (IQR-normalized), Winkler interval score for prediction intervals. Implementations and unit tests: [`eval/metrics.py`](src/floodbhm/eval/metrics.py) and [`tests/test_metrics.py`](tests/test_metrics.py).

### Diagnostic standard

Every BHM fit is required to pass a fixed diagnostic checklist before any downstream analysis is computed:

- $\hat{R} < 1.01$ for every monitored parameter,
- $\mathrm{ESS}_{\text{bulk}} > 400$ and $\mathrm{ESS}_{\text{tail}} > 400$,
- Zero divergent transitions,
- $\mathrm{BFMI} > 0.3$ on every chain,
- Posterior predictive p-value on the discrepancy statistic $T(y) = \mathrm{Var}(y)$ in a plausible range.

The check is wrapped in a single `DiagnosticsReport` object; CI flags any regression. Implementation: [`eval/posterior_diagnostics.py`](src/floodbhm/eval/posterior_diagnostics.py).

---

## Data

| Dataset | Source | License |
|---|---|---|
| GAGES-II watershed boundaries + 18 basin characteristics | USGS | Public domain |
| NWIS daily streamflow (`00060_Mean`), 1985-2019 | USGS via `dataretrieval` | Public domain |
| NLCD impervious surface (9 epochs, 30 m) | USGS / MRLC | Public domain |
| PRISM precipitation + mean temperature | Oregon State PRISM Climate Group | Free for non-commercial research |
| Daymet `tmin`, `tmax`, `srad`, `vp`, `swe`, `prcp` | ORNL DAAC | Public domain |
| gISA (global Impervious Surface Area, 1985-2018) | Liu et al. (2020) | CC-BY 4.0 |
| Köppen-Geiger classification (30 classes, ~1 km) | Beck et al. (2018) | CC-BY 4.0 |
| NID dam inventory | USACE | Public domain |

Raw data is not stored in git. The pre-computed zonal-statistics parquet bundle will be released on Zenodo with SHA256-verified download via `scripts/download_data.py`. See [`docs/data.md`](docs/data.md) for the variable dictionary.

## Reproducibility

End-to-end reproduction via `make reproduce`. Step-by-step instructions: [`docs/reproduce.md`](docs/reproduce.md).

- Hydra config-driven — every hyperparameter lives in `configs/`, no hardcoded absolute paths anywhere in the package.
- All random seeds fixed (`RAND_SEED = 42` for numpy / sklearn / PyTorch; `random_seed = 100` for Bambi chains).
- BHM posteriors will be cached on Zenodo with SHA256 hashes; downstream stages verify before using.
- `pre-commit` hooks block notebook outputs and large files from entering history.
- GitHub Actions CI re-runs `tests/` on every push and pull request across Python 3.11 and 3.12.

## License

MIT License — see [`LICENSE`](LICENSE).

## Acknowledgments

This work is being developed at the UMass Amherst Department of Civil and Environmental Engineering. Computation is supported by the Unity HPC cluster. The implementation builds on `bambi`, `blackjax`, `bayeux`, `gpytorch`, `quantile-forest`, `arviz`, and `pymc`, and is informed by the NeuralHydrology benchmark of Kratzert et al. (2019).

## Contact

kwanhyuckkim@umass.edu
