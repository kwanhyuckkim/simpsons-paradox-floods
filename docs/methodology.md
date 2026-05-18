# Methodology

The pipeline has four sequential stages. Each stage produces a cacheable artifact
(parquet, netCDF, or pickle) so downstream work does not re-fit upstream models.

## Stage 1 — Recursive Feature Elimination

A QRF-based backward-elimination procedure with a stability-weighted importance score.
At each iteration:

1. Fit `RandomForestQuantileRegressor(n_estimators=1000, max_features='sqrt')` on the
   current feature set under a 10-fold `GroupKFold(groups=GAGE_ID)`.
2. Compute per-feature mean MDI importance and across-fold standard deviation.
3. Define stability $s_j = \sigma_j / \mu_j$ and a composite score
   $c_j = \mu_j \cdot \left(1 - \min(s_j, 0.3) / 0.3\right)$.
4. Drop the feature with the lowest composite score.

The stability term penalizes high-variance importances and breaks ties when two
features have similar mean importance.

## Stage 2 — Bayesian Hierarchical Model

Bambi formula:

```
streamflow_specific ~ 1 + sum(covs_fixed)
                       + (1 + sum(covs_random) | BHM_Category)
```

where:

- `covs_fixed` and `covs_random` are subsets of the RFE survivors.
- `BHM_Category = Natural_Managed × HUC02 × Köppen × area_bin` (a 4-tuple). Small
  groups with fewer than 10 members are merged with the nearest larger group by
  BallTree haversine distance.
- `streamflow_specific = streamflow / AREA` (specific discharge, m/s).

### Priors (corrected from the original code)

```python
priors = {
    cov: bmb.Prior("Normal", mu=0, sigma=2.5)         # weakly informative
    for cov in covariates
}
priors["sigma"] = bmb.Prior("HalfNormal", sigma=1.0)  # positive support
priors["1|BHM_Category"] = bmb.Prior(
    "Normal", mu=0,
    sigma=bmb.Prior("HalfNormal", sigma=1.0),
)
```

The original code defined a `Normal(0, 10⁴)` sigma — this is replaced with
`HalfNormal(1)` to enforce positive support and reasonable scale.

### Sampler

`bambi.Model.fit(draws=2000, chains=4, cores=4, target_accept=0.99,
nuts_sampler="blackjax", backend="bayeux")`

Non-centered parameterization is requested via Bambi to avoid funnel pathologies.

### Diagnostics

Every BHM fit reports:

- `arviz.summary` with `r_hat < 1.01` for all parameters
- `ess_bulk > 400` and `ess_tail > 400`
- Zero divergent transitions
- BFMI > 0.3 per chain
- Posterior predictive check via `az.plot_ppc` with Bayesian p-value

See `src/floodbhm/eval/posterior_diagnostics.py`.

## Stage 3 — Gaussian Process residuals

For each dynamic covariate $v$ in `{prcp_mean, tmax_mean, tmin_mean, srad_mean, vp_mean,
swe_mean, amc, cum_ppt, isa_mean, mean_ai}`:

1. Compute BHM residuals at training gauge locations.
2. Fit a GPyTorch `ExactGP` with `ScaleKernel(RBFKernel(ard_num_dims=2))` on
   `(LAT_GAGE, LNG_GAGE)`.
3. Train for 200 iterations with `Adam(lr=0.01)`; report final negative marginal log
   likelihood and lengthscale.
4. Predict at every train and test gauge → new feature column `{v}_spatial_gp`.

For large training sets (>10k rows) we switch to `InducingPointKernel` with 500
inducing points (`SparseBasinYearGP`).

## Stage 4 — Quantile Random Forest stacking

`RandomForestQuantileRegressor(n_estimators=1000, max_features='sqrt',
min_samples_leaf=1, bootstrap=True)` trained on the union of:

- the RFE-selected raw covariates,
- the BHM β-features (one per group-covariate pair, materialized from the posterior),
- the GP `{v}_spatial_gp` features.

Predictions report quantiles `[0.05, 0.25, 0.5, 0.75, 0.95]` for prediction-interval
coverage (PICP, NMPIW, Winkler IS_α).

## Validation

Train/test split is by `unique AREA` within 10 area-quantile bins (80/20), preserving
gauge-disjoint sets. RFE and QRF CV use `GroupKFold(groups=GAGE_ID)`.

For Bayesian validation we additionally hold out one entire `BHM_Category` cluster at
a time (leave-one-cohort-out, LOCO) to assess generalization to truly ungauged
cohorts.

## Comparison baselines

The paper compares the full stack against:

- Plain OLS over `streamflow ~ sum(covs)` (complete pooling)
- BMLR (Bayesian Multilinear Regression, no random effects)
- Plain QRF (no BHM, no GP)
- LSTM trained with NeuralHydrology (Kratzert et al. 2019) on the same gauges

Ablation isolates the contribution of each stage by removing one component at a time.
