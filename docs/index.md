# Simpson's Paradox Floods

**Hierarchical Bayesian Model → Gaussian Process → Quantile Random Forest** stacking
pipeline that reveals impervious-surface flood elasticity hidden by Simpson's paradox
in CONUS watersheds.

## What this is

A reproducible research codebase accompanying the preprint
*"Hierarchical Bayesian resolution of Simpson's paradox in impervious-surface flood
elasticity across 600+ US watersheds"* (DOI TBD).

The pipeline answers the question: when impervious surface area (ISA) grows in a
watershed, do flood peaks grow with it? Naive regression over hundreds of US gauges
returns a counterintuitive negative association — even though 78% of individual gauges
show positive elasticity. The remedy is hierarchical pooling that respects the
between-gauge heterogeneity in basin type, hydrologic region, climate, and area.

## What this gives you

- **A clean Python package** (`floodbhm`) for the 4-stage stacking pipeline.
- **Eight demo notebooks** mapped 1:1 to the figures of the paper.
- **CLI scripts** ready for HPC SLURM submission.
- **Hydra configs** for all hyperparameters (no hardcoded paths anywhere).
- **A diagnostic standard**: every BHM run reports R-hat, ESS, divergence counts,
  posterior predictive checks.
- **An open data trail**: every cached posterior + processed parquet on Zenodo with
  SHA256 hashes.

## Quick links

- [Methodology](methodology.md) — model architecture, priors, samplers
- [Data](data.md) — sources, licenses, download instructions
- [Reproduce](reproduce.md) — step-by-step replication
- [API Reference](api.md) — auto-generated function docs

## License

MIT. See [`LICENSE`](https://github.com/kwanhyuckkim/simpsons-paradox-floods/blob/main/LICENSE).

## Contact

Kwan-Hyuck Kim, Department of Civil and Environmental Engineering, UMass Amherst.
kwanhyuckkim@umass.edu
