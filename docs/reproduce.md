# Reproduce

This page describes how to reproduce every figure and table in the paper.

## Computational requirements

| Stage | CPU | GPU | RAM | Wall time |
|---|---|---|---|---|
| RFE | 48 cores | — | 16 GB | ~30 min |
| BHM | 4 cores | — | 32 GB | ~3 hours |
| GP | 8 cores | 1 GPU (24 GB) | 16 GB | ~1 hour |
| QRF stacking | 48 cores | — | 16 GB | ~15 min |
| Figures | 1 core | — | 8 GB | ~10 min |

End-to-end on a single HPC node: ~4-5 hours. Without a GPU, the GP stage falls back to
CPU and runs ~6 hours instead of 1.

## Option A — quickstart (5 minutes)

A toy example with 50 gauges is included in the repo for sanity checking:

```bash
uv sync --extra dev
uv run python examples/quickstart.py
```

This runs the full pipeline on a small subset and produces three preview figures in
`results/quickstart/`.

## Option B — full reproduction

### Step 1 — Install

```bash
git clone https://github.com/kwanhyuckkim/simpsons-paradox-floods.git
cd simpsons-paradox-floods
uv sync --extra dev
```

### Step 2 — Download cached data

```bash
uv run python scripts/download_data.py
```

This downloads the pre-computed parquet bundle from Zenodo and verifies SHA256 hashes
against `src/floodbhm/data/manifests.py`. ~330 MB total.

### Step 3 — RFE (optional, results cached)

```bash
uv run python scripts/run_rfe.py
```

Skip this if you trust the released `rfe_results_final.csv`.

### Step 4 — BHM fit (HPC recommended)

```bash
# Local (slow):
uv run python scripts/run_bhm.py model=bhm_4tuple prior=weak sampler=nuts_blackjax

# HPC (fast):
sbatch scripts/slurm/bhm.sbatch
```

The cached posterior `bhm_BHM_Category_2000draws_4chains.nc` is also released on
Zenodo if you prefer to skip re-fitting.

### Step 5 — GP residuals

```bash
uv run python scripts/run_gp.py
```

### Step 6 — QRF stacking + figures

```bash
uv run python scripts/run_qrf_stack.py
uv run python scripts/make_figures.py
```

Figures land in `results/figures/`.

## Verifying reproduction

Each major figure has a deterministic checksum recorded in
`tests/golden/figure_checksums.json`. Run:

```bash
uv run pytest tests/test_reproduction.py -v
```

to compare your locally generated figures against the released ones.

## Reproducibility guarantees

- All random seeds are fixed: `RAND_SEED=42` for numpy, sklearn, PyTorch.
- BHM uses `random_seed=100` in `bambi.fit(...)` per chain.
- GP uses `torch.manual_seed(42)` before each kernel fit.
- All Hydra configs are logged to `outputs/<date>/<time>/.hydra/config.yaml`.
- Each notebook stamps the git SHA and Python version in its first cell.

## Troubleshooting

**BHM divergent transitions appear**: increase `target_accept` to 0.995, enable
non-centered parameterization explicitly in Bambi.

**GP fails on CPU with OOM**: reduce `inducing_points` from 500 to 250 in
`configs/model/gp_sparse.yaml`.

**`exactextract` zonal stats slow**: the parquet downloaded from Zenodo already
contains the precomputed zonal stats. Re-running the zonal stats from raw rasters is
~12 hours and is not part of the standard reproduction path.
