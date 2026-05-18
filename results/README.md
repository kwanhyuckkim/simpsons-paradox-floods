# Results

This directory holds derived artifacts produced by the pipeline. None of these
files are stored in git — they are reproduced by running the scripts.

Expected subdirectories after a full reproduction:

| Directory | Contents | Source script |
|---|---|---|
| `rfe/` | RFE history CSV, selected feature list | `scripts/run_rfe.py` |
| `posteriors/` | Bambi `.nc` inference data | `scripts/run_bhm.py` |
| `gp/` | Per-variable GP features parquet + training logs | `scripts/run_gp.py` |
| `stack/` | Stacked QRF model, test predictions, test metrics | `scripts/run_qrf_stack.py` |
| `figures/` | Paper figures (PNG + PDF) | `scripts/make_figures.py` |
| `quickstart/` | Toy example outputs | `examples/quickstart.py` |
