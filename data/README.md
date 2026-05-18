# Data

Raw and processed data files are not stored in this git repository. See
[`docs/data.md`](../docs/data.md) for the full list of datasets, licenses, and
the Zenodo DOI from which the pre-computed parquet bundle is downloaded.

Run:

```bash
uv run python scripts/download_data.py
```

to populate this directory with the cached parquet files. The script verifies
SHA256 hashes against the manifest in `src/floodbhm/data/manifests.py`.
