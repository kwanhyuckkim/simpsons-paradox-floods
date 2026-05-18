# Data

Raw data is **not** included in this git repository. Processed artifacts and cached
posteriors are released on Zenodo with permanent DOIs.

## Sources

| Dataset | Variable role | Resolution | License |
|---|---|---|---|
| **GAGES-II** | Watershed boundaries + 18 basin characteristics | Static | Public domain (USGS) |
| **USGS NWIS** | Daily streamflow `00060_Mean` 1985-2019 | 1 day | Public domain (USGS) |
| **NLCD** | Land cover 9 classes 1992-2019 (9 epochs) | 30 m | Public domain (USGS / MRLC) |
| **PRISM** | Daily precipitation + mean temperature | 4 km | Free for non-commercial research |
| **Daymet** | Daily `tmin`, `tmax`, `srad`, `vp`, `swe`, `prcp` | 1 km | Public domain (ORNL DAAC) |
| **gISA** | Global Impervious Surface Area 1985-2018 | 30 m | CC-BY 4.0 (Liu et al. 2020) |
| **Köppen-Geiger** | Climate classification (30 classes) | ~1 km | CC-BY 4.0 (Beck et al. 2018) |
| **NID** | Dam inventory (storage, max discharge, year completed) | Per-dam | Public domain (USACE) |

## Pre-computed artifacts (Zenodo)

To skip the ~50 GB raw download and ~12 hours of zonal-statistics preprocessing, we
publish two parquet bundles on Zenodo:

```
zenodo://<DOI>/
├── peaks_v2_category_ver2.parquet              # ~80 MB, primary input
├── rfe_results_final.csv                        # RFE rankings
├── bhm_BHM_Category_2000draws_4chains.nc        # BHM posterior (~250 MB)
├── gp_residuals_per_variable.parquet            # GP features
└── README.md                                    # data dictionary + SHA256 hashes
```

Download with:

```bash
uv run python scripts/download_data.py
```

This script verifies SHA256 hashes against the values recorded in
`src/floodbhm/data/manifests.py`.

## Data dictionary

### `peaks_v2_category_ver2.parquet`

One row per `(GAGE_ID, year)` pair (annual peak event).

| Column | Type | Units | Description |
|---|---|---|---|
| `GAGE_ID` | str | — | USGS site identifier (8-digit string) |
| `year` | int | — | Water year of peak event |
| `streamflow` | float | m³/s | Annual peak discharge |
| `date` | datetime | — | Date of peak |
| `prcp_mean` | float | mm | Peak-event basin-mean precipitation |
| `cum_ppt` | float | mm | 5-day antecedent cumulative precipitation |
| `amc` | float | mm | Antecedent moisture condition proxy |
| `T_c` | float | hours | Time of concentration (basin-area-binned formula) |
| `isa_mean` | float | fraction | Basin-mean impervious surface area |
| `mean_ai` | float | — | Aridity index (PET/P long-term mean) |
| `HUC02` | str | — | USGS Hydrologic Unit Code level 2 |
| `koppen_category` | str | — | Köppen-Geiger class label |
| `Natural_Managed` | str | `Natural`/`Managed` | NID dam-presence flag |
| `area_hydro_fine` | int | — | Area bin id (0-9) |
| `BHM_Category` | str | — | Composite 4-tuple grouping label |
| `AREA, BAS_COMPACTNESS, BFI_AVE, ...` | float | — | GAGES-II basin characteristics (18 columns) |

## Provenance

A full data lineage diagram is in `docs/figures/data_lineage.svg` (TODO). Hashing of
intermediate parquets is performed in `src/floodbhm/data/integrity.py`.

## Ethics and licensing

All datasets used are open access. We comply with USGS, ORNL DAAC, and Oregon State
PRISM acceptable use policies. No personally identifiable information is processed.

When redistributing the cached parquets, please cite the original data providers in
addition to this codebase.
