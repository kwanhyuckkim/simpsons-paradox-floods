# simpsons-paradox-floods

Predicting annual peak streamflow at ungauged watersheds, mainly to see how impervious surface (ISA) changes flood peaks.

When I pooled all the gauges into one regression, the ISA slope came out with the wrong sign for most of the individual watersheds. The watersheds with a lot of impervious surface are also smaller, more urban, in other climate zones, and sometimes regulated by dams, and the pooled fit picks up those differences instead (Simpson's paradox). So the core of this repo is a hierarchical model where each cluster of watersheds gets its own intercept and ISA slope, while the clusters still share information.

The other steps are there to make that model usable:

1. stability-weighted recursive feature elimination, so the model is identifiable
2. the Bayesian hierarchical model for the ISA effect
3. a spatial Gaussian process on the residuals, for what the hierarchy misses
4. quantile random forest stacking, to get prediction intervals

This is still a work in progress. I'm writing it up as a paper, so things will change.

## Running it

Python 3.11 or newer.

```bash
pip install -e .
# or
conda env create -f environment.yml
conda activate floodbhm

python examples/quickstart.py   # small synthetic example, no download
```

The quickstart writes `results/quickstart/simpsons_demo.png`.

For the real data each step is its own script: `scripts/run_rfe.py`, `run_bhm.py`, `run_gp.py`, `run_qrf_stack.py`, then `make_figures.py`. Settings are in `configs/` (Hydra), and `scripts/slurm/` has the batch template I use on the cluster.

## Data

The raw data is not in the repo. I use:

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

I plan to put the processed zonal-statistics files on Zenodo, and `scripts/download_data.py` will pull them once they are up. They are not up yet.

## Credits

MIT license. I built this in the Civil and Environmental Engineering department at UMass Amherst and ran it on the Unity cluster. It leans on bambi, blackjax, bayeux, gpytorch, quantile-forest, arviz and pymc, and parts of the setup were informed by the NeuralHydrology benchmark (Kratzert et al., 2019).
