#!/bin/bash
#SBATCH --job-name=bhm-fit
#SBATCH --output=logs/bhm_%j.out
#SBATCH --error=logs/bhm_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=cpu

set -euo pipefail

# Activate environment (adjust to your HPC's module system)
# module load conda
# conda activate floodbhm

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

cd "$SLURM_SUBMIT_DIR"

uv run python scripts/run_bhm.py \
    sampler.draws=2000 \
    sampler.tune=1000 \
    sampler.chains=4 \
    sampler.cores=4 \
    sampler.target_accept=0.99 \
    sampler.nuts_sampler=blackjax \
    diagnostics.raise_on_failure=false
