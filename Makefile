# Makefile for common project tasks.
# Run `make help` for the list of available targets.

PYTHON ?= python
UV ?= uv

.PHONY: help install dev test lint format type-check clean reproduce docs serve-docs notebooks

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
		awk -F':.*?##' 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install package + runtime dependencies
	$(UV) sync

dev:  ## Install package with dev dependencies
	$(UV) sync --extra dev --extra docs
	$(UV) run pre-commit install

test:  ## Run pytest test suite
	$(UV) run pytest -v --cov=floodbhm --cov-report=term-missing

lint:  ## Run ruff lint
	$(UV) run ruff check src/ tests/ scripts/

format:  ## Auto-format code
	$(UV) run ruff format src/ tests/ scripts/
	$(UV) run ruff check --fix src/ tests/ scripts/

type-check:  ## Run mypy
	$(UV) run mypy src/

clean:  ## Remove build/cache artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

reproduce:  ## Reproduce all paper figures end-to-end (requires data download)
	@echo "Step 1/5: download external data (see docs/data.md)"
	$(UV) run python scripts/download_data.py
	@echo "Step 2/5: run RFE feature selection"
	$(UV) run python scripts/run_rfe.py
	@echo "Step 3/5: fit BHM (this requires HPC or ~4 GPU-hours)"
	$(UV) run python scripts/run_bhm.py
	@echo "Step 4/5: fit GP residuals"
	$(UV) run python scripts/run_gp.py
	@echo "Step 5/5: QRF stacking + figures"
	$(UV) run python scripts/run_qrf_stack.py
	$(UV) run python scripts/make_figures.py

docs:  ## Build documentation
	$(UV) run mkdocs build --strict

serve-docs:  ## Serve documentation locally
	$(UV) run mkdocs serve

notebooks:  ## Execute all demo notebooks (smoke test)
	$(UV) run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
