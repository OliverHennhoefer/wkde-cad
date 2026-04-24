# Between Resolution Collapse and Variance Inflation: Stabilizing Weighted Conformal Anomaly Detection

## Overview

This repository contains the experimental code for weighted conformal anomaly
detection using [nonconform](https://github.com/OliverHennhoefer/nonconform).
It supports:

- Empirical conformal detection
- Randomized empirical conformal detection
- Probabilistic conformal detection
- Weighted variants of the above

## Quick Start (fresh machine, no `.venv`)

### Prerequisites

- Python `3.12+` (required by `pyproject.toml`)
- `uv` installed

### Setup and run

```bash
# From repository root
uv sync --frozen
uv run python -m src.experiment
```

Notes:

- First run downloads datasets via `oddball` and caches them locally.
- If `outputs/model_selection/<dataset>.csv` exists, Phase 1 is skipped.
- If `outputs/experiment_results/<dataset>.csv` exists, Phase 2 is skipped.

To force a full rerun:

```bash
# PowerShell
Remove-Item outputs\model_selection\*.csv -ErrorAction SilentlyContinue
Remove-Item outputs\experiment_results\*.csv -ErrorAction SilentlyContinue
uv run python -m src.experiment
```

## Pip Fallback (no `uv` on reviewer machine)

A pinned fallback lockfile is included: `requirements.lock.txt`.

```bash
# PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
python -m src.experiment
```

## Workflow

### Phase 1: Model selection

Run:

```bash
uv run python -m src.experiment
```

Summaries:

```bash
uv run python -m src.scripts.model_summary outputs/model_selection/breastw.csv
uv run python -m src.scripts.model_summary outputs/model_selection/breastw.csv --metric prauc
uv run python -m src.scripts.model_summary outputs/model_selection/breastw.csv --format csv
uv run python -m src.scripts.model_summary outputs/model_selection/*.csv
```

### Phase 2: Experiments

Run:

```bash
uv run python -m src.experiment
```

Summaries:

```bash
uv run python -m src.scripts.experiment_summary outputs/experiment_results/ionosphere.csv
uv run python -m src.scripts.experiment_summary outputs/experiment_results/ionosphere.csv --group-by approach train_size
uv run python -m src.scripts.experiment_summary outputs/experiment_results/ionosphere.csv --group-by approach train_size test_size
uv run python -m src.scripts.experiment_summary outputs/experiment_results/*.csv --format csv
uv run python -m src.scripts.experiment_summary outputs/experiment_results/*.csv
uv run python -m src.scripts.experiment_summary results/deterministic/*.csv --format latex --condition-label Deterministic
```

Generate pruning plot blocks:

```bash
uv run python -m src.scripts.pruning_summary
uv run python -m src.scripts.pruning_summary --results-root results --trials 20 --mean-precision 3 --sem-precision 4
```

## Output format

### Model selection CSV

- Columns: `seed, dataset, model, fold, prauc, rocauc, brier, is_best`
- One row per seed (mean fold metrics)
- Best model marked by `is_best=True`

### Experiment CSV

- Columns: `seed, dataset, model, approach, train_size, test_size, n_train, n_test, n_test_normal, n_test_anomaly, actual_anomaly_rate, fdr, power`
- One row per `(seed, approach, train_size, test_size)`
- Appended summary rows use `seed="mean"` with formatted mean +/- std

## Configuration

All experiment settings are in `src/config.toml`:

```toml
[global]
meta_seeds = 20
train_split = 0.5
selection_folds = 10
fdr_rate = 0.1
n_bootstraps = 100
n_trials = 100
weight_estimator = "forest_bagged"  # "forest" or "forest_bagged"
pruning = "heterogeneous"           # "deterministic", "homogeneous", "heterogeneous"
test_use_proportion = 0.5
test_anomaly_rate = 0.05
approaches = ["empirical_weighted", "empirical_randomized_weighted"]

[experiments]
datasets = ["wbc", "ionosphere", "wdbc", "breastw", "vowels", "cardio", "musk", "satellite", "mammography"]
models = ["iforest", "loda", "inne", "hbos", "abod"]
```

## Additional docs

- `SETUP.md`: unified copy/paste setup and rerun instructions.
