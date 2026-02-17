# Between Resolution Collapse and Variance Inflation: Stabilizing Weighted Conformal Anomaly Detection

## Overview

This project implements a comprehensive experimentation framework for evaluating conformal anomaly detection methods using the [nonconform](https://github.com/OliverHennhoefer/nonconform) library. It compares multiple approaches:

- **Empirical**: Standard conformal prediction with JackknifeBootstrap
- **Empirical Randomized**: Empirical conformal with randomized tie-breaking
- **Probabilistic**: Probabilistic conformal prediction with tuning trials
- **Empirical Weighted**: Weighted conformal prediction with covariate shift handling
- **Empirical Randomized Weighted**: Weighted empirical conformal with randomized tie-breaking
- **Probabilistic Weighted**: Weighted probabilistic conformal prediction

## Key Features

- **Two-phase workflow**: Model selection followed by experimentation
- **Multiple anomaly detectors**: IForest, LODA, InnE, HBOS, COPOD, ECOD
- **Flexible configuration**: Configurable approaches, test batch sizing, and parameters
- **Incremental results saving**: Results saved immediately to prevent data loss
- **FDR control**: Benjamini-Hochberg and weighted FDR control methods
- **Automated summaries**: Tools for aggregating and analyzing results

# Usage

## Workflow

### Phase 1: Model Selection

Model selection won't run if `outputs/model_selection/<dataset>.csv` exists.

```bash
python -m src.experiment  # Runs Phase 1 first
```

Summarize model selection results:

```bash
# Basic usage
uv run python -m src.scripts.model_summary outputs/model_selection/breastw.csv

# Sort by different metric
uv run python -m src.scripts.model_summary outputs/model_selection/breastw.csv --metric prauc

# CSV output
uv run python -m src.scripts.model_summary outputs/model_selection/breastw.csv --format csv

# Multiple files
uv run python -m src.scripts.model_summary outputs/model_selection/*.csv
```

### Phase 2: Experimentation

Experiments won't run if `outputs/experiment_results/<dataset>.csv` exists.

```bash
python -m src.experiment  # Runs Phase 2 with best models
```

Results are saved incrementally - each approach's results are written immediately
to CSV, ensuring no data loss on interruption.

Summarize experiment results:

```bash
# Group by approach only (default)
uv run python -m src.scripts.experiment_summary outputs/experiment_results/ionosphere.csv

# Group by approach and train_size
uv run python -m src.scripts.experiment_summary outputs/experiment_results/ionosphere.csv --group-by approach train_size

# Group by approach, train_size, and test_size (full detail)
uv run python -m src.scripts.experiment_summary outputs/experiment_results/ionosphere.csv --group-by approach train_size test_size

# CSV output for processing
uv run python -m src.scripts.experiment_summary outputs/experiment_results/*.csv --format csv

# Multiple datasets
uv run python -m src.scripts.experiment_summary outputs/experiment_results/*.csv

# Full LaTeX table from deterministic results
uv run python -m src.scripts.experiment_summary results/deterministic/*.csv --format latex --condition-label Deterministic
```

Generate pruning comparison plot blocks:

```bash
# PGFPlots blocks for W. EDF (det./homog./het.) and W. KDE (Ours)
uv run python -m src.scripts.pruning_summary

# Optional configuration
uv run python -m src.scripts.pruning_summary --results-root results --trials 20 --mean-precision 3 --sem-precision 4
```

## Output Format

### Model Selection Results

- Columns: seed, dataset, model, fold, prauc, rocauc, brier, is_best
- One row per seed with mean statistics
- Best model marked with `is_best=True`

### Experiment Results

- Columns: seed, dataset, model, approach, train_size, test_size, n_train, n_test,
  n_test_normal, n_test_anomaly, actual_anomaly_rate, fdr, power
- Individual results for each (seed, approach, train_size, test_size) combination
- Summary rows with `seed="mean"` showing aggregated statistics (mean +/- std)

### Pruning Summary Output

- Prints PGFPlots `\addplot+` blocks with `mean(power) +- SEM(power)` (`SEM = std/sqrt(n)`, `n=20`)
- Includes datasets: WBC, Ionosphere, WDBC, BreastCa, Vowels, Cardio, Satellite, Mammogr.
- Excludes `Musk` intentionally (all methods have identical `1.0` power)


## Configuration

All experiments are configured via `src/config.toml`:

```toml
[global]
meta_seeds = 20              # Number of seeds (uses 1..20)
train_split = 0.5             # Proportion of data for training
selection_folds = 10          # Number of cross-validation folds for model selection
fdr_rate = 0.1                # Nominal false discovery rate
n_bootstraps = 100            # Bootstrap iterations
n_trials = 100                # Probabilistic tuning trials
weight_estimator = "forest"   # "forest" or "forest_bagged"
test_use_proportion = 0.5     # Proportion of test fold to use (0.5 = 50% of test split)
test_anomaly_rate = 0.05      # Target anomaly rate in test set (0.05 = 5% anomalies)
approaches = ["empirical", "empirical_randomized", "probabilistic", "empirical_weighted", "empirical_randomized_weighted", "probabilistic_weighted"]

[experiments]
datasets = ["ionosphere", "breast", ...]
models = ["iforest", "loda", "inne", "hbos", "copod", "ecod"]
```

## Project Structure

```
learning-the-null/
docs/
  usage.md                       # Workflow and output formats
outputs/
  model_selection/               # Phase 1 results
  experiment_results/            # Phase 2 results
figures/
  figure_1/                      # Figure 1 visualization
  figure_3/                      # Figure 3 visualization
src/
  experiment.py                  # Main experiment script
  config.toml                    # Main configuration
  scripts/
    model_summary.py             # Model selection summarization
    experiment_summary.py        # Experiment results summarization
    pruning_summary.py           # Pruning comparison PGFPlots summarization
  utils/
    data_loader.py               # Data loading utilities
    registry.py                  # Dataset and model registry
    logger.py                    # Logging utilities
```
