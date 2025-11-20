# learning-the-null

Experimental testbed for conformal anomaly detection with false discovery rate (FDR) control.

## Overview

This project implements a comprehensive experimentation framework for evaluating conformal anomaly detection methods using the [nonconform](https://github.com/OliverHennhoefer/nonconform) library. It compares multiple approaches:

- **Empirical**: Standard conformal prediction with JackknifeBootstrap
- **Probabilistic**: Probabilistic conformal prediction with tuning trials
- **Empirical Weighted**: Weighted conformal prediction with covariate shift handling
- **Probabilistic Weighted**: Weighted probabilistic conformal prediction

## Key Features

- **Two-phase workflow**: Model selection followed by experimentation
- **Multiple anomaly detectors**: IForest, LODA, InnE, HBOS, COPOD, ECOD
- **Flexible configuration**: Configurable approaches, train/test sizes, and parameters
- **Incremental results saving**: Results saved immediately to prevent data loss
- **FDR control**: Benjamini-Hochberg and weighted FDR control methods
- **Automated summaries**: Tools for aggregating and analyzing results

## Configuration

All experiments are configured via `code/config.toml`:

```toml
[global]
meta_seeds = [1, 2, 3, ...]  # Random seeds for reproducibility
train_split = 0.5             # Proportion of data for training
fdr_rate = 0.1                # Nominal false discovery rate
n_bootstraps = 100            # Bootstrap iterations
n_trials = 100                # Probabilistic tuning trials
train_sizes = [100, 300, 1000]  # Training sizes to vary
test_sizes = [50, 100]        # Test batch sizes to vary
n_anomalies_fixed = 3         # Fixed anomalies per test batch
approaches = ["empirical", "probabilistic", "empirical_weighted", "probabilistic_weighted"]

[experiments]
datasets = ["ionosphere", "breast", ...]
models = ["iforest", "loda", "inne", "hbos", "copod", "ecod"]
```

## Workflow

### Phase 1: Model Selection

Model selection won't run if `experiments/model_selection/<dataset>.csv` exists.

```bash
python experiments/experiment_1.py  # Runs Phase 1 first
```

**Summarize model selection results:**

```bash
# Basic usage
uv run python -m code.utils.model_summary experiments/model_selection/breast.csv

# Sort by different metric
uv run python -m code.utils.model_summary experiments/model_selection/breast.csv --metric prauc

# CSV output
uv run python -m code.utils.model_summary experiments/model_selection/breast.csv --format csv

# Multiple files
uv run python -m code.utils.model_summary experiments/model_selection/*.csv
```

### Phase 2: Experimentation

Experiments won't run if `experiments/results/experiment1/<dataset>.csv` exists.

```bash
python experiments/experiment_1.py  # Runs Phase 2 with best models
```

**Results are saved incrementally** - each approach's results are written immediately to CSV, ensuring no data loss on interruption.

**Summarize experiment results:**

```bash
# Group by approach only (default)
uv run python -m code.utils.experiment_summary experiments/results/experiment1/ionosphere.csv

# Group by approach and train_size
uv run python -m code.utils.experiment_summary experiments/results/experiment1/ionosphere.csv --group-by approach train_size

# Group by approach, train_size, and test_size (full detail)
uv run python -m code.utils.experiment_summary experiments/results/experiment1/ionosphere.csv --group-by approach train_size test_size

# CSV output for processing
uv run python -m code.utils.experiment_summary experiments/results/experiment1/*.csv --format csv

# Multiple datasets
uv run python -m code.utils.experiment_summary experiments/results/experiment1/*.csv
```

## Project Structure

```
learning-the-null/
├── code/
│   ├── config.toml          # Main configuration
│   ├── utils/
│   │   ├── model_summary.py       # Model selection summarization
│   │   ├── experiment_summary.py  # Experiment results summarization
│   │   ├── registry.py            # Dataset and model registry
│   │   └── logger.py              # Logging utilities
├── experiments/
│   ├── experiment_1.py      # Main experiment script
│   ├── model_selection/     # Phase 1 results
│   └── results/
│       └── experiment1/     # Phase 2 results
└── figures/                 # Visualization scripts
```

## Output Format

### Model Selection Results
- Columns: seed, dataset, model, fold, prauc, rocauc, brier, is_best
- One row per seed with mean statistics
- Best model marked with `is_best=True`

### Experiment Results
- Columns: seed, dataset, model, approach, train_size, test_size, n_train, n_test, n_test_normal, n_test_anomaly, actual_anomaly_rate, fdr, power
- Individual results for each (seed, approach, train_size, test_size) combination
- Summary rows with `seed="mean"` showing aggregated statistics (mean ± std)