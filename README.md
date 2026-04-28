# Between Resolution Collapse and Variance Inflation: Weighted Conformal Anomaly Detection

This repository runs covariate-shift experiments for weighted conformal anomaly
detection with `nonconform`.

## Quick Start

From the repository root:

```bash
uv sync --frozen
uv run python -m src.experiment --config config.toml
```

The runner uses `config.toml` by default and writes:

- `outputs/model_selection/<dataset>.csv`
- `outputs/experiment_results/<dataset>.csv`
- `outputs/experiment_results/config.toml`

Existing dataset result CSVs are skipped unless `--force` is passed.

## Experiment CLI

```bash
uv run python -m src.experiment \
  --config config.toml \
  --jobs 5
```

This uses the datasets, seeds, severities, methods, and output directory from
`config.toml`.

Common options:

- `--config`: TOML config path, default `config.toml`
- `--datasets`: dataset names from `[experiment].datasets`
- `--seeds`: explicit meta-seeds to evaluate
- `--severities`: covariate-shift severities
- `--approaches`: method list override
- `--output-dir`: result CSV folder
- `--jobs`: worker count
- `--force`: recompute existing dataset outputs and overwrite the saved config snapshot

The configured `[methods].approaches` list can include both weighted and
unweighted methods. At `severity = 0`, every configured method runs. At nonzero
severities, unweighted methods are skipped internally and only weighted methods
run.

## Useful Commands

```bash
uv run python -m src.scripts.covariate_shift_summary outputs/experiment_results/*.csv
uv run python -m src.scripts.fdr_table outputs/experiment_results
uv run python -m src.scripts.plot_covariate_shift
```

## Configuration

Edit `config.toml` to change datasets, models, seeds, severity levels,
approaches, split sizes, conformal settings, weighting, covariate-shift bounds,
plots, and output directories. The main sections are `[experiment]`,
`[model_selection]`, `[splits]`, `[conformal]`, `[weighting]`,
`[covariate_shift]`, `[methods]`, and `[plots]`.

For step-by-step setup instructions, see `SETUP.md`.
