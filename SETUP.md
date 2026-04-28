# Setup Guide

This guide assumes you are starting from a fresh clone and are new to running
Python projects from a terminal.

## 1. Open A Terminal In The Repository

Move into the project folder before running commands:

```bash
cd /path/to/wkde-cad
```

You are in the right place when this command lists files such as `README.md`,
`pyproject.toml`, and `config.toml`:

```bash
ls
```

On Windows PowerShell, use:

```powershell
dir
```

## 2. Install Requirements

Install Python 3.12 or newer. Check your version:

```bash
python --version
```

Install `uv` if it is not already available:

```bash
python -m pip install uv
```

Then install the project dependencies:

```bash
uv sync --frozen
```

## 3. Run The Experiment

Start the covariate-shift experiment runner:

```bash
uv run python -m src.experiment
```

The first run may take longer because datasets are downloaded and cached.

The default `config.toml` contains one method list under `[methods]`. You can
leave both weighted and unweighted methods in that list: unweighted methods run
only for `severity = 0`, while weighted methods run for every configured
severity.

## 4. Check The Outputs

After a successful run, these files should exist:

- `outputs/model_selection/wbc.csv`
- `outputs/experiment_results/wbc.csv`
- `outputs/experiment_results/config.toml`

The saved `config.toml` inside `outputs/experiment_results` records the setup
used for those result files.

## 5. Run A Smaller Test Experiment

For a quick smoke test:

```bash
uv run python -m src.experiment --datasets wbc --seeds 1 --severities 0 --jobs 1 --output-dir outputs/smoke_test --force
```

## 6. Rerun Existing Results

By default, existing dataset result CSVs are skipped. To recompute them:

```bash
uv run python -m src.experiment --force
```

To write into a separate folder:

```bash
uv run python -m src.experiment --output-dir outputs/my_run --force
```

## 7. Edit The Config

Most experiment settings are in these `config.toml` sections:

- `[experiment]`: datasets, seed count, severities, result output folder
- `[model_selection]`: candidate models, cross-validation folds, model-selection output folder
- `[splits]`: train/test split and target test anomaly rate
- `[conformal]`: FDR level, bootstrap/trial counts, pruning mode
- `[weighting]`: oracle vs estimated weights, estimated-weight model
- `[covariate_shift]`: propensity clipping bounds
- `[methods]`: candidate approaches
- `[plots]`: plot output folder and bin count

## 8. Summarize Results

Print a covariate-shift summary table:

```bash
uv run python -m src.scripts.covariate_shift_summary outputs/experiment_results/*.csv
```

Create an FDR table:

```bash
uv run python -m src.scripts.fdr_table outputs/experiment_results
```

Generate covariate-shift plots:

```bash
uv run python -m src.scripts.plot_covariate_shift
```

## 9. Common Problems

If `uv` is not found, install it with:

```bash
python -m pip install uv
```

If `python` points to an old Python version, try:

```bash
python3.12 --version
python3.12 -m pip install uv
```

If a run is skipped, the output CSV already exists. Pass `--force` or choose a
new `--output-dir`.

If dataset download fails, check your internet connection and rerun the same
command.

## 10. Pip Fallback

Use this only if `uv` is unavailable.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m src.experiment
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m src.experiment
```
