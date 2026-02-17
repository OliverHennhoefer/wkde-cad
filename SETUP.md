# Setup

Minimal setup for a fresh clone or after deleting `.venv`.

## Requirements

- Python `3.12+`
- Internet access on first run (datasets are downloaded once and cached)

## Recommended (`uv`)

```bash
# from repository root
uv sync --frozen
uv run python -m src.experiment
```

## Fallback (pip only)

```powershell
# PowerShell, from repository root
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
python -m src.experiment
```

## Rerun from scratch

```powershell
Remove-Item outputs\model_selection\*.csv -ErrorAction SilentlyContinue
Remove-Item outputs\experiment_results\*.csv -ErrorAction SilentlyContinue
```

Then run again:

```bash
uv run python -m src.experiment
```

## Quick check

After a successful run, these should exist:

- `outputs/model_selection/wbc.csv`
- `outputs/experiment_results/wbc.csv`
