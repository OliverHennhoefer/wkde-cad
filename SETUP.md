# Setup

## Install

From the repository root:

```bash
uv sync --frozen
```

The project requires Python 3.12 or newer and installs the dependencies listed
in `pyproject.toml`.

## Run the Main Experiments

The project uses standalone figure scripts and an empirical benchmark:

```bash
uv run python -m src.scripts.figure1_phase_transition
uv run python -m src.scripts.figure2_perfect_score_resolution
uv run python -m src.scripts.figure3_randomized_pvalue_instability
uv run python -m src.scripts.figure4_clipping_frontier
uv run python -m src.scripts.empirical_benchmark
```

Outputs are written under the repository-root `outputs/` folder:
`outputs/figure1`, `outputs/figure2`, `outputs/figure3`, `outputs/figure4`,
and `outputs/empirical_benchmark`. Long-running loops show tqdm progress bars.

The empirical benchmark reads
`src/scripts/empirical_benchmark/empirical_benchmark.toml` by default and writes
dataset result CSVs under the configured output directory. It supports the old
benchmark flags:

```bash
uv run python -m src.scripts.empirical_benchmark --datasets wbc --seeds 1 --severities 0 1 --output-dir outputs/empirical_benchmark/results/logistic --jobs 1 --force
```

## Validate

Run the full test suite:

```bash
uv run python -m pytest -q
```

The suite covers the promoted figure experiment modules and their compact smoke
runs.
