# Between Resolution Collapse and Variance Inflation

This repository contains four main figure experiments and one empirical
covariate-shift benchmark for weighted conformal anomaly detection. Experiments
write generated artifacts under the repository-root `outputs/` folder and show
tqdm progress bars for long-running simulation loops.

All commands below are single-line commands. They avoid shell-specific line
continuations, so they work the same way in Windows PowerShell, Windows cmd,
Bash, zsh, and CI shells.

## Install

```bash
uv sync --frozen
```

## Figure 1: Phase Transition

Run the default Figure 1 experiment:

```bash
uv run python -m src.scripts.figure1_phase_transition --modes unweighted weighted --weighted-pruning homogeneous --workers 11 --wcs-batch-size 512
```

Default parameters:

- `--modes`: `unweighted weighted`
- `--weighted-pruning`: `homogeneous`
- `--workers`: `11` in this checkout; if omitted, the script uses `FIGURE1_WORKERS` or `max(1, os.cpu_count() - 1)`
- `--wcs-batch-size`: `512`
- `OUT_ROOT`: `outputs/figure1`
- `SCENARIOS`: `baseline(alpha=0.10, pi1=0.10)`, `alpha_005(alpha=0.05, pi1=0.10)`, `pi1_001(alpha=0.10, pi1=0.01)`
- `N_VALUES`: `10 20 30 40 50 75 100 150 200 300 500 750 1000 2000 4000`
- `M_VALUES`: `20 29 41 58 82 118 168 239 340 485 691 985 1403 2000`
- `RHO_VALUES`: `0.0` to `3.0`, `61` grid points
- `KAPPA_VALUES`: `inf 2.0 2.5 3.0 3.5 4.0`
- `HEATMAP_KAPPAS`: `inf 3.0`
- `PANEL_C_KAPPA`: `3.0`
- `SUPPLEMENT_N_VALUES`: `8000 32000`
- `SUPPLEMENT_M_VALUES`: `20 29 41 58 82 118 168`
- `SUPPLEMENT_RHO_VALUES`: `0.0` to `4.0`, `61` grid points
- `INCLUDE_SUPPLEMENT_GRID`: `False`
- `N_SEEDS`: `100`
- `BASE_SEED`: `20260509`
- `HEATMAP_X_BINS`: `12`
- `HEATMAP_Y_BINS`: `12`
- `HEATMAP_VIEW_LIMITS`: `(2.25, 4.75)`
- `HEATMAP_CELL_TRIALS`: `25`
- `HEATMAP_MAX_ACCEPT_ATTEMPTS_PER_CELL`: `5000`
- `WEIGHTED_HEATMAP_RHO_CANDIDATES`: `0.5 1.0 1.5 2.0 2.5 3.0`
- `COLLAPSE_BINS`: `-2.5` to `2.5`, `35` grid points

## Figure 2: Perfect-Score Resolution

Run the default Figure 2 experiment:

```bash
uv run python -m src.scripts.figure2_perfect_score_resolution
```

This script has no CLI parameters. The command uses these fixed defaults:

- `OUT_DIR`: `outputs/figure2`
- `WORKERS`: `8` in this checkout; internally `max(1, min(8, os.cpu_count() - 1))`
- `ALPHA`: `0.10`
- `PI1`: `0.05`
- `DELTA_SCORE`: `1.0`
- `D`: `10`
- `N_VALUES`: `50 75 100 150 200 300 500 750 1000`
- `M_VALUES`: `50 100 200 500 1000 2000`
- `RHO_VALUES`: `0.0` to `2.5`, `51` grid points
- `N_SEEDS`: `100`
- `PANEL_C_M_VALUES`: `50 100 500 1000 2000`
- `DELTA_BINS`: `-2.5` to `2.5`, `31` grid points
- `LOG_ESS_BINS`: `1.0` to `3.1`, `22` grid points
- `PHASE_VIEW_LIMITS`: `(2.25, 4.75)`
- `PHASE_X_BINS`: `12`
- `PHASE_Y_BINS`: `12`
- `PHASE_CELL_TRIALS`: `100`
- `PHASE_MAX_ACCEPT_ATTEMPTS_PER_CELL`: `5000`
- `PHASE_RHO_CANDIDATES`: `0.5 1.0 1.5 2.0 2.5 3.0`
- `POWER_CONFIGS`: `(alpha=0.10, pi1=0.05)`, `(alpha=0.05, pi1=0.05)`, `(alpha=0.10, pi1=0.01)`, `(alpha=0.05, pi1=0.01)`

## Figure 3: Randomized P-Value Instability

Run the default Figure 3 experiment:

```bash
uv run python -m src.scripts.figure3_randomized_pvalue_instability
```

This script has no CLI parameters. The command uses these fixed defaults:

- `OUT_DIR`: `outputs/figure3`
- `ALPHA`: `0.10`
- `N_CAL`: `500`
- `M`: `1000`
- `N_ANOMALY`: `10`
- `DELTA_SCORE`: `1.0`
- `D`: `10`
- `RHO_VALUES`: `0.0` to `2.5`, `11` grid points
- `N_WORLD_SEEDS`: `40`
- `N_RANDOMIZATIONS`: `20000`
- `SIMULATION_BATCH_SIZE`: `10000`
- `FRONTIER_BINS`: `12`
- `INLIER_TEST_WEIGHT_CAP_FRACTION`: `0.5`
- `BASE_SEED`: `20260512`

## Figure 4: Clipping Frontier

Run the default Figure 4 experiment:

```bash
uv run python -m src.scripts.figure4_clipping_frontier
```

This script has no CLI parameters. The command uses these fixed defaults:

- `OUT_DIR`: `outputs/figure4`
- `RHO_VALUES`: `0.5 1.5 3.0`
- `N_CAL`: `500`
- `M`: `1000`
- `N_SEEDS`: `500`
- `ALPHA`: `0.10`
- `CLIP_CAPS`: `1 1.5 2 3 5 8 13 21 34 55 89 inf`
- `TAIL_PROBS`: `0.0001` to `0.5`, `80` geometric grid points
- `EPSILON`: `1e-300`
- `BASE_SEED`: `20260512`

## Figure 5: Operational Envelope

Run the default Figure 5 operational-envelope evaluation:

```bash
uv run python -m src.scripts.figure5_operational_envelope --workers 11 --wcs-batch-size 512 --cell-trials 100
```

Default parameters:

- `OUT_DIR`: `outputs/figure5`
- `--workers`: `11` in this checkout; if omitted, the script uses `max(1, os.cpu_count() - 1)`
- `--wcs-batch-size`: `512`
- `--cell-trials`: `100`
- `--force`: omitted, so matching cached summaries are reused
- `N_EFF_BINS`: `log10 N_eff` from `1.3` to `3.7`, `12` displayed bins
- `M_VALUES`: `50 100 200 500 1000 2000`
- `PI1_VALUES`: `0.005 0.01 0.02 0.05 0.10 0.20`
- `ALPHA_VALUES`: `0.01 0.025 0.05 0.10 0.20`
- `REQUIRED_ALPHA_VALUES`: `0.05 0.10`
- score regimes: perfect score and finite score with `kappa = 3.0`
- shift/weight setup: oracle Gaussian covariate shift in `D = 10`
- WCS pruning: homogeneous

Outputs:

- `figure5_operational_power_atlas.png`: `2 x 4` controlled power atlas varying `m`, anomaly count/rate, and `alpha`
- `figure5_operational_power_atlas_summary.csv`: compact atlas cell summaries
- `figure5_operational_power_atlas_tikz.csv`: PGFPlots/TikZ-friendly long table with heatmap cells, cell bounds, panel IDs, power, FDR, AUROC, and rank-diagnostic values
- `figure5_operational_power_atlas_reference_tikz.csv`: PGFPlots/TikZ-friendly reference table for default-setting markers and contour levels
- `figure5_required_ess_design_map.png`: minimum effective calibration size needed for 80% power
- `figure5_required_ess_summary.csv`: compact required-ESS summaries
- `figure5_required_ess_tikz.csv`: PGFPlots/TikZ-friendly long table with required-ESS heatmap cells and out-of-range status labels
- `figure5_empirical_projection.png`: empirical benchmark projection onto the resolution plane
- `figure5_empirical_projection_summary.csv`: compact empirical projection diagnostics
- `figure5_empirical_projection_tikz.csv`: PGFPlots/TikZ-friendly point table for empirical projections and the diagonal reference boundary

The PNG files are intended for quick visual inspection. The `*_tikz.csv` files
are the camera-ready plotting inputs: they use stable panel names, `x`/`y`
coordinates, cell-bound columns (`x_left`, `x_right`, `y_bottom`, `y_top`),
numeric plotting values, labels, and `style_key` fields so a LaTeX project can
consume them directly with `pgfplots` table filters or equivalent tooling.

## Empirical Covariate-Shift Benchmark

Run the default empirical benchmark:

```bash
uv run python -m src.scripts.empirical_benchmark --config src/scripts/empirical_benchmark/empirical_benchmark.toml --datasets wbc ionosphere wdbc breastw vowels cardio musk satellite mammography --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 --severities 0.0 0.5 1.0 4.0 --approaches empirical empirical_randomized empirical_weighted empirical_randomized_weighted --output-dir outputs/empirical_benchmark/results/logistic --jobs 10
```

Default CLI parameters:

- `--config`: `src/scripts/empirical_benchmark/empirical_benchmark.toml`
- `--datasets`: `wbc ionosphere wdbc breastw vowels cardio musk satellite mammography`
- `--seeds`: `1` through `25`
- `--severities`: `0.0 0.5 1.0 4.0`
- `--approaches`: `empirical empirical_randomized empirical_weighted empirical_randomized_weighted`
- `--output-dir`: `outputs/empirical_benchmark/results/logistic`
- `--jobs`: `10` in this checkout; internally `min(cpu_count, 2)` when `cpu_count <= 4`, otherwise `max(1, cpu_count - 2)`
- `--force`: omitted, so `False`

Default TOML parameters used by that command:

- `[model_selection].models`: `iforest inne hbos`
- `[model_selection].folds`: `10`
- `[model_selection].output_dir`: `outputs/empirical_benchmark/model_selection`
- `[splits].train_split`: `0.4`
- `[splits].test_use_proportion`: `0.5`
- `[splits].test_anomaly_rate`: `0.05`
- `[conformal].fdr_rate`: `0.1`
- `[conformal].split_calib`: `0.5`
- `[conformal].pruning`: `homogeneous`
- conformal calibration strategy: vanilla `Split(n_calib=0.5)`
- `[weighting].mode`: `estimated`
- `[weighting].estimator`: `forest`
- `[weighting].n_bootstraps`: `50`, used only for `forest_bagged`
- `[covariate_shift].propensity_min`: `0.3`
- `[covariate_shift].propensity_max`: `0.7`
- `[plots].output_dir`: `outputs/empirical_benchmark/plots`
- `[plots].bins`: `10`

At `severity = 0.0`, both unweighted and weighted empirical approaches can run.
At nonzero severities, only weighted empirical approaches are run internally.

## Tests

Run the full test suite:

```bash
uv run python -m pytest -q
```
