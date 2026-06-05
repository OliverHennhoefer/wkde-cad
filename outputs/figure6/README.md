# Figure 6: Standard Unweighted CAD Power Atlas

Figure 6 is the warm-up experiment for standard split-conformal anomaly
detection. It intentionally removes covariate shift, importance weights, and WCS
pruning so that the only finite-sample bottleneck is ordinary calibration
resolution.

## Purpose

The plot isolates what can already go wrong in the unweighted setting:
conformal p-values have minimum resolution `1 / (N_cal + 1)`, so BH can be
unable to make discoveries even when the anomaly score is strong. Because there
are no weights, `N_eff = N_cal` exactly in every cell.

Figure 6 should be read before Figure 5. Figure 5 then adds weight
concentration and covariate shift on top of the same resolution logic.

## Command

Default run:

```bash
uv run python -m src.scripts.figure6_unweighted_power_atlas --workers 11 --calibration-repeats 25 --test-repeats 100 --force
```

For a quick smoke test, use fewer repeats:

```bash
uv run python -m src.scripts.figure6_unweighted_power_atlas --workers 1 --calibration-repeats 1 --test-repeats 1 --force
```

`--cell-trials` is kept only as a legacy/debug shortcut. It uses one calibration
set per cell and evaluates the requested number of test batches against it.

## Synthetic Setup

- Calibration scores: iid `N(0, 1)`.
- Inlier test scores: iid `N(0, 1)`.
- No covariate shift: calibration and test inliers come from the same score
  distribution.
- Uniform weights only: no importance weights are generated or used.
- Ordinary upper-tail conformal p-values:
  `p_j = (1 + #{calibration scores >= test score_j}) / (N_cal + 1)`.
- Multiple testing: standard BH at nominal level `alpha`.
- Test batch labels are known only for simulation summaries, not for decisions.

Score regimes:

- `perfect`: every anomaly score is placed above the maximum calibration or
  inlier score in that realization by `DELTA_SCORE = 1.0`.
- `finite_k1`: anomaly scores are iid `N(1, 1)`.
- `finite_k2`: anomaly scores are iid `N(2, 1)`.
- `finite_k3`: anomaly scores are iid `N(3, 1)`.

## Grid

The atlas uses the same sweeps as the Figure 5 power atlas where possible.

- Rows: `perfect`, `finite_k1`, `finite_k2`, `finite_k3`.
- Columns: batch-size sweep with fixed anomaly rate, batch-size sweep with fixed
  anomaly count, anomaly-rate sweep, and nominal-FDR sweep.
- `N_EFF_BINS = np.linspace(1.3, 4.1, 15)`, with cell centers from about 40 to
  10,000 calibration points.
- `M_VALUES = [50, 100, 200, 500, 1000, 2000]`.
- `PI1_VALUES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]`.
- `ALPHA_VALUES = [0.01, 0.025, 0.05, 0.10, 0.20]`.
- Baselines: `alpha = 0.10`, `pi1 = 0.05`, `m = 1000`, and fixed anomaly
  count `10` where applicable.

## Nested Monte Carlo

The default estimator uses `25 x 100 = 2,500` realizations per atlas cell:

- draw one split-conformal calibration set;
- hold it fixed;
- evaluate 100 independent test batches against that calibration set;
- repeat this for 25 independent calibration sets;
- average power and FDP over all calibration/test pairs.

This matches the marginal-FDR estimand more directly than drawing a fresh
calibration set for every single test batch. It estimates
`E_calibration E_test[FDP | calibration]`, and the same nested summary is used
for both the power atlas and the FDR-ratio supplement.

## Plot Readout

Main power atlas:

- Heatmap: empirical statistical power.
- Red boundary `#E53935`: `median_log10_rank_delta = 0`, the rank-resolution
  boundary where the smallest possible conformal p-values become compatible
  with at least one BH rejection.
- The former 80% empirical-power contour is omitted so the heatmap carries the
  power scale without an arbitrary secondary line.
- No default-setting star marker is shown.
- Cell outlines and dark grid boundaries are suppressed for readability.

Supplementary FDR-ratio atlas:

- Heatmap: empirical FDR divided by nominal `alpha`.
- Red boundary `#E53935`: same rank-resolution boundary.
- The former FDR/alpha = 1 contour is omitted because the estimated contour can
  look more precise than the Monte Carlo resolution supports.
- The supplement is a replot of the same summary data, not a separate
  simulation.

## Outputs

- `figure6_unweighted_power_atlas.png`
- `figure6_unweighted_power_atlas_summary.csv`
- `figure6_unweighted_power_atlas_tikz.csv`
- `figure6_unweighted_power_atlas_reference_tikz.csv`
- `figure6_supp_fdr_ratio_atlas.png`
- `figure6_supp_fdr_ratio_atlas_tikz.csv`
- `figure6_supp_fdr_ratio_atlas_reference_tikz.csv`

The summary CSV includes `calibration_repeats`,
`test_repeats_per_calibration`, and `count` so the Monte Carlo design can be
audited from the output file.

## Caveats

The FDR values remain Monte Carlo estimates. Cells with very low discovery
probability can have noisy FDR ratios because FDP is observed only when
discoveries occur. The nested design improves interpretability of the marginal
estimate, but it does not remove ordinary Monte Carlo error.

This nested estimator is currently implemented for Figure 6 only. Figure 5
keeps its weighted covariate-shift design unchanged.
