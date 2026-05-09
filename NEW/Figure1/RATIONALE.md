# Figure 1 Rationale: Finite-Sample Detectability Phase Transition

This figure is a controlled theorem-validation experiment, not a realistic
anomaly-detection benchmark. The goal is to show when exact weighted conformal
p-values become too coarse for BH to make even a first discovery, even when the
anomaly score is oracle-quality.

In simple terms, the figure asks:

```text
Can weighted conformal p-values become too coarse to let BH discover anomalies,
even when the anomaly score is good?
```

## Synthetic Model

Calibration inliers are drawn from

```text
P0: Z ~ N(0, 1), T ~ N(0, 1).
```

Shifted test inliers are drawn from

```text
Q_rho: Z ~ N(rho, 1), T ~ N(0, 1).
```

Anomalies are drawn from

```text
A_{rho,kappa}: Z ~ N(rho, 1), T ~ N(kappa, 1).
```

The score is the oracle anomaly direction, `S = T`, and the oracle covariate
shift weight is

```text
w = exp(rho * Z - rho^2 / 2).
```

This separates the benign covariate-shift direction `Z` from the anomaly
direction `T`.

## Diagnostic

The exact weighted conformal p-value convention is

```text
p_j = (w_j + sum_i w_i 1{S_i >= S_j}) / (w_j + sum_i w_i),
```

where larger scores are more anomalous. The smallest attainable p-value for a
test point is its self-weight atom:

```text
p_min_j = w_j / (w_j + sum_i w_i).
```

For the phase diagrams, `p_min_(1)` is computed over the actual anomaly test
points in the synthetic experiment. This avoids a misleading case where a
low-weight inlier has a tiny self-atom but cannot attain that p-value because
its score is not anomalous.

The first BH discovery cannot occur if the attainable resolution is worse than
the first BH threshold:

```text
p_min_(1) > alpha / m.
```

Equivalently, the main phase diagram uses

```text
log10(1 / p_min_(1)) vs. log10(m / alpha),
```

with the theoretical boundary `y = x`.

The CSV also records a rank-aware no-discovery certificate for BH, because with
many anomalies BH can occasionally reject at a higher rank even when the first
threshold alone is blocked.

## Oracle Interpretation

All panels use oracle covariate-shift weights. The experiment assumes the exact
density ratio is known:

```text
w = dQ_rho / dP_0 = exp(rho * Z - rho^2 / 2).
```

This makes the result optimistic relative to estimated-weight workflows. If
weights were learned from data, finite-sample detectability would usually be
noisier or worse because of weight-estimation variance, bias, clipping, and
instability in low-overlap regions. The point of the figure is therefore:

```text
Even with oracle weights, exact weighted CAD can lose detectability because the
weighted conformal p-values are too discrete.
```

The score is also oracle in the sense that it uses the true anomaly direction
`S = T`. The difference between the perfect-score and finite-score heatmaps is
not weight estimation. It is score separation:

- perfect score: `S_anom = +infinity`, so anomalies attain their p-value floor.
- finite score: `S_anom ~ N(kappa, 1)`, so calibration tail mass can remain
  above an anomaly score and increase its p-value.

## Testing Burden

The testing burden is

```text
B = m / alpha
```

where `m` is the number of hypotheses or test points being checked, and `alpha`
is the target FDR level.

BH's first rejection threshold is

```text
alpha / m.
```

So if `m` is large, or `alpha` is small, the first required p-value becomes very
tiny. Equivalently,

```text
testing burden = m / alpha = 1 / (alpha / m).
```

Example:

```text
m = 1000, alpha = 0.1
alpha / m = 0.0001
m / alpha = 10000
```

So the method needs to be able to produce p-values around `1e-4` before the
first BH discovery is even possible. In the plot, the x-axis is
`log10(m / alpha)`. Higher x-values mean more simultaneous tests, a stricter
first-discovery requirement, and a greater burden on p-value resolution.

The log axes translate roughly as follows. On the heatmap x-axis, each `+1`
means the testing burden is multiplied by `10`. With `alpha = 0.1`,
`log10(m / alpha) = 4` corresponds to `m / alpha = 10,000`, or about
`m = 1,000` tests. On the y-axis, `log10(1 / p_min) = 4` means the attainable
p-value floor is about `1e-4`.

For the collapse plots, `log10(delta) = 0` means the attainable p-value floor
equals the relevant BH scale. Negative values are favorable; positive values
mean the floor is too large by a factor of `10^x`.

## Parameters Varied and Fixed

The simulation varies:

- calibration size `n`,
- test batch size `m`,
- covariate-shift severity `rho`,
- score strength `kappa`,
- random seed.

The split heatmap figure now also varies two scenario-level quantities:

| Scenario | alpha | anomaly fraction pi1 | Purpose |
| --- | ---: | ---: | --- |
| `baseline` | 0.10 | 0.10 | Main setting used by the original panels |
| `alpha_005` | 0.05 | 0.10 | Stricter FDR target / larger testing burden |
| `pi1_001` | 0.10 | 0.01 | Rare-anomaly setting with fewer true small p-values |

Changing `alpha` mostly moves the BH scale through `m / alpha`. Changing
`pi1` is different: it changes how many true anomaly p-values can support later
BH ranks.

## Panel Interpretation

### Figure: Standalone Panel A, Controlled Gaussian Shift

The standalone schematic figure shows the synthetic data setup.

- Gray points: calibration inliers from the original distribution.
- Blue points: shifted test inliers.
- Orange points: anomalies.
- Horizontal direction `Z`: benign covariate shift. This affects the weights.
- Vertical direction `T`: anomaly direction. This is the score, `S = T`.

Simple meaning: the experiment cleanly separates distribution shift from
anomaly signal. It is intentionally stylized; the goal is mechanism isolation,
not realistic anomaly detection.

### Figure: Heatmap Grid

The heatmap figure is a `3 x 2` grid. Columns compare score quality:

- left column: perfect-score exact WEDF, `kappa = infinity`;
- right column: finite-score exact WEDF, `kappa = 3.0`.

Rows compare scenario changes:

- row 1: baseline, `alpha = 0.10`, `pi1 = 0.10`;
- row 2: stricter FDR, `alpha = 0.05`, `pi1 = 0.10`;
- row 3: rare anomalies, `alpha = 0.10`, `pi1 = 0.01`.

Each heatmap panel is a theorem-style phase diagram.

- x-axis: testing burden, `log10(m / alpha)`. Farther right means BH needs
  smaller p-values.
- y-axis: attainable p-value resolution, `log10(1 / p_min)`. Higher means the
  conformal method can produce smaller p-values.
- pixel color: probability that BH finds at least one true anomaly in that
  diagnostic bin.
- dashed diagonal: the theoretical boundary `1 / p_min = m / alpha`.

Simple meaning: when the attainable p-values are too coarse relative to the
testing burden, exact weighted CAD cannot discover anything. This happens even
with perfect anomaly scores.

The phase diagrams are rendered as pixel-based heatmaps rather than scatter
plots. The simulation grid is intentionally widened beyond the original main
paper grid to make these heatmap rectangles supported by actual simulations:
the base grid uses `n` from `10` to `4000` and `rho` from `0` to `3.0`.
A targeted high-resolution supplement adds `n = 8000, 32000`, `rho` up to
`4.0`, and the smaller `m` values. This fills more of the upper-left
high-resolution/low-testing-burden region with real simulations. White regions
mean no simulated setting landed in that diagnostic bin; they are left blank
rather than interpolated.

For display, the heatmap y-axis is capped at the 99.5th percentile of simulated
attainable resolution. The omitted top tail consists of rare extreme draws that
otherwise stretch the viewport and create unsupported holes; no colors are
interpolated or guessed.

The finite-score column is worse than the perfect-score column solely because
finite scores overlap with the calibration score distribution. The weights
remain oracle in both columns.

### Figure: Collapse Diagnostics

The collapse figure is a `1 x 2` plot and uses only the baseline scenario
`alpha = 0.10`, `pi1 = 0.10`.

The left panel keeps the first-threshold diagnostic from the prior Panel D:

```text
delta = p_min_(1) / (alpha / m)
```

- `delta > 1`: the best attainable p-value is larger than the first BH
  threshold.
- `delta < 1`: discovery becomes feasible.
- separate lines show different anomaly strengths `kappa`, including stepwise
  finite-score levels below `3`.

Simple meaning: the detectability ratio predicts whether discoveries are
possible. Stronger anomaly scores increase discovery probability, but they
cannot fully overcome bad p-value resolution.

Panel D is intentionally the `r = 1` version of the detectability ratio. It
compares against the first BH threshold only, so curves may remain above zero
for `delta > 1` when BH can still reject at later ranks.

The right panel keeps the rank-aware BH-scale diagnostic from the prior Panel E:

```text
Delta_BH = min_r p_min_(r) / ((r / m) alpha)
```

Here `p_min_(r)` is the `r`-th smallest attainable p-value among the anomaly
test points. This mirrors the heuristic definition
`delta_j(r) = p_min_j / ((r / m) alpha)` and then summarizes the realized
configuration by the most favorable putative rejection rank.

- `Delta_BH > 1`: no anomaly-side attainable p-value rank reaches its matching
  BH scale.
- `Delta_BH < 1`: the realized weights have enough rank-aware resolution for
  at least some putative anomaly discovery count.

Simple meaning: Panel E is the BH-rank-aware version of Panel D. Its transition
is expected to align more tightly with the vertical boundary at zero because it
accounts for BH's later thresholds, not only the first one.

## Output

The script `figure1_phase_transition.py` is intentionally flat and
self-contained. It now writes compact summaries and three standalone figures:

- `figure1_panel_a_schematic.png`: standalone Gaussian-shift schematic.
- `figure1_heatmaps_alpha_pi_sensitivity.png`: `3 x 2` heatmap grid for
  baseline, stricter-alpha, and rare-anomaly scenarios.
- `figure1_collapse_diagnostics.png`: `1 x 2` baseline collapse figure for the
  first-threshold and rank-aware diagnostics.
- `figure1_heatmap_summary.csv`: compact binned heatmap probabilities.
- `figure1_collapse_summary.csv`: compact binned collapse probabilities.

The older raw per-trial CSV and five-panel PNG may still exist locally from
earlier iterations, but the current workflow does not rely on them.

The script avoids touching the main project code under `src/`.
