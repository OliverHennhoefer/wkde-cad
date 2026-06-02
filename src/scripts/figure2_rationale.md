# Figure 2 Rationale: Perfect-Score Resolution Stress Test

This figure is a mechanism-isolation experiment. It is not meant to be a
realistic anomaly-detection benchmark. The goal is to show that exact weighted
conformal CAD can fail even when the anomaly score is already perfect relative
to the calibration tail.

The experiment asks:

```text
Can BH make no discoveries even when every anomaly is more extreme than every
calibration point?
```

## Synthetic Model

Calibration covariates are drawn from a 10-dimensional standard normal:

```text
X_i^cal ~ P_0 = N(0, I_10).
```

Shifted test covariates use the same nuisance coordinates, but the first
coordinate is shifted:

```text
X_j^test ~ Q_rho,       X_{j1} ~ N(rho, 1),   X_{jk} ~ N(0, 1), k > 1.
```

The oracle density ratio is therefore:

```text
w(x) = dQ_rho / dP_0 = exp(rho * x_1 - rho^2 / 2).
```

Scores are generated independently of the covariates:

```text
S_i^cal ~ N(0, 1),
S_j^0   ~ N(0, 1),
S_j^1   = max_i S_i^cal + 1.
```

Thus every anomaly is beyond the calibration tail by construction. Any failure
to reject cannot be blamed on the detector failing to rank anomalies above the
calibration sample.

## Exact Weighted EDF Floor

The exact weighted conformal p-value convention matches Figure 1:

```text
p_j = (w_j + sum_i w_i 1{S_i^cal >= S_j}) / (w_j + sum_i w_i).
```

For a perfect anomaly,

```text
S_j^1 > max_i S_i^cal,
```

so the calibration tail mass is zero and the p-value reduces to its smallest
attainable weighted atom:

```text
p_j = w_j / (W_cal + w_j).
```

This is the p-value floor. The score cannot make the exact WEDF p-value any
smaller.

## BH Detectability Diagnostic

BH at level `alpha` can make a first discovery only if some p-value crosses:

```text
alpha / m.
```

The figure therefore tracks:

```text
delta_min = p_min_anom / (alpha / m).
```

When `delta_min > 1`, even the smallest attainable anomaly p-value is too large
for the first BH threshold. The script also records a rank-aware version:

```text
Delta_BH = min_r p_min_(r) / (alpha * r / m),
```

which accounts for later BH thresholds when several anomalies are present.

## Designed Phase Grid

The main heatmap is a conditional diagnostic grid, not an adaptive binning of
naturally observed simulation settings. The plotted axes match the supported
Figure 1 phase-diagram regime:

```text
x = log10(m / alpha),          y = log10(1 / p_min_anom),
2.25 <= x, y <= 4.75.
```

Each visible cell is simulated directly. The x-cell center sets
`m = round(alpha * 10^x)`. Weighted `(n, rho)` configurations are sampled until
the realized `log10(1 / p_min_anom)` lands inside the target y-cell. Every
heatmap cell must have at least one accepted trial; a blank cell is a pipeline
failure, not an unsupported plotting region.

The `2.25..4.75` viewport is the same computationally meaningful regime chosen
for Figure 1. Extending farther upward would require much larger exact
auxiliary-row evaluations for WCS while adding limited practical information
about the p-value floor mechanism.

## Methods Shown

The main figure keeps the weighted oracle, perfect-score setting fixed and
compares three p-value constructions:

- Exact WEDF: the finite-sample conformal p-value with the full test self-atom.
- Randomized WEDF: a diagnostic relaxation that randomly splits the test
  self-atom.
- Oracle continuous tail: a diagnostic upper benchmark with anomaly p-values
  set to zero.

KDE is intentionally excluded. This avoids finite-sample validity objections
and keeps the figure focused on the discreteness mechanism.

## Output

The script `figure2_perfect_score_resolution.py` writes:

- `figure2_perfect_score_resolution.png`: the `1 x 3` main mechanism figure:
  score schematic, supported square phase heatmap, and method-relaxation
  detectability curves.
- `figure2_power_configurations.png`: supplemental `2 x 2` power/ESS
  sensitivity analysis varying `alpha` and the anomaly rate, with
  `log10(mean N_eff)` on a shared linear x-axis.
- `figure2_perfect_score_summary.csv`: one compact row per designed phase cell,
  with accepted-trial counts, p-value floors, diagnostics, and method outcomes.
- `figure2_key_settings_table.csv`: a short aggregate table contrasting
  feasible and certified-impossible regions.
- `figure2_power_configurations_summary.csv`: the compact binned summary used
  for the supplemental configuration plot.

The workflow is standalone and does not depend on Figure 1 outputs or modify
the project code under `src/`.
