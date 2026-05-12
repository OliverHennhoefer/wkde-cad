# Figure 4 Rationale: Clipping Frontier

This figure is a controlled mechanism experiment. It asks whether clipping
oracle covariate-shift weights removes the weighted conformal resolution
problem, or merely changes the trade-off.

Calibration null scores are drawn from `P: Z ~ N(0, 1)`. Shifted null scores are
drawn from `Q_rho: Z ~ N(rho, 1)`, and the anomaly score is `S = Z`. The oracle
density ratio is

```text
w(z) = dQ_rho / dP = exp(rho z - rho^2 / 2).
```

For a clipping cap `c`, the experiment uses

```text
w_c(z) = min(w(z), c)
```

on both calibration and shifted-null test points.

The resolution diagnostic is the largest shifted-null self-atom,

```text
max_j w_c(Z_j^test) / (sum_i w_c(Z_i^cal) + w_c(Z_j^test)).
```

Smaller values mean the weighted conformal p-values can attain finer small
p-values. Clipping lowers this atom and raises the calibration effective sample
size because it suppresses extreme weights.

The adaptation diagnostic compares the normalized clipped target tail

```text
T_c(s) = E_P[min(w(Z), c) 1{Z >= s}] / E_P[min(w(Z), c)]
```

against the true shifted-null tail

```text
T_Q(s) = P_Q(Z >= s).
```

The main adaptation curve is the mean absolute log mismatch
`mean_s |log10(T_c(s) / T_Q(s))|` over a fixed shifted-null tail grid. The CSV
also records signed and absolute target-Q tail bias, plus finite-sample
weighted-tail bias from simulated calibration samples.

The point is that clipping improves finite-sample resolution by changing the
target distribution away from `Q`. It changes the frontier; it does not remove
the resolution-adaptation trade-off.
