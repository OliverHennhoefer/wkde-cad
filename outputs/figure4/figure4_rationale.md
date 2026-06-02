# Figure 4 Rationale: Clipping Frontier

This figure is a controlled mechanism experiment, not a new method benchmark.
It asks whether clipping oracle covariate-shift weights removes weighted
conformal resolution collapse or merely changes the target distribution.

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

on both calibration and shifted-null test points. This improves resolution by
reducing large self-atoms in weighted conformal p-values and raising calibration
effective sample size.

The cost is that clipping no longer adapts to the true shifted null `Q`. It
instead targets

```text
Q_c(dz) propto min(w(z), c) P(dz).
```

The theorem-facing adaptation cost is therefore the distance between `Q` and
`Q_c`. The summary records the clipped target normalizer, the reference and
shifted mass where `w > c`, and the total variation gap
`TV(Q, Q_c)`.

The plotted frontier makes the trade-off explicit. Tight caps move leftward in
Panel C by lowering max atom mass, but they move upward by increasing oracle
shifted-null tail mismatch. The unclipped point has zero target mismatch but can
have poor finite-sample resolution. Thus clipping changes the
adaptation-resolution frontier; it does not remove it.
