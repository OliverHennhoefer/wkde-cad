# Randomized p-value instability

This is a theorem-validation simulation, not an empirical benchmark. Each point
freezes the calibration/test scores and weights; only the uniforms in randomized
weighted conformal p-values are resampled.

Perfect score separation isolates randomization. Inliers lie below the
calibration range and their intervals remain above `alpha`; anomalies lie above
the range, so their intervals contain only weighted self-atoms.

The diagnostic is the rank interval ratio
`min_r U_(r) / (alpha * r / m)`, where `U_(r)` is the `r`-th smallest anomaly
interval upper endpoint. The frontier compares the conditional theorem with
Monte Carlo miss probabilities and discovery-count variances across fixed
weight worlds.
