# Figure 3 Rationale: Randomization Instability Frontier

This figure is a theorem-validation experiment, not a benchmark. Each point is
one fixed world: calibration scores, test scores, calibration weights, and test
weights are frozen. Only the uniforms inside the randomized weighted conformal
p-values are resampled.

Perfect score separation is enforced in every fixed world. Inliers are placed
below the calibration range, so their randomized p-values stay above `alpha` and
cannot drive BH discoveries; high-shift inlier weights are capped only to enforce
this guardrail. Anomalies are placed above the calibration range, so their
intervals are determined only by their weighted self-atoms.

The central diagnostic is the rank interval ratio,

```text
min_r U_(r) / (alpha * r / m),
```

where `U_(r)` is the `r`-th smallest anomaly interval upper endpoint. Larger
values mean the randomization intervals are wide relative to the BH scale. The
frontier panels show that the conditional interval theorem predicts both miss
probability and discovery-count variance across fixed weight worlds.
