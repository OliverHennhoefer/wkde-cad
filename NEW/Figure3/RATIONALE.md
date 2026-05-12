# Figure 3 Rationale: Randomized P-Value Instability

This figure conditions on one fixed set of scores and weights, then resamples
only the uniforms used in randomized weighted conformal p-values. It isolates
the variability caused by randomizing the test self-atom.

For the anomaly points, scores are placed above the calibration range, so the
calibration tail mass is zero and the randomized WEDF p-value is uniform on
`[0, w_j / (W_cal + w_j)]`. For inliers, scores are placed below the calibration
range, so their randomized p-values stay above `alpha` and cannot drive BH.

The plotted theorem curve is conditional on the frozen intervals. Each anomaly
p-value is assigned to the BH threshold bins `alpha * k / m`; a dynamic program
over these independent categorical variables gives the exact distribution of
the BH anomaly discovery count. The observed histogram should match this curve
up to Monte Carlo error.

The summary also records observed and theorem means and variances of the BH
discovery count, which makes the randomization-driven variance inflation visible
without changing the fixed score separation or fixed weights.
