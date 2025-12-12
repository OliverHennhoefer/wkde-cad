"""Figure 2: Visualization of Weighted vs Unweighted EDF and KDE Methods.

This figure demonstrates the effect of density ratio weighting on:
- Empirical Distribution Function (EDF) - histogram representation
- Kernel Density Estimation (KDE) - smooth density curves

2x2 grid layout for two-column LaTeX format.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d
from KDEpy import FFTKDE

# Set seed for reproducibility
np.random.seed(42)

# ============================================================================
# Generate synthetic data to clearly demonstrate weighting effects
# ============================================================================
# Calibration distribution: mixture of two gaussians (bimodal)
n_calib = 500
calib_component1 = np.random.normal(loc=1.5, scale=0.4, size=int(n_calib * 0.7))
calib_component2 = np.random.normal(loc=3.5, scale=0.5, size=int(n_calib * 0.3))
calibration_scores = np.concatenate([calib_component1, calib_component2])

# For weighted conformal: simulate covariate shift
# Higher weights for calibration points that are more "test-like"
# (i.e., closer to where anomalies might appear - higher scores)
def compute_density_ratio_weights(scores, test_score):
    """Simulate density ratio weights: higher weight for scores closer to test region."""
    # Simulate weights as if test distribution is shifted towards higher scores
    # w(x) proportional to exp(-distance_to_high_region)
    high_region_center = 4.0
    distances = np.abs(scores - high_region_center)
    raw_weights = np.exp(-0.5 * distances)
    # Normalize and clip for stability (similar to nonconform's approach)
    weights = raw_weights / np.mean(raw_weights)
    weights = np.clip(weights, 0.35, 45.0)
    return weights

# Select a test point in the "interesting" region (between modes)
test_score = 2.8

# Compute weights for demonstration
calib_weights = compute_density_ratio_weights(calibration_scores, test_score)
test_weight = compute_density_ratio_weights(np.array([test_score]), test_score)[0]

# ============================================================================
# Configure publication-quality matplotlib parameters
# ============================================================================
SMALL_SIZE = 8
MEDIUM_SIZE = 9
LARGE_SIZE = 10

plt.rc("font", size=SMALL_SIZE, family="serif")
plt.rc("mathtext", fontset="cm")
plt.rc("axes", titlesize=MEDIUM_SIZE, labelsize=SMALL_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=LARGE_SIZE)

# Define colorblind-friendly colors
color_calibration = "#DE8F05"  # Orange - calibration distribution
color_test = "#0173B2"  # Blue - test point
color_weighted = "#029E73"  # Green - weighted distribution

# Create 2x2 subplot layout (width > height for two-column LaTeX)
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5), dpi=300)
(ax1, ax2), (ax3, ax4) = axes

# Common histogram bins for consistency
n_bins = 25
hist_range = (0, 5.5)

# ============================================================================
# Panel (a): Unweighted EDF (Histogram)
# ============================================================================
ax1.hist(
    calibration_scores,
    bins=n_bins,
    range=hist_range,
    alpha=0.7,
    color=color_calibration,
    edgecolor="white",
    linewidth=0.5,
    density=True,
    label="Calibration",
)

# Mark test point
ax1.axvline(test_score, color=color_test, linewidth=2, linestyle="-", label="Test point")

# Compute unweighted p-value
n_cal = len(calibration_scores)
count_ge = np.sum(calibration_scores >= test_score)
p_unweighted = (1.0 + count_ge) / (1.0 + n_cal)

# Shade area to the right of test point (representing p-value)
y_max = ax1.get_ylim()[1]
ax1.fill_betweenx([0, y_max * 0.95], test_score, hist_range[1],
                   alpha=0.2, color=color_test, zorder=0)

ax1.text(test_score + 0.15, y_max * 0.75, f"$p = {p_unweighted:.3f}$",
         fontsize=9, color=color_test, fontweight="bold")

ax1.set_xlabel("Nonconformity Score")
ax1.set_ylabel("Density")
ax1.set_title("(a) Unweighted Empirical", loc="left", fontweight="bold")
ax1.legend(loc="upper right", frameon=False, fontsize=7)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_xlim(hist_range)

# ============================================================================
# Panel (b): Weighted EDF (Histogram)
# ============================================================================
# Normalize weights for histogram display
norm_weights = calib_weights / calib_weights.sum() * len(calib_weights)

ax2.hist(
    calibration_scores,
    bins=n_bins,
    range=hist_range,
    weights=norm_weights,
    alpha=0.7,
    color=color_weighted,
    edgecolor="white",
    linewidth=0.5,
    density=True,
    label="Weighted Calib.",
)

# Mark test point
ax2.axvline(test_score, color=color_test, linewidth=2, linestyle="-", label="Test point")

# Compute weighted p-value
mask_ge = calibration_scores >= test_score
weighted_sum_ge = np.sum(calib_weights[mask_ge])
p_weighted = (weighted_sum_ge + test_weight) / (np.sum(calib_weights) + test_weight)

# Shade area to the right of test point
y_max = ax2.get_ylim()[1]
ax2.fill_betweenx([0, y_max * 0.95], test_score, hist_range[1],
                   alpha=0.2, color=color_test, zorder=0)

ax2.text(test_score + 0.15, y_max * 0.75, f"$p = {p_weighted:.3f}$",
         fontsize=9, color=color_test, fontweight="bold")

ax2.set_xlabel("Nonconformity Score")
ax2.set_ylabel("Density")
ax2.set_title("(b) Weighted Empirical", loc="left", fontweight="bold")
ax2.legend(loc="upper right", frameon=False, fontsize=7)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xlim(hist_range)

# ============================================================================
# Panel (c): Unweighted KDE
# ============================================================================
# Compute bandwidth using Scott's rule on calibration scores
# Use fixed bandwidth for both weighted and unweighted for fair comparison
std_cal = np.std(calibration_scores)
n_cal_kde = len(calibration_scores)
scott_bw = 1.06 * std_cal * n_cal_kde ** (-1/5)

# Fit unweighted KDE
kde_unweighted = FFTKDE(kernel="gaussian", bw=scott_bw)
kde_unweighted.fit(calibration_scores)
x_grid, pdf_unweighted = kde_unweighted.evaluate(2**12)

# Compute CDF via numerical integration
cdf_unweighted = integrate.cumulative_trapezoid(pdf_unweighted, x_grid, initial=0)
cdf_unweighted = cdf_unweighted / cdf_unweighted[-1]  # Normalize

# Interpolate CDF for test point
cdf_func = interp1d(x_grid, cdf_unweighted, kind="linear", bounds_error=False, fill_value=(0, 1))
p_kde_unweighted = 1.0 - cdf_func(test_score)  # Survival function

# Plot PDF
ax3.plot(x_grid, pdf_unweighted, color=color_calibration, linewidth=1.5, label="KDE")
ax3.fill_between(x_grid, pdf_unweighted, alpha=0.3, color=color_calibration)

# Shade survival area (x >= test_score)
mask_survival = x_grid >= test_score
ax3.fill_between(x_grid[mask_survival], pdf_unweighted[mask_survival],
                  alpha=0.5, color=color_test, label=f"$P(X \\geq x)$")

# Mark test point
ax3.axvline(test_score, color=color_test, linewidth=2, linestyle="-")

# Annotate p-value
y_max = ax3.get_ylim()[1]
ax3.text(test_score + 0.15, y_max * 0.75, f"$p = {float(p_kde_unweighted):.3f}$",
         fontsize=9, color=color_test, fontweight="bold")

ax3.set_xlabel("Nonconformity Score")
ax3.set_ylabel("Density")
ax3.set_title("(c) Unweighted Probabilistic", loc="left", fontweight="bold")
ax3.legend(loc="upper right", frameon=False, fontsize=7)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_xlim(hist_range)

# ============================================================================
# Panel (d): Weighted KDE
# ============================================================================
# Fit weighted KDE (using same bandwidth as unweighted for fair comparison)
kde_weighted = FFTKDE(kernel="gaussian", bw=scott_bw)
kde_weighted.fit(calibration_scores, weights=calib_weights)
x_grid_w, pdf_weighted = kde_weighted.evaluate(2**12)

# Compute CDF via numerical integration
cdf_weighted = integrate.cumulative_trapezoid(pdf_weighted, x_grid_w, initial=0)
cdf_weighted = cdf_weighted / cdf_weighted[-1]  # Normalize

# Interpolate CDF for test point
cdf_func_w = interp1d(x_grid_w, cdf_weighted, kind="linear", bounds_error=False, fill_value=(0, 1))
p_kde_weighted = 1.0 - cdf_func_w(test_score)  # Survival function

# Plot PDF
ax4.plot(x_grid_w, pdf_weighted, color=color_weighted, linewidth=1.5, label="Weighted KDE")
ax4.fill_between(x_grid_w, pdf_weighted, alpha=0.3, color=color_weighted)

# Shade survival area (x >= test_score)
mask_survival_w = x_grid_w >= test_score
ax4.fill_between(x_grid_w[mask_survival_w], pdf_weighted[mask_survival_w],
                  alpha=0.5, color=color_test, label=f"$P(X \\geq x)$")

# Mark test point
ax4.axvline(test_score, color=color_test, linewidth=2, linestyle="-")

# Annotate p-value
y_max = ax4.get_ylim()[1]
ax4.text(test_score + 0.15, y_max * 0.75, f"$p = {float(p_kde_weighted):.3f}$",
         fontsize=9, color=color_test, fontweight="bold")

ax4.set_xlabel("Nonconformity Score")
ax4.set_ylabel("Density")
ax4.set_title("(d) Weighted Probabilistic", loc="left", fontweight="bold")
ax4.legend(loc="upper right", frameon=False, fontsize=7)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.set_xlim(hist_range)

# ============================================================================
# Final adjustments
# ============================================================================
plt.tight_layout()
plt.savefig("figures/figure_2/figure_2.pdf", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nP-value comparison for test score = {test_score:.3f}:")
print(f"  Unweighted Empirical: {p_unweighted:.4f}")
print(f"  Weighted Empirical:   {p_weighted:.4f}")
print(f"  Unweighted KDE:       {float(p_kde_unweighted):.4f}")
print(f"  Weighted KDE:         {float(p_kde_weighted):.4f}")
print(f"\nWeighting shifts mass towards higher scores, making the test point")
print(f"appear less extreme (higher p-value in weighted methods).")
