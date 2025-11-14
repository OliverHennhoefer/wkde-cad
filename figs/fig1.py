import numpy as np
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from nonconform.strategy import Split
from nonconform.detection import ConformalDetector
from nonconform.utils.data import load, Dataset

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=42)

estimator = ConformalDetector(
    detector=IForest(behaviour="new"), strategy=Split(n_calib=1_000), seed=42
)

estimator.fit(x_train)

calibration_scores = estimator._calibration_set  # noqa
estimates_raw = estimator.predict(x_test, raw=True)
estimates = estimator.predict(x_test)
decisions = false_discovery_control(estimates, method="bh") <= 0.1

# Configure publication-quality matplotlib parameters
SMALL_SIZE = 8
MEDIUM_SIZE = 9
LARGE_SIZE = 10

plt.rc("font", size=SMALL_SIZE, family="sans-serif")
plt.rc("axes", titlesize=MEDIUM_SIZE, labelsize=SMALL_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=LARGE_SIZE)

# Define colorblind-friendly colors
color_calibration = "#DE8F05"  # Orange/yellow-ish
color_test = "#0173B2"  # Blue
color_anomaly = "#CC0000"  # Red for anomalies

# Create 1x3 subplot layout
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.5), dpi=300)

# Panel 1: Overlaid histograms with test split by ground truth
# Split test data by ground truth labels
estimates_raw_normal = estimates_raw[y_test == 0]
estimates_raw_anomaly = estimates_raw[y_test == 1]

# Plot calibration histogram
ax1.hist(
    calibration_scores,
    bins=50,
    alpha=0.5,
    color=color_calibration,
    label="Calibration",
    density=True,
    edgecolor="none",
)

# Plot stacked test histograms (normal + anomaly)
ax1.hist(
    [estimates_raw_normal, estimates_raw_anomaly],
    bins=50,
    alpha=0.5,
    color=[color_test, color_anomaly],
    label=["Test (Normal)", "Test (Anomaly)"],
    density=True,
    edgecolor="none",
    stacked=True,
)

ax1.set_xlabel("Nonconformity Score")
ax1.set_ylabel("Density")
ax1.legend(loc="upper right", frameon=False)
ax1.set_title("(a)", loc="left", fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Panel 2: P-value histogram
ax2.hist(estimates, bins=50, alpha=0.7, color=color_test, edgecolor="none")
ax2.set_xlabel("P-value")
ax2.set_ylabel("Frequency")
ax2.set_title("(b)", loc="left", fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Panel 3: Benjamini-Hochberg visualization
# Sort p-values and corresponding ground truth labels
sorted_idx = np.argsort(estimates)
sorted_pvals = estimates[sorted_idx]
sorted_labels = y_test[sorted_idx]
sorted_decisions = decisions[sorted_idx]

# Create ranks and BH threshold line
n = len(estimates)
ranks = np.arange(1, n + 1)
alpha = 0.1
bh_threshold = (ranks / n) * alpha

# Find zoom range - show up to last discovery + margin
if np.any(sorted_decisions):
    last_discovery_idx = np.where(sorted_decisions)[0][-1]
    zoom_limit = min(n, int(last_discovery_idx * 1.5))  # 50% margin
else:
    zoom_limit = min(n, 1000)  # Default zoom if no discoveries

# Plot by ground truth: anomalies (red) and normal (blue)
mask_anomaly = sorted_labels[:zoom_limit] == 1
mask_normal = sorted_labels[:zoom_limit] == 0

if np.any(mask_anomaly):
    ax3.scatter(
        ranks[:zoom_limit][mask_anomaly],
        sorted_pvals[:zoom_limit][mask_anomaly],
        c=color_anomaly,
        s=10,
        alpha=0.6,
        edgecolor="none",
        label="Anomaly",
        zorder=3,
    )
if np.any(mask_normal):
    ax3.scatter(
        ranks[:zoom_limit][mask_normal],
        sorted_pvals[:zoom_limit][mask_normal],
        c=color_test,
        s=10,
        alpha=0.6,
        edgecolor="none",
        label="Normal",
        zorder=2,
    )

# Plot BH threshold line
ax3.plot(
    ranks[:zoom_limit],
    bh_threshold[:zoom_limit],
    "k--",
    linewidth=1.5,
    alpha=0.8,
    label=f"BH threshold (α={alpha})",
    zorder=1,
)

ax3.set_xlabel("Rank")
ax3.set_ylabel("P-value")
ax3.set_title("(c)", loc="left", fontweight="bold")
ax3.legend(frameon=False, fontsize=7)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Apply tight layout and save
plt.tight_layout()
plt.savefig("fig1.pdf", dpi=300, bbox_inches="tight")
plt.show()
