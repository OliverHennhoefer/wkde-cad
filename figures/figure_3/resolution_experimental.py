"""
Resolution Phase Diagram from Experimental Results

Plots actual experimental configurations to show which settings theoretically
allow anomaly detection given Benjamini-Hochberg's lower p-value bound constraint.

Theory:
- With N calibration samples, minimum non-zero p-value = 1/N
- BH rejection: To reject k anomalies at FDR α, need p_k ≤ k·α/m
- Critical line: m = k·α·N
- Below line (green): Detection theoretically possible
- Above line (red): Detection theoretically impossible
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 1. LOAD EXPERIMENTAL DATA
# ==========================================

def load_experiment_configurations():
    """
    Load all experiment result CSVs and extract unique configurations per dataset.
    Returns list of tuples: (dataset_name, N_cal, m_batch, k_min, actual_anom_rate)
    """
    # Path to experiment results
    results_dir = Path(__file__).parent.parent.parent / "outputs" / "experiment_results"

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    csv_files = list(results_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    print(f"Found {len(csv_files)} result files")

    configurations = []

    for csv_file in csv_files:
        # Read CSV
        df = pd.read_csv(csv_file)

        # Filter out summary rows (seed="mean")
        df = df[df["seed"] != "mean"]

        if len(df) == 0:
            continue

        # Get dataset name
        dataset_name = df["dataset"].iloc[0]

        # Extract configuration (same across all rows for a dataset)
        n_cal = df["train_size"].iloc[0]  # Calibration/training size
        m_batch = df["test_size"].iloc[0]  # Test batch size
        k_min = df["n_test_anomaly"].iloc[0]  # Minimum anomalies
        anom_rate = df["actual_anomaly_rate"].iloc[0]  # Actual anomaly rate

        configurations.append((dataset_name, n_cal, m_batch, k_min, anom_rate))

        print(f"  {dataset_name}: N={n_cal}, m={m_batch}, k={k_min}, rate={anom_rate:.3f}")

    if not configurations:
        raise ValueError("No valid configurations found in CSV files")

    return configurations


# ==========================================
# 2. PLOTTING LOGIC
# ==========================================

def plot_resolution_diagram(ax, data, alpha=0.1):
    """
    Plots resolution phase diagram with actual experimental data.

    Args:
        ax: Matplotlib axis object
        data: List of tuples (name, N_cal, m_batch, k_min, anom_rate)
        alpha: Nominal FDR rate (default: 0.1)
    """
    # Extract coordinates
    names = [d[0] for d in data]
    N_vals = np.array([d[1] for d in data])
    m_vals = np.array([d[2] for d in data])
    k_vals = np.array([d[3] for d in data])

    # --- A. Draw Critical Lines (one for each unique k value) ---
    unique_k = sorted(set(k_vals))

    # Define X axis limits
    x_max = max(N_vals) * 1.3
    x_range = np.linspace(0, x_max, 100)

    # Color map for different k values
    colors_k = plt.cm.Set1(np.linspace(0, 1, len(unique_k)))

    for idx, k in enumerate(unique_k):
        slope = k * alpha
        y_critical = slope * x_range

        ax.plot(
            x_range,
            y_critical,
            color=colors_k[idx],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Critical Line $k={k}$ ($m = {slope:.2f}N$)",
        )

    # --- B. Shading Zones (use largest k for shading) ---
    max_k = max(unique_k)
    slope_max = max_k * alpha
    y_critical_max = slope_max * x_range

    y_top_limit = max(m_vals) * 1.4

    # Red Zone (Extrapolation Required / Detection Impossible)
    ax.fill_between(
        x_range,
        y_critical_max,
        y_top_limit,
        color="#ffcccc",
        alpha=0.4,
        label="Extrapolation Zone\n(Detection Impossible)",
    )

    # Green Zone (Interpolation Possible / Detection Possible)
    ax.fill_between(
        x_range,
        0,
        y_critical_max,
        color="#ccffcc",
        alpha=0.4,
        label="Interpolation Zone\n(Detection Possible)",
    )

    # --- C. Plot Dataset Points ---
    for i, (name, n, m, k, rate) in enumerate(data):
        # Color by k value
        k_idx = unique_k.index(k)
        color = colors_k[k_idx]

        # Marker based on whether point is above/below critical line
        critical_m = k * alpha * n
        if m > critical_m:
            marker = "^"  # Triangle up: in red zone (detection impossible)
            size = 120
        else:
            marker = "o"  # Circle: in green zone (detection possible)
            size = 100

        ax.scatter(n, m, color=color, marker=marker, s=size,
                  zorder=10, alpha=0.8)

        # Label all datasets (adjust positioning to avoid overlap)
        offset_y = y_top_limit * 0.03
        offset_x = n * 0.02

        ax.text(
            n + offset_x,
            m + offset_y,
            f"{name}\n$k={k}$",
            fontsize=8,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # --- D. Formatting ---
    ax.set_title(
        f"Experimental Resolution Phase Diagram (Nominal FDR $\\alpha={alpha}$)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Calibration Size ($N_{cal}$ = train_size)", fontsize=10)
    ax.set_ylabel("Test Batch Size ($m$ = test_size)", fontsize=10)
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(bottom=0, top=y_top_limit)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)


# ==========================================
# 3. GENERATE FIGURE
# ==========================================

def main():
    """Main function to generate the plot."""
    print("Loading experimental configurations...")
    data = load_experiment_configurations()

    print(f"\nGenerating plot with {len(data)} datasets...")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot
    plot_resolution_diagram(ax, data, alpha=0.1)

    # Add subtitle with interpretation
    plt.suptitle(
        "Theoretical Detection Feasibility: Points above critical line require extrapolation (KDE)\n"
        "Points below critical line can use standard ECDF-based conformal prediction",
        fontsize=10,
        y=0.98,
    )

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent / "resolution_experimental.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
