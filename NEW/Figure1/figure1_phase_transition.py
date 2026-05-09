from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
SCHEMATIC_PATH = OUT_DIR / "figure1_panel_a_schematic.png"
HEATMAP_PATH = OUT_DIR / "figure1_heatmaps_alpha_pi_sensitivity.png"
COLLAPSE_PATH = OUT_DIR / "figure1_collapse_diagnostics.png"
HEATMAP_SUMMARY_PATH = OUT_DIR / "figure1_heatmap_summary.csv"
COLLAPSE_SUMMARY_PATH = OUT_DIR / "figure1_collapse_summary.csv"
SCHEMATIC_POINTS_TIKZ_PATH = OUT_DIR / "figure1_schematic_points_tikz.csv"
SCHEMATIC_ANNOTATIONS_TIKZ_PATH = OUT_DIR / "figure1_schematic_annotations_tikz.csv"
HEATMAP_TIKZ_PATH = OUT_DIR / "figure1_heatmap_tikz.csv"
HEATMAP_BOUNDARY_TIKZ_PATH = OUT_DIR / "figure1_heatmap_boundary_tikz.csv"
COLLAPSE_TIKZ_PATH = OUT_DIR / "figure1_collapse_tikz.csv"
COLLAPSE_REFERENCE_TIKZ_PATH = OUT_DIR / "figure1_collapse_reference_tikz.csv"

SUMMARY_VERSION = "split-v1"
TIKZ_EXPORT_VERSION = "tikz-v1"
SCENARIOS = [
    {"name": "baseline", "alpha": 0.1, "pi1": 0.1, "label": r"$\alpha=0.10,\ \pi_1=0.10$"},
    {"name": "alpha_005", "alpha": 0.05, "pi1": 0.1, "label": r"$\alpha=0.05,\ \pi_1=0.10$"},
    {"name": "pi1_001", "alpha": 0.1, "pi1": 0.01, "label": r"$\alpha=0.10,\ \pi_1=0.01$"},
]
SCENARIO_BY_NAME = {scenario["name"]: scenario for scenario in SCENARIOS}
BASELINE_SCENARIO = "baseline"

N_VALUES = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 4000]
M_VALUES = sorted(
    set(np.rint(np.logspace(np.log10(20), np.log10(2000), 14)).astype(int))
)
RHO_VALUES = np.linspace(0.0, 3.0, 61)
KAPPA_VALUES = [np.inf, 2.0, 2.5, 3.0, 3.5, 4.0]
HEATMAP_KAPPAS = [np.inf, 3.0]
PANEL_C_KAPPA = 3.0
SUPPLEMENT_N_VALUES = [8000, 32000]
SUPPLEMENT_M_VALUES = M_VALUES[:7]
SUPPLEMENT_RHO_VALUES = np.linspace(0.0, 4.0, 61)
N_SEEDS = 100
BASE_SEED = 20260509
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))

HEATMAP_Y_LOWER_QUANTILE = 0.0
HEATMAP_Y_UPPER_QUANTILE = 0.995
HEATMAP_Y_BINS = 34
COLLAPSE_BINS = np.linspace(-2.5, 2.5, 35)

COLORS = {
    "calibration": "#3a3a3a",
    "shifted": "#2b6cb0",
    "anomaly": "#c2410c",
}
KAPPA_COLORS = {
    "inf": "#111111",
    "2.0": "#7c3aed",
    "2.5": "#1d4ed8",
    "3.0": "#0891b2",
    "3.5": "#c2410c",
    "4.0": "#2f855a",
}


def kappa_label(kappa: str | float) -> str:
    if isinstance(kappa, str):
        return "inf" if kappa.lower() == "inf" else f"{float(kappa):.1f}"
    return "inf" if np.isinf(kappa) else f"{kappa:.1f}"


def rng_for(*values: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *values]))


def bh_decisions(p_values: np.ndarray, alpha: float) -> np.ndarray:
    order = np.argsort(p_values, kind="mergesort")
    sorted_p = p_values[order]
    thresholds = alpha * np.arange(1, len(p_values) + 1) / len(p_values)
    passed = sorted_p <= thresholds
    decisions = np.zeros(len(p_values), dtype=bool)
    if np.any(passed):
        cutoff = sorted_p[np.flatnonzero(passed)[-1]]
        decisions = p_values <= cutoff
    return decisions


def weighted_tail_p_values(
    sorted_calib_scores: np.ndarray,
    suffix_calib_weights: np.ndarray,
    total_calib_weight: float,
    test_scores: np.ndarray,
    test_weights: np.ndarray,
) -> np.ndarray:
    # Larger scores are more anomalous, so the tail mass is calibration weight
    # at scores at least as large as the test score.
    tail_start = np.searchsorted(sorted_calib_scores, test_scores, side="left")
    tail_mass = suffix_calib_weights[tail_start]
    return (test_weights + tail_mass) / (test_weights + total_calib_weight)


def x_edges_for_alpha(alpha: float) -> tuple[np.ndarray, dict[int, int]]:
    x_values = np.array([np.log10(m / alpha) for m in M_VALUES], dtype=float)
    midpoints = (x_values[:-1] + x_values[1:]) / 2.0
    x_edges = np.concatenate(
        [
            [x_values[0] - (midpoints[0] - x_values[0])],
            midpoints,
            [x_values[-1] + (x_values[-1] - midpoints[-1])],
        ]
    )
    return x_edges, {m: idx for idx, m in enumerate(M_VALUES)}


def tasks_for_scenario(scenario_name: str) -> list[tuple[str, str, int, float, tuple[int, ...]]]:
    base_tasks = [
        ("base", scenario_name, n, float(rho), tuple(M_VALUES))
        for n in N_VALUES
        for rho in RHO_VALUES
    ]
    supplement_tasks = [
        ("high_resolution_low_burden", scenario_name, n, float(rho), tuple(SUPPLEMENT_M_VALUES))
        for n in SUPPLEMENT_N_VALUES
        for rho in SUPPLEMENT_RHO_VALUES
    ]
    return base_tasks + supplement_tasks


def resolution_block(task: tuple[str, str, int, float, tuple[int, ...]]) -> tuple[str, np.ndarray]:
    _, scenario_name, n, rho, m_values = task
    scenario = SCENARIO_BY_NAME[scenario_name]
    pi1 = float(scenario["pi1"])
    rho_code = int(round(rho * 1000))
    values = []

    for seed in range(N_SEEDS):
        calib_rng = rng_for(n, rho_code, seed, 0)
        calib_z = calib_rng.normal(0.0, 1.0, n)
        calib_weights = np.exp(rho * calib_z - 0.5 * rho**2)
        total_calib_weight = float(np.sum(calib_weights))

        for m in m_values:
            n_anomaly = max(1, int(round(pi1 * m)))
            n_inlier = m - n_anomaly
            test_rng = rng_for(n, m, rho_code, seed, 1)
            _ = test_rng.normal(rho, 1.0, n_inlier)
            anomaly_z = test_rng.normal(rho, 1.0, n_anomaly)
            anomaly_weights = np.exp(rho * anomaly_z - 0.5 * rho**2)
            p_min_anomaly = anomaly_weights / (anomaly_weights + total_calib_weight)
            values.append(np.log10(1.0 / float(np.min(p_min_anomaly))))

    return scenario_name, np.asarray(values, dtype=float)


def compute_y_edges(tasks: list[tuple[str, str, int, float, tuple[int, ...]]]) -> dict[str, np.ndarray]:
    y_values = {scenario["name"]: [] for scenario in SCENARIOS}
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for scenario_name, block_values in executor.map(resolution_block, tasks):
            y_values[scenario_name].append(block_values)

    edges = {}
    for scenario in SCENARIOS:
        scenario_name = scenario["name"]
        values = np.concatenate(y_values[scenario_name])
        lower = float(np.quantile(values, HEATMAP_Y_LOWER_QUANTILE))
        upper = float(np.quantile(values, HEATMAP_Y_UPPER_QUANTILE))
        edges[scenario_name] = np.linspace(lower, upper, HEATMAP_Y_BINS)
    return edges


def add_count(mapping: dict[tuple, list[float]], key: tuple, x: float, success: bool) -> None:
    bucket = mapping.setdefault(key, [0.0, 0.0, 0.0])
    bucket[0] += float(success)
    bucket[1] += 1.0
    bucket[2] += float(x)


def simulate_summary_block(
    task: tuple[str, str, int, float, tuple[int, ...]],
    y_edges_by_scenario: dict[str, np.ndarray],
) -> tuple[dict[tuple, list[float]], dict[tuple, list[float]]]:
    grid_part, scenario_name, n, rho, m_values = task
    scenario = SCENARIO_BY_NAME[scenario_name]
    alpha = float(scenario["alpha"])
    pi1 = float(scenario["pi1"])
    rho_code = int(round(rho * 1000))
    y_edges = y_edges_by_scenario[scenario_name]
    _, x_bin_by_m = x_edges_for_alpha(alpha)

    heatmap_counts: dict[tuple, list[float]] = {}
    collapse_counts: dict[tuple, list[float]] = {}
    kappas_for_task = KAPPA_VALUES if scenario_name == BASELINE_SCENARIO else HEATMAP_KAPPAS

    for seed in range(N_SEEDS):
        calib_rng = rng_for(n, rho_code, seed, 0)
        calib_z = calib_rng.normal(0.0, 1.0, n)
        calib_t = calib_rng.normal(0.0, 1.0, n)
        calib_weights = np.exp(rho * calib_z - 0.5 * rho**2)
        total_calib_weight = float(np.sum(calib_weights))

        order = np.argsort(calib_t, kind="mergesort")
        sorted_calib_scores = calib_t[order]
        sorted_calib_weights = calib_weights[order]
        suffix_calib_weights = np.concatenate(
            ([0.0], np.cumsum(sorted_calib_weights[::-1]))
        )[::-1]

        for m in m_values:
            n_anomaly = max(1, int(round(pi1 * m)))
            n_inlier = m - n_anomaly
            test_rng = rng_for(n, m, rho_code, seed, 1)

            inlier_z = test_rng.normal(rho, 1.0, n_inlier)
            anomaly_z = test_rng.normal(rho, 1.0, n_anomaly)
            test_z = np.concatenate([inlier_z, anomaly_z])
            test_weights = np.exp(rho * test_z - 0.5 * rho**2)
            inlier_scores = test_rng.normal(0.0, 1.0, n_inlier)
            anomaly_noise = test_rng.normal(0.0, 1.0, n_anomaly)
            y_true = np.concatenate(
                [np.zeros(n_inlier, dtype=bool), np.ones(n_anomaly, dtype=bool)]
            )

            p_min_anomaly = test_weights[n_inlier:] / (
                test_weights[n_inlier:] + total_calib_weight
            )
            sorted_p_min_anomaly = np.sort(p_min_anomaly, kind="mergesort")
            anomaly_ranks = np.arange(1, n_anomaly + 1)
            p_min_first = float(sorted_p_min_anomaly[0])
            rank_delta = float(
                np.min(sorted_p_min_anomaly / (alpha * anomaly_ranks / m))
            )
            delta = p_min_first / (alpha / m)
            log_delta = float(np.log10(delta))
            log_rank_delta = float(np.log10(rank_delta))
            log_resolution = float(np.log10(1.0 / p_min_first))
            x_bin = x_bin_by_m[m]
            y_bin = int(np.searchsorted(y_edges, log_resolution, side="right") - 1)
            in_heatmap_y_range = 0 <= y_bin < len(y_edges) - 1

            for kappa in kappas_for_task:
                label = kappa_label(kappa)
                if np.isinf(kappa):
                    anomaly_scores = np.full(n_anomaly, np.inf)
                else:
                    anomaly_scores = kappa + anomaly_noise
                test_scores = np.concatenate([inlier_scores, anomaly_scores])
                p_values = weighted_tail_p_values(
                    sorted_calib_scores,
                    suffix_calib_weights,
                    total_calib_weight,
                    test_scores,
                    test_weights,
                )
                decisions = bh_decisions(p_values, alpha)
                true_discovery = bool(np.any(decisions & y_true))

                if label in {"inf", "3.0"} and in_heatmap_y_range:
                    key = (scenario_name, label, x_bin, y_bin)
                    add_count(heatmap_counts, key, 0.0, true_discovery)

                if scenario_name == BASELINE_SCENARIO:
                    delta_bin = int(np.searchsorted(COLLAPSE_BINS, log_delta, side="right") - 1)
                    if 0 <= delta_bin < len(COLLAPSE_BINS) - 1:
                        key = ("first_threshold", label, delta_bin)
                        add_count(collapse_counts, key, log_delta, true_discovery)

                    rank_bin = int(
                        np.searchsorted(COLLAPSE_BINS, log_rank_delta, side="right") - 1
                    )
                    if 0 <= rank_bin < len(COLLAPSE_BINS) - 1:
                        key = ("rank_aware", label, rank_bin)
                        add_count(collapse_counts, key, log_rank_delta, true_discovery)

    return heatmap_counts, collapse_counts


def merge_counts(target: dict[tuple, list[float]], update: dict[tuple, list[float]]) -> None:
    for key, values in update.items():
        bucket = target.setdefault(key, [0.0, 0.0, 0.0])
        bucket[0] += values[0]
        bucket[1] += values[1]
        bucket[2] += values[2]


def build_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = []
    for scenario in SCENARIOS:
        tasks.extend(tasks_for_scenario(scenario["name"]))

    print("computing heatmap y-axis support", flush=True)
    y_edges_by_scenario = compute_y_edges(tasks)

    heatmap_counts: dict[tuple, list[float]] = {}
    collapse_counts: dict[tuple, list[float]] = {}
    total_tasks = len(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = executor.map(
            simulate_summary_block,
            tasks,
            [y_edges_by_scenario] * len(tasks),
        )
        for done, (heatmap_update, collapse_update) in enumerate(futures, start=1):
            merge_counts(heatmap_counts, heatmap_update)
            merge_counts(collapse_counts, collapse_update)
            if done % len(RHO_VALUES) == 0 or done == total_tasks:
                print(f"completed summary block {done}/{total_tasks}", flush=True)

    heatmap_rows = []
    for (scenario_name, kappa, x_bin, y_bin), values in heatmap_counts.items():
        scenario = SCENARIO_BY_NAME[scenario_name]
        x_edges, _ = x_edges_for_alpha(float(scenario["alpha"]))
        y_edges = y_edges_by_scenario[scenario_name]
        successes, count, _ = values
        heatmap_rows.append(
            {
                "summary_version": SUMMARY_VERSION,
                "scenario": scenario_name,
                "alpha": scenario["alpha"],
                "pi1": scenario["pi1"],
                "kappa": kappa,
                "x_bin": x_bin,
                "y_bin": y_bin,
                "x_left": x_edges[x_bin],
                "x_right": x_edges[x_bin + 1],
                "y_bottom": y_edges[y_bin],
                "y_top": y_edges[y_bin + 1],
                "successes": successes,
                "count": count,
                "probability": successes / count,
            }
        )

    collapse_rows = []
    for (diagnostic, kappa, bin_idx), values in collapse_counts.items():
        successes, count, sum_x = values
        collapse_rows.append(
            {
                "summary_version": SUMMARY_VERSION,
                "scenario": BASELINE_SCENARIO,
                "diagnostic": diagnostic,
                "kappa": kappa,
                "bin": bin_idx,
                "x_left": COLLAPSE_BINS[bin_idx],
                "x_right": COLLAPSE_BINS[bin_idx + 1],
                "x": sum_x / count,
                "successes": successes,
                "count": count,
                "probability": successes / count,
            }
        )

    return pd.DataFrame(heatmap_rows), pd.DataFrame(collapse_rows)


def summaries_are_current() -> bool:
    if not HEATMAP_SUMMARY_PATH.exists() or not COLLAPSE_SUMMARY_PATH.exists():
        return False
    heatmap_header = pd.read_csv(HEATMAP_SUMMARY_PATH, nrows=1)
    collapse_header = pd.read_csv(COLLAPSE_SUMMARY_PATH, nrows=1)
    if heatmap_header.empty or collapse_header.empty:
        return False
    return (
        str(heatmap_header["summary_version"].iloc[0]) == SUMMARY_VERSION
        and str(collapse_header["summary_version"].iloc[0]) == SUMMARY_VERSION
    )


def normalize_summary_keys(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    if "kappa" in summary.columns:
        summary["kappa"] = summary["kappa"].map(kappa_label)
    return summary


def load_or_build_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    if summaries_are_current():
        print("loading existing compact summaries", flush=True)
        return (
            normalize_summary_keys(pd.read_csv(HEATMAP_SUMMARY_PATH)),
            normalize_summary_keys(pd.read_csv(COLLAPSE_SUMMARY_PATH)),
        )

    heatmap_summary, collapse_summary = build_summaries()
    heatmap_summary.to_csv(HEATMAP_SUMMARY_PATH, index=False)
    collapse_summary.to_csv(COLLAPSE_SUMMARY_PATH, index=False)
    return heatmap_summary, collapse_summary


def plot_schematic(ax: plt.Axes) -> None:
    rng = np.random.default_rng(BASE_SEED)
    rho = 1.5
    kappa = 3.5
    n = 300
    m = 500
    pi1 = 0.1
    n_anomaly = int(round(pi1 * m))
    n_inlier = m - n_anomaly

    calib = rng.normal([0.0, 0.0], [1.0, 1.0], size=(n, 2))
    shifted = np.column_stack(
        [rng.normal(rho, 1.0, n_inlier), rng.normal(0.0, 1.0, n_inlier)]
    )
    anomaly = np.column_stack(
        [rng.normal(rho, 1.0, n_anomaly), rng.normal(kappa, 1.0, n_anomaly)]
    )

    ax.scatter(
        calib[:, 0],
        calib[:, 1],
        s=12,
        alpha=0.24,
        color=COLORS["calibration"],
        label=r"$P_0$ calibration",
    )
    ax.scatter(
        shifted[:, 0],
        shifted[:, 1],
        s=13,
        alpha=0.35,
        color=COLORS["shifted"],
        label=r"$Q_\rho$ shifted inliers",
    )
    ax.scatter(
        anomaly[:, 0],
        anomaly[:, 1],
        s=18,
        alpha=0.78,
        color=COLORS["anomaly"],
        label=r"$A_{\rho,\kappa}$ anomalies",
    )
    ax.annotate(
        "benign covariate shift",
        xy=(rho, -2.7),
        xytext=(0.1, -2.7),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "0.25"},
        ha="left",
        va="center",
        fontsize=9,
    )
    ax.annotate(
        "anomaly direction",
        xy=(rho + 2.35, kappa),
        xytext=(rho + 2.35, 0.15),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "0.25"},
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=9,
    )
    ax.text(-3.1, 4.85, r"score $S=T$", fontsize=9)
    ax.text(-3.1, 4.45, r"weights depend on $Z$", fontsize=9)
    ax.set_title("Controlled Gaussian shift")
    ax.set_xlabel(r"benign shift direction $Z$")
    ax.set_ylabel(r"anomaly-score direction $T$")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.18, linewidth=0.6)


def plot_panel_a() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    plot_schematic(ax)
    fig.savefig(SCHEMATIC_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def heatmap_matrix(summary: pd.DataFrame, scenario: str, kappa: str) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    block = summary[(summary["scenario"].eq(scenario)) & (summary["kappa"].eq(kappa))]
    x_edges = np.unique(np.concatenate([block["x_left"].to_numpy(), block["x_right"].to_numpy()]))
    y_edges = np.unique(np.concatenate([block["y_bottom"].to_numpy(), block["y_top"].to_numpy()]))
    matrix = np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan)
    for row in block.itertuples(index=False):
        matrix[int(row.x_bin), int(row.y_bin)] = float(row.probability)
    return x_edges, y_edges, np.ma.masked_invalid(matrix)


def plot_heatmap_grid(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.6, 12.8),
        constrained_layout=True,
        sharey=False,
    )
    column_specs = [("inf", "Perfect-score exact WEDF"), ("3.0", r"Finite-score exact WEDF ($\kappa=3.0$)")]
    mesh = None
    for row_idx, scenario in enumerate(SCENARIOS):
        for col_idx, (kappa, title) in enumerate(column_specs):
            ax = axes[row_idx, col_idx]
            x_edges, y_edges, matrix = heatmap_matrix(summary, scenario["name"], kappa)
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                matrix.T,
                cmap="viridis",
                norm=Normalize(vmin=0.0, vmax=1.0),
                shading="flat",
            )
            diagonal_x = np.linspace(float(x_edges[0]), float(x_edges[-1]), 200)
            ax.plot(diagonal_x, diagonal_x, color="black", linewidth=1.1, linestyle="--")
            ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
            ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
            if row_idx == 0:
                ax.set_title(title)
            ax.set_xlabel(r"$\log_{10}(m/\alpha)$")
            ax.set_ylabel(r"$\log_{10}(1/p^{\min}_{(1)})$")
            ax.grid(alpha=0.18, linewidth=0.6)
            ax.text(
                0.02,
                0.96,
                scenario["label"],
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("Pr(BH finds >=1 anomaly)")
    fig.suptitle("Finite-sample detectability under alpha and anomaly-rate changes", fontsize=14)
    fig.savefig(HEATMAP_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_collapse_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    diagnostic: str,
    title: str,
    xlabel: str,
) -> None:
    block = summary[summary["diagnostic"].eq(diagnostic)].copy()
    for kappa in ["inf", "2.0", "2.5", "3.0", "3.5", "4.0"]:
        line = block[block["kappa"].eq(kappa)].sort_values("x")
        kappa_display = r"\infty" if kappa == "inf" else kappa
        ax.plot(
            line["x"],
            line["probability"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=KAPPA_COLORS[kappa],
            label=rf"$\kappa={kappa_display}$",
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pr(BH finds >=1 anomaly)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")


def plot_collapse_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), constrained_layout=True)
    plot_collapse_panel(
        axes[0],
        summary,
        "first_threshold",
        "D. First-threshold diagnostic",
        r"$\log_{10}\delta$,  $\delta=p^{\min}_{(1)}/(\alpha/m)$",
    )
    plot_collapse_panel(
        axes[1],
        summary,
        "rank_aware",
        "E. Rank-aware BH-scale diagnostic",
        r"$\log_{10}\Delta_{\mathrm{BH}}$,  "
        r"$\Delta_{\mathrm{BH}}=\min_r p^{\min}_{(r)}/((r/m)\alpha)$",
    )
    fig.suptitle(r"Baseline collapse diagnostics ($\alpha=0.10,\ \pi_1=0.10$)", fontsize=14)
    fig.savefig(COLLAPSE_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def export_schematic_tikz() -> None:
    rng = np.random.default_rng(BASE_SEED)
    rho = 1.5
    kappa = 3.5
    n = 300
    m = 500
    pi1 = 0.1
    n_anomaly = int(round(pi1 * m))
    n_inlier = m - n_anomaly

    calib = rng.normal([0.0, 0.0], [1.0, 1.0], size=(n, 2))
    shifted = np.column_stack(
        [rng.normal(rho, 1.0, n_inlier), rng.normal(0.0, 1.0, n_inlier)]
    )
    anomaly = np.column_stack(
        [rng.normal(rho, 1.0, n_anomaly), rng.normal(kappa, 1.0, n_anomaly)]
    )

    point_rows = []
    for group, label, style_key, points, plot_order in [
        ("calibration", r"$P_0$ calibration", "calibration", calib, 1),
        ("shifted_inlier", r"$Q_\rho$ shifted inliers", "shifted", shifted, 2),
        ("anomaly", r"$A_{\rho,\kappa}$ anomalies", "anomaly", anomaly, 3),
    ]:
        for idx, (x, y) in enumerate(points):
            point_rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure1",
                    "panel": "A",
                    "panel_title": "Controlled Gaussian shift",
                    "plot_order": plot_order,
                    "point_index": idx,
                    "group": group,
                    "method": "",
                    "scenario": "",
                    "kappa": "",
                    "m": "",
                    "x": x,
                    "y": y,
                    "value": "",
                    "probability": "",
                    "count": "",
                    "style_key": style_key,
                    "label": label,
                }
            )
    pd.DataFrame(point_rows).to_csv(SCHEMATIC_POINTS_TIKZ_PATH, index=False)

    annotation_rows = [
        {
            "export_version": TIKZ_EXPORT_VERSION,
            "figure": "figure1",
            "panel": "A",
            "panel_title": "Controlled Gaussian shift",
            "plot_order": 1,
            "object_type": "arrow",
            "x": 0.1,
            "y": -2.7,
            "x_end": rho,
            "y_end": -2.7,
            "style_key": "covariate_shift_arrow",
            "label": "benign covariate shift",
        },
        {
            "export_version": TIKZ_EXPORT_VERSION,
            "figure": "figure1",
            "panel": "A",
            "panel_title": "Controlled Gaussian shift",
            "plot_order": 2,
            "object_type": "arrow",
            "x": rho + 2.35,
            "y": 0.15,
            "x_end": rho + 2.35,
            "y_end": kappa,
            "style_key": "anomaly_direction_arrow",
            "label": "anomaly direction",
        },
        {
            "export_version": TIKZ_EXPORT_VERSION,
            "figure": "figure1",
            "panel": "A",
            "panel_title": "Controlled Gaussian shift",
            "plot_order": 3,
            "object_type": "text",
            "x": -3.1,
            "y": 4.85,
            "x_end": "",
            "y_end": "",
            "style_key": "text",
            "label": r"score $S=T$",
        },
        {
            "export_version": TIKZ_EXPORT_VERSION,
            "figure": "figure1",
            "panel": "A",
            "panel_title": "Controlled Gaussian shift",
            "plot_order": 4,
            "object_type": "text",
            "x": -3.1,
            "y": 4.45,
            "x_end": "",
            "y_end": "",
            "style_key": "text",
            "label": r"weights depend on $Z$",
        },
    ]
    pd.DataFrame(annotation_rows).to_csv(SCHEMATIC_ANNOTATIONS_TIKZ_PATH, index=False)


def export_heatmap_tikz(summary: pd.DataFrame) -> None:
    column_specs = [
        ("inf", "Perfect-score exact WEDF"),
        ("3.0", r"Finite-score exact WEDF ($\kappa=3.0$)"),
    ]
    rows = []
    boundary_rows = []
    for row_idx, scenario in enumerate(SCENARIOS):
        for col_idx, (kappa, title) in enumerate(column_specs):
            block = summary[
                summary["scenario"].eq(scenario["name"]) & summary["kappa"].eq(kappa)
            ].copy()
            panel = f"heatmap_r{row_idx + 1}_c{col_idx + 1}"
            block["export_version"] = TIKZ_EXPORT_VERSION
            block["figure"] = "figure1"
            block["panel"] = panel
            block["panel_title"] = title
            block["plot_order"] = row_idx * len(column_specs) + col_idx + 1
            block["group"] = "heatmap_cell"
            block["method"] = "exact_wedf"
            block["m"] = ""
            block["x"] = (block["x_left"] + block["x_right"]) / 2.0
            block["y"] = (block["y_bottom"] + block["y_top"]) / 2.0
            block["value"] = block["probability"]
            block["style_key"] = "discovery_probability"
            block["label"] = scenario["label"]
            rows.append(
                block[
                    [
                        "export_version",
                        "figure",
                        "panel",
                        "panel_title",
                        "plot_order",
                        "group",
                        "method",
                        "scenario",
                        "kappa",
                        "m",
                        "x",
                        "y",
                        "x_left",
                        "x_right",
                        "y_bottom",
                        "y_top",
                        "value",
                        "probability",
                        "count",
                        "style_key",
                        "label",
                    ]
                ]
            )

            x_edges = np.unique(
                np.concatenate([block["x_left"].to_numpy(), block["x_right"].to_numpy()])
            )
            for point_order, x in enumerate([float(x_edges[0]), float(x_edges[-1])], start=1):
                boundary_rows.append(
                    {
                        "export_version": TIKZ_EXPORT_VERSION,
                        "figure": "figure1",
                        "panel": panel,
                        "panel_title": title,
                        "plot_order": point_order,
                        "group": "bh_boundary",
                        "method": "",
                        "scenario": scenario["name"],
                        "kappa": kappa,
                        "m": "",
                        "x": x,
                        "y": x,
                        "value": "",
                        "probability": "",
                        "count": "",
                        "style_key": "diagonal_y_equals_x",
                        "label": r"$1/p=m/\alpha$",
                    }
                )

    pd.concat(rows, ignore_index=True).to_csv(HEATMAP_TIKZ_PATH, index=False)
    pd.DataFrame(boundary_rows).to_csv(HEATMAP_BOUNDARY_TIKZ_PATH, index=False)


def export_collapse_tikz(summary: pd.DataFrame) -> None:
    panel_specs = [
        (
            "D",
            "first_threshold",
            "First-threshold diagnostic",
            r"$\log_{10}\delta$",
        ),
        (
            "E",
            "rank_aware",
            "Rank-aware BH-scale diagnostic",
            r"$\log_{10}\Delta_{\mathrm{BH}}$",
        ),
    ]
    rows = []
    reference_rows = []
    for panel_order, (panel, diagnostic, title, label) in enumerate(panel_specs, start=1):
        block = summary[summary["diagnostic"].eq(diagnostic)].copy()
        block["export_version"] = TIKZ_EXPORT_VERSION
        block["figure"] = "figure1"
        block["panel"] = panel
        block["panel_title"] = title
        block["plot_order"] = block["kappa"].map(
            {kappa: idx + 1 for idx, kappa in enumerate(["inf", "2.0", "2.5", "3.0", "3.5", "4.0"])}
        )
        block["group"] = diagnostic
        block["method"] = "exact_wedf"
        block["m"] = ""
        block["y"] = block["probability"]
        block["value"] = block["probability"]
        block["style_key"] = "kappa_" + block["kappa"].astype(str)
        block["label"] = label
        rows.append(
            block[
                [
                    "export_version",
                    "figure",
                    "panel",
                    "panel_title",
                    "plot_order",
                    "group",
                    "method",
                    "scenario",
                    "kappa",
                    "m",
                    "x",
                    "y",
                    "x_left",
                    "x_right",
                    "value",
                    "probability",
                    "count",
                    "style_key",
                    "label",
                ]
            ]
        )
        for point_order, y in enumerate([-0.03, 1.03], start=1):
            reference_rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure1",
                    "panel": panel,
                    "panel_title": title,
                    "plot_order": panel_order * 10 + point_order,
                    "group": "reference",
                    "method": "",
                    "scenario": BASELINE_SCENARIO,
                    "kappa": "",
                    "m": "",
                    "x": 0.0,
                    "y": y,
                    "value": "",
                    "probability": "",
                    "count": "",
                    "style_key": "vertical_zero",
                    "label": label,
                }
            )
    pd.concat(rows, ignore_index=True).to_csv(COLLAPSE_TIKZ_PATH, index=False)
    pd.DataFrame(reference_rows).to_csv(COLLAPSE_REFERENCE_TIKZ_PATH, index=False)


def export_tikz_csvs(heatmap_summary: pd.DataFrame, collapse_summary: pd.DataFrame) -> None:
    export_schematic_tikz()
    export_heatmap_tikz(heatmap_summary)
    export_collapse_tikz(collapse_summary)


def validate_outputs(heatmap_summary: pd.DataFrame, collapse_summary: pd.DataFrame) -> None:
    expected_scenarios = {scenario["name"] for scenario in SCENARIOS}
    observed_scenarios = set(heatmap_summary["scenario"].unique())
    if observed_scenarios != expected_scenarios:
        raise RuntimeError(f"Unexpected heatmap scenarios: {observed_scenarios}")
    if set(collapse_summary["scenario"].unique()) != {BASELINE_SCENARIO}:
        raise RuntimeError("Collapse summary must contain only the baseline scenario.")
    for path in [
        SCHEMATIC_PATH,
        HEATMAP_PATH,
        COLLAPSE_PATH,
        SCHEMATIC_POINTS_TIKZ_PATH,
        SCHEMATIC_ANNOTATIONS_TIKZ_PATH,
        HEATMAP_TIKZ_PATH,
        HEATMAP_BOUNDARY_TIKZ_PATH,
        COLLAPSE_TIKZ_PATH,
        COLLAPSE_REFERENCE_TIKZ_PATH,
    ]:
        if not path.exists():
            raise RuntimeError(f"Missing output: {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )
    heatmap_summary, collapse_summary = load_or_build_summaries()
    plot_panel_a()
    plot_heatmap_grid(heatmap_summary)
    plot_collapse_figure(collapse_summary)
    export_tikz_csvs(heatmap_summary, collapse_summary)
    validate_outputs(heatmap_summary, collapse_summary)
    print(SCHEMATIC_PATH)
    print(HEATMAP_PATH)
    print(COLLAPSE_PATH)
    print(SCHEMATIC_POINTS_TIKZ_PATH)
    print(SCHEMATIC_ANNOTATIONS_TIKZ_PATH)
    print(HEATMAP_TIKZ_PATH)
    print(HEATMAP_BOUNDARY_TIKZ_PATH)
    print(COLLAPSE_TIKZ_PATH)
    print(COLLAPSE_REFERENCE_TIKZ_PATH)


if __name__ == "__main__":
    main()
