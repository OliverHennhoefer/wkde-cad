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
from scipy.stats import norm


OUT_DIR = Path(__file__).resolve().parent
FIGURE_PATH = OUT_DIR / "figure2_perfect_score_resolution.png"
SUMMARY_PATH = OUT_DIR / "figure2_perfect_score_summary.csv"
TABLE_PATH = OUT_DIR / "figure2_key_settings_table.csv"
POWER_CONFIG_FIGURE_PATH = OUT_DIR / "figure2_power_configurations.png"
POWER_CONFIG_SUMMARY_PATH = OUT_DIR / "figure2_power_configurations_summary.csv"
PANEL_A_SCORES_TIKZ_PATH = OUT_DIR / "figure2_panel_a_scores_tikz.csv"
PANEL_A_ANNOTATIONS_TIKZ_PATH = OUT_DIR / "figure2_panel_a_annotations_tikz.csv"
PANEL_B_HEATMAP_TIKZ_PATH = OUT_DIR / "figure2_panel_b_heatmap_tikz.csv"
PANEL_B_BOUNDARY_TIKZ_PATH = OUT_DIR / "figure2_panel_b_boundary_tikz.csv"
PANEL_C_POWER_TIKZ_PATH = OUT_DIR / "figure2_panel_c_power_tikz.csv"
PANEL_D_DETECTABILITY_TIKZ_PATH = OUT_DIR / "figure2_panel_d_detectability_tikz.csv"
PANEL_D_REFERENCE_TIKZ_PATH = OUT_DIR / "figure2_panel_d_reference_tikz.csv"
POWER_CONFIGURATIONS_TIKZ_PATH = OUT_DIR / "figure2_power_configurations_tikz.csv"

SUMMARY_VERSION = "perfect-score-v1"
POWER_CONFIG_SUMMARY_VERSION = "power-config-v1"
TIKZ_EXPORT_VERSION = "tikz-v1"
BASE_SEED = 20260509
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))

ALPHA = 0.10
PI1 = 0.05
DELTA_SCORE = 1.0
D = 10
N_VALUES = [50, 75, 100, 150, 200, 300, 500, 750, 1000]
M_VALUES = [50, 100, 200, 500, 1000, 2000]
RHO_VALUES = np.linspace(0.0, 2.5, 51)
N_SEEDS = 100

PANEL_C_M_VALUES = [50, 100, 500, 1000, 2000]
DELTA_BINS = np.linspace(-2.5, 2.5, 31)
LOG_ESS_BINS = np.linspace(1.0, 3.1, 22)
Y_BINS = np.linspace(0.0, 4.6, 47)
POWER_CONFIGS = [
    {
        "name": "alpha_010_pi1_005",
        "alpha": 0.10,
        "pi1": 0.05,
        "label": r"$\alpha=0.10,\ \pi_1=0.05$",
    },
    {
        "name": "alpha_005_pi1_005",
        "alpha": 0.05,
        "pi1": 0.05,
        "label": r"$\alpha=0.05,\ \pi_1=0.05$",
    },
    {
        "name": "alpha_010_pi1_001",
        "alpha": 0.10,
        "pi1": 0.01,
        "label": r"$\alpha=0.10,\ \pi_1=0.01$",
    },
    {
        "name": "alpha_005_pi1_001",
        "alpha": 0.05,
        "pi1": 0.01,
        "label": r"$\alpha=0.05,\ \pi_1=0.01$",
    },
]
POWER_PI_VALUES = sorted({float(config["pi1"]) for config in POWER_CONFIGS})

COLORS = {
    "calibration": "#4a4a4a",
    "inlier": "#2563eb",
    "anomaly": "#dc2626",
    "exact_wedf": "#111111",
    "randomized_wedf": "#7c3aed",
    "oracle_continuous_tail": "#0f766e",
}

METHOD_LABELS = {
    "exact_wedf": "Exact WEDF",
    "randomized_wedf": "Randomized WEDF",
    "oracle_continuous_tail": "Oracle continuous tail",
}


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


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights**2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(weights) ** 2 / denominator)


def weighted_tail_p_values(
    sorted_calib_scores: np.ndarray,
    suffix_calib_weights: np.ndarray,
    total_calib_weight: float,
    test_scores: np.ndarray,
    test_weights: np.ndarray,
) -> np.ndarray:
    tail_start = np.searchsorted(sorted_calib_scores, test_scores, side="left")
    tail_mass = suffix_calib_weights[tail_start]
    return (test_weights + tail_mass) / (test_weights + total_calib_weight)


def randomized_weighted_tail_p_values(
    sorted_calib_scores: np.ndarray,
    suffix_calib_weights: np.ndarray,
    total_calib_weight: float,
    test_scores: np.ndarray,
    test_weights: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    tail_start = np.searchsorted(sorted_calib_scores, test_scores, side="left")
    tail_mass = suffix_calib_weights[tail_start]
    return (uniforms * test_weights + tail_mass) / (test_weights + total_calib_weight)


def oracle_continuous_tail_p_values(test_scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    p_values = norm.sf(test_scores)
    p_values[y_true] = 0.0
    return p_values


def discovery_metrics(
    p_values: np.ndarray,
    y_true: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[bool, float, float]:
    decisions = bh_decisions(p_values, alpha)
    n_rejections = int(np.sum(decisions))
    true_rejections = int(np.sum(decisions & y_true))
    false_rejections = n_rejections - true_rejections
    n_anomaly = int(np.sum(y_true))
    power = true_rejections / n_anomaly if n_anomaly else 0.0
    fdr = false_rejections / n_rejections if n_rejections else 0.0
    return bool(true_rejections > 0), float(power), float(fdr)


def empirical_auc(anomaly_score: float, inlier_scores: np.ndarray) -> float:
    less = float(np.sum(inlier_scores < anomaly_score))
    ties = float(np.sum(inlier_scores == anomaly_score))
    return (less + 0.5 * ties) / len(inlier_scores)


def simulate_block(task: tuple[int, float]) -> list[dict[str, float | int | str]]:
    n_cal, rho = task
    rho_code = int(round(rho * 1000))
    rows: list[dict[str, float | int | str]] = []

    for seed in range(N_SEEDS):
        calib_rng = rng_for(n_cal, rho_code, seed, 0)
        calib_x = calib_rng.normal(0.0, 1.0, size=(n_cal, D))
        calib_scores = calib_rng.normal(0.0, 1.0, n_cal)
        calib_weights = np.exp(rho * calib_x[:, 0] - 0.5 * rho**2)
        total_calib_weight = float(np.sum(calib_weights))
        calib_ess = effective_sample_size(calib_weights)
        max_normalized_weight = float(np.max(calib_weights) / total_calib_weight)

        order = np.argsort(calib_scores, kind="mergesort")
        sorted_calib_scores = calib_scores[order]
        sorted_calib_weights = calib_weights[order]
        suffix_calib_weights = np.concatenate(
            ([0.0], np.cumsum(sorted_calib_weights[::-1]))
        )[::-1]
        max_calib_score = float(np.max(calib_scores))

        for m in M_VALUES:
            n_anomaly = max(1, int(round(PI1 * m)))
            n_inlier = m - n_anomaly
            test_rng = rng_for(n_cal, m, rho_code, seed, 1)
            inlier_x = test_rng.normal(0.0, 1.0, size=(n_inlier, D))
            anomaly_x = test_rng.normal(0.0, 1.0, size=(n_anomaly, D))
            inlier_x[:, 0] += rho
            anomaly_x[:, 0] += rho
            test_x = np.vstack([inlier_x, anomaly_x])
            test_weights = np.exp(rho * test_x[:, 0] - 0.5 * rho**2)

            inlier_scores = test_rng.normal(0.0, 1.0, n_inlier)
            anomaly_scores = np.full(n_anomaly, max_calib_score + DELTA_SCORE)
            test_scores = np.concatenate([inlier_scores, anomaly_scores])
            y_true = np.concatenate(
                [np.zeros(n_inlier, dtype=bool), np.ones(n_anomaly, dtype=bool)]
            )

            exact_p_values = weighted_tail_p_values(
                sorted_calib_scores,
                suffix_calib_weights,
                total_calib_weight,
                test_scores,
                test_weights,
            )
            random_p_values = randomized_weighted_tail_p_values(
                sorted_calib_scores,
                suffix_calib_weights,
                total_calib_weight,
                test_scores,
                test_weights,
                test_rng.random(m),
            )
            oracle_p_values = oracle_continuous_tail_p_values(test_scores, y_true)

            anomaly_weights = test_weights[n_inlier:]
            p_min_anomaly_values = anomaly_weights / (
                anomaly_weights + total_calib_weight
            )
            sorted_p_min_anomaly = np.sort(p_min_anomaly_values, kind="mergesort")
            anomaly_ranks = np.arange(1, n_anomaly + 1)
            p_min_anomaly = float(sorted_p_min_anomaly[0])
            delta_min = p_min_anomaly / (ALPHA / m)
            rank_delta = float(
                np.min(sorted_p_min_anomaly / (ALPHA * anomaly_ranks / m))
            )

            exact_any, exact_power, exact_fdr = discovery_metrics(exact_p_values, y_true)
            random_any, random_power, random_fdr = discovery_metrics(random_p_values, y_true)
            oracle_any, oracle_power, oracle_fdr = discovery_metrics(oracle_p_values, y_true)

            rows.append(
                {
                    "summary_version": SUMMARY_VERSION,
                    "n_cal": n_cal,
                    "m": m,
                    "rho": rho,
                    "seed": seed,
                    "alpha": ALPHA,
                    "pi1": PI1,
                    "n_anomaly": n_anomaly,
                    "calib_ess": calib_ess,
                    "max_normalized_calib_weight": max_normalized_weight,
                    "p_min_anom": p_min_anomaly,
                    "delta_min": float(delta_min),
                    "rank_delta": rank_delta,
                    "certified_no_first_discovery": bool(delta_min > 1.0),
                    "certified_no_rank_discovery": bool(rank_delta > 1.0),
                    "log10_m_over_alpha": float(np.log10(m / ALPHA)),
                    "log10_inverse_p_min": float(np.log10(1.0 / p_min_anomaly)),
                    "log10_delta_min": float(np.log10(delta_min)),
                    "log10_rank_delta": float(np.log10(rank_delta)),
                    "auroc": empirical_auc(float(anomaly_scores[0]), inlier_scores),
                    "perfect_separation_from_calibration": bool(
                        np.all(anomaly_scores > max_calib_score)
                    ),
                    "anomalies_exceed_all_inliers": bool(
                        float(np.min(anomaly_scores)) > float(np.max(inlier_scores))
                    ),
                    "exact_any_discovery": exact_any,
                    "exact_power": exact_power,
                    "exact_fdr": exact_fdr,
                    "randomized_any_discovery": random_any,
                    "randomized_power": random_power,
                    "randomized_fdr": random_fdr,
                    "oracle_any_discovery": oracle_any,
                    "oracle_power": oracle_power,
                    "oracle_fdr": oracle_fdr,
                }
            )

    return rows


def summary_is_current() -> bool:
    if not SUMMARY_PATH.exists():
        return False
    header = pd.read_csv(SUMMARY_PATH, nrows=1)
    return not header.empty and str(header["summary_version"].iloc[0]) == SUMMARY_VERSION


def build_summary() -> pd.DataFrame:
    tasks = [(n_cal, float(rho)) for n_cal in N_VALUES for rho in RHO_VALUES]
    rows: list[dict[str, float | int | str]] = []
    total_tasks = len(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for done, block_rows in enumerate(executor.map(simulate_block, tasks), start=1):
            rows.extend(block_rows)
            if done % len(RHO_VALUES) == 0 or done == total_tasks:
                print(f"completed Figure 2 simulation block {done}/{total_tasks}", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def load_or_build_summary() -> pd.DataFrame:
    if summary_is_current():
        print("loading existing Figure 2 summary", flush=True)
        return pd.read_csv(SUMMARY_PATH)
    return build_summary()


def add_power_config_count(
    counts: dict[tuple[str, int, int], list[float]],
    key: tuple[str, int, int],
    *,
    neff: float,
    power: float,
    any_discovery: bool,
    certified_no_rank: bool,
    auroc: float,
) -> None:
    bucket = counts.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    bucket[0] += 1.0
    bucket[1] += neff
    bucket[2] += power
    bucket[3] += float(any_discovery)
    bucket[4] += float(certified_no_rank)
    bucket[5] += auroc


def simulate_power_config_block(task: tuple[int, float]) -> dict[tuple[str, int, int], list[float]]:
    n_cal, rho = task
    rho_code = int(round(rho * 1000))
    counts: dict[tuple[str, int, int], list[float]] = {}

    for seed in range(N_SEEDS):
        calib_rng = rng_for(n_cal, rho_code, seed, 100)
        calib_x = calib_rng.normal(0.0, 1.0, size=(n_cal, D))
        calib_scores = calib_rng.normal(0.0, 1.0, n_cal)
        calib_weights = np.exp(rho * calib_x[:, 0] - 0.5 * rho**2)
        total_calib_weight = float(np.sum(calib_weights))
        calib_ess = effective_sample_size(calib_weights)
        log_ess_bin = int(
            np.searchsorted(LOG_ESS_BINS, np.log10(calib_ess), side="right") - 1
        )
        if not 0 <= log_ess_bin < len(LOG_ESS_BINS) - 1:
            continue

        order = np.argsort(calib_scores, kind="mergesort")
        sorted_calib_scores = calib_scores[order]
        sorted_calib_weights = calib_weights[order]
        suffix_calib_weights = np.concatenate(
            ([0.0], np.cumsum(sorted_calib_weights[::-1]))
        )[::-1]
        max_calib_score = float(np.max(calib_scores))

        for m in M_VALUES:
            for pi_idx, pi1 in enumerate(POWER_PI_VALUES):
                n_anomaly = max(1, int(round(pi1 * m)))
                n_inlier = m - n_anomaly
                test_rng = rng_for(n_cal, m, rho_code, seed, 200 + pi_idx)
                inlier_x = test_rng.normal(0.0, 1.0, size=(n_inlier, D))
                anomaly_x = test_rng.normal(0.0, 1.0, size=(n_anomaly, D))
                inlier_x[:, 0] += rho
                anomaly_x[:, 0] += rho
                test_x = np.vstack([inlier_x, anomaly_x])
                test_weights = np.exp(rho * test_x[:, 0] - 0.5 * rho**2)

                inlier_scores = test_rng.normal(0.0, 1.0, n_inlier)
                anomaly_scores = np.full(n_anomaly, max_calib_score + DELTA_SCORE)
                test_scores = np.concatenate([inlier_scores, anomaly_scores])
                y_true = np.concatenate(
                    [np.zeros(n_inlier, dtype=bool), np.ones(n_anomaly, dtype=bool)]
                )
                exact_p_values = weighted_tail_p_values(
                    sorted_calib_scores,
                    suffix_calib_weights,
                    total_calib_weight,
                    test_scores,
                    test_weights,
                )

                anomaly_weights = test_weights[n_inlier:]
                p_min_anomaly_values = anomaly_weights / (
                    anomaly_weights + total_calib_weight
                )
                sorted_p_min_anomaly = np.sort(p_min_anomaly_values, kind="mergesort")
                anomaly_ranks = np.arange(1, n_anomaly + 1)
                auroc = empirical_auc(float(anomaly_scores[0]), inlier_scores)

                for config in POWER_CONFIGS:
                    if float(config["pi1"]) != pi1:
                        continue
                    alpha = float(config["alpha"])
                    rank_delta = float(
                        np.min(sorted_p_min_anomaly / (alpha * anomaly_ranks / m))
                    )
                    any_discovery, power, _ = discovery_metrics(
                        exact_p_values,
                        y_true,
                        alpha,
                    )
                    key = (str(config["name"]), m, log_ess_bin)
                    add_power_config_count(
                        counts,
                        key,
                        neff=calib_ess,
                        power=power,
                        any_discovery=any_discovery,
                        certified_no_rank=rank_delta > 1.0,
                        auroc=auroc,
                    )

    return counts


def merge_power_config_counts(
    target: dict[tuple[str, int, int], list[float]],
    update: dict[tuple[str, int, int], list[float]],
) -> None:
    for key, values in update.items():
        bucket = target.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for idx, value in enumerate(values):
            bucket[idx] += value


def power_config_summary_is_current() -> bool:
    if not POWER_CONFIG_SUMMARY_PATH.exists():
        return False
    header = pd.read_csv(POWER_CONFIG_SUMMARY_PATH, nrows=1)
    return (
        not header.empty
        and str(header["summary_version"].iloc[0]) == POWER_CONFIG_SUMMARY_VERSION
    )


def build_power_config_summary() -> pd.DataFrame:
    tasks = [(n_cal, float(rho)) for n_cal in N_VALUES for rho in RHO_VALUES]
    counts: dict[tuple[str, int, int], list[float]] = {}
    total_tasks = len(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for done, update in enumerate(
            executor.map(simulate_power_config_block, tasks),
            start=1,
        ):
            merge_power_config_counts(counts, update)
            if done % len(RHO_VALUES) == 0 or done == total_tasks:
                print(
                    f"completed Figure 2 power configuration block {done}/{total_tasks}",
                    flush=True,
                )

    rows = []
    config_by_name = {str(config["name"]): config for config in POWER_CONFIGS}
    for (scenario, m, bin_idx), values in counts.items():
        count, sum_neff, sum_power, sum_any, sum_certified, sum_auroc = values
        config = config_by_name[scenario]
        rows.append(
            {
                "summary_version": POWER_CONFIG_SUMMARY_VERSION,
                "scenario": scenario,
                "alpha": float(config["alpha"]),
                "pi1": float(config["pi1"]),
                "label": str(config["label"]),
                "m": m,
                "log10_neff_left": LOG_ESS_BINS[bin_idx],
                "log10_neff_right": LOG_ESS_BINS[bin_idx + 1],
                "mean_neff": sum_neff / count,
                "power": sum_power / count,
                "discovery_probability": sum_any / count,
                "certified_no_rank_rate": sum_certified / count,
                "mean_auroc": sum_auroc / count,
                "count": int(count),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(POWER_CONFIG_SUMMARY_PATH, index=False)
    return summary


def load_or_build_power_config_summary() -> pd.DataFrame:
    if power_config_summary_is_current():
        print("loading existing Figure 2 power configuration summary", flush=True)
        return pd.read_csv(POWER_CONFIG_SUMMARY_PATH)
    return build_power_config_summary()


def plot_score_schematic(ax: plt.Axes) -> None:
    rng = rng_for(0, 0, 0, 9)
    n_cal = 80
    n_inlier = 70
    n_anomaly = 12
    calib_scores = rng.normal(0.0, 1.0, n_cal)
    inlier_scores = rng.normal(0.0, 1.0, n_inlier)
    max_calib = float(np.max(calib_scores))
    anomaly_scores = max_calib + DELTA_SCORE + rng.uniform(0.0, 0.12, n_anomaly)

    ax.scatter(
        calib_scores,
        rng.normal(0.20, 0.018, n_cal),
        s=16,
        color=COLORS["calibration"],
        alpha=0.45,
        label="calibration scores",
    )
    ax.scatter(
        inlier_scores,
        rng.normal(0.34, 0.018, n_inlier),
        s=16,
        color=COLORS["inlier"],
        alpha=0.55,
        label="test inliers",
    )
    ax.scatter(
        anomaly_scores,
        rng.normal(0.50, 0.018, n_anomaly),
        s=28,
        color=COLORS["anomaly"],
        alpha=0.9,
        label="test anomalies",
    )
    ax.axvline(max_calib, color="black", linestyle="--", linewidth=1.1)
    ax.text(
        max_calib + 0.06,
        0.61,
        r"$\max_i S_i^{cal}$",
        ha="left",
        va="center",
        fontsize=9,
    )
    ax.set_ylim(0.1, 0.68)
    ax.set_yticks([0.20, 0.34, 0.50])
    ax.set_yticklabels(["calibration", "inliers", "anomalies"])
    ax.set_xlabel("anomaly score")
    ax.set_title("A. Perfect score separation")
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def plot_pvalue_floor(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x_values = np.array([np.log10(m / ALPHA) for m in M_VALUES], dtype=float)
    x_midpoints = (x_values[:-1] + x_values[1:]) / 2.0
    x_edges = np.concatenate(
        [
            [x_values[0] - (x_midpoints[0] - x_values[0])],
            x_midpoints,
            [x_values[-1] + (x_values[-1] - x_midpoints[-1])],
        ]
    )
    matrix = np.full((len(x_edges) - 1, len(Y_BINS) - 1), np.nan)
    for x_idx, m in enumerate(M_VALUES):
        block = summary[summary["m"].eq(m)]
        y_idx = np.searchsorted(Y_BINS, block["log10_inverse_p_min"], side="right") - 1
        for bin_idx in range(len(Y_BINS) - 1):
            in_bin = y_idx == bin_idx
            if np.any(in_bin):
                matrix[x_idx, bin_idx] = float(block.loc[in_bin, "exact_any_discovery"].mean())

    mesh = ax.pcolormesh(
        x_edges,
        Y_BINS,
        np.ma.masked_invalid(matrix).T,
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=1.0),
        shading="flat",
    )
    diagonal_x = np.linspace(float(x_edges[0]), float(x_edges[-1]), 200)
    ax.plot(diagonal_x, diagonal_x, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(Y_BINS[0]), float(Y_BINS[-1]))
    ax.set_xlabel(r"$\log_{10}(m/\alpha)$")
    ax.set_ylabel(r"$\log_{10}(1/p^{min}_{anom})$")
    ax.set_title("B. P-value floor versus BH threshold")
    ax.grid(alpha=0.18, linewidth=0.6)
    cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pr(exact WEDF discovers)")


def plot_power_vs_ess(ax: plt.Axes, summary: pd.DataFrame) -> None:
    for m in PANEL_C_M_VALUES:
        block = summary[summary["m"].eq(m)].copy()
        block["log10_ess"] = np.log10(block["calib_ess"])
        bin_idx = np.searchsorted(LOG_ESS_BINS, block["log10_ess"], side="right") - 1
        xs = []
        ys = []
        blocked = []
        for idx in range(len(LOG_ESS_BINS) - 1):
            in_bin = bin_idx == idx
            if np.sum(in_bin) < 20:
                continue
            bin_block = block.loc[in_bin]
            xs.append(float(np.median(bin_block["calib_ess"])))
            ys.append(float(np.mean(bin_block["exact_power"])))
            blocked.append(float(np.median(bin_block["rank_delta"])) > 1.0)
        if not xs:
            continue
        line = ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=f"m={m}")[0]
        blocked_x = np.asarray(xs)[np.asarray(blocked, dtype=bool)]
        if len(blocked_x):
            ax.scatter(
                blocked_x,
                np.asarray(ys)[np.asarray(blocked, dtype=bool)],
                s=42,
                facecolors="none",
                edgecolors=line.get_color(),
                linewidths=1.2,
            )

    median_auc = float(summary["auroc"].median())
    ax.text(
        0.03,
        0.08,
        f"median AUROC = {median_auc:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"median calibration $N_{eff}$")
    ax.set_ylabel("exact WEDF power")
    ax.set_title("C. Power collapse despite perfect scores")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")


def method_curve(summary: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    bin_idx = np.searchsorted(DELTA_BINS, summary["log10_delta_min"], side="right") - 1
    for idx in range(len(DELTA_BINS) - 1):
        in_bin = bin_idx == idx
        if np.sum(in_bin) < 20:
            continue
        block = summary.loc[in_bin]
        rows.append(
            {
                "x": float(block["log10_delta_min"].mean()),
                "probability": float(block[column].mean()),
                "count": int(len(block)),
            }
        )
    return pd.DataFrame(rows)


def plot_detectability_ratio(ax: plt.Axes, summary: pd.DataFrame) -> None:
    specs = [
        ("exact_wedf", "exact_any_discovery"),
        ("randomized_wedf", "randomized_any_discovery"),
        ("oracle_continuous_tail", "oracle_any_discovery"),
    ]
    for method, column in specs:
        curve = method_curve(summary, column)
        ax.plot(
            curve["x"],
            curve["probability"],
            marker="o",
            markersize=3,
            linewidth=1.7,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(float(DELTA_BINS[0]), float(DELTA_BINS[-1]))
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"$\log_{10}\delta_{min}$,  $\delta_{min}=p^{min}_{anom}/(\alpha/m)$")
    ax.set_ylabel("Pr(BH finds >=1 anomaly)")
    ax.set_title("D. Detectability ratio calibration")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")


def write_key_table(summary: pd.DataFrame) -> None:
    summary = summary.copy()
    summary["setting"] = np.where(
        summary["rank_delta"] <= 1.0,
        "feasible by rank-aware diagnostic",
        "certified no rank discovery",
    )
    rows = []
    for setting in ["feasible by rank-aware diagnostic", "certified no rank discovery"]:
        block = summary[summary["setting"].eq(setting)]
        if block.empty:
            continue
        rows.append(
            {
                "summary_version": SUMMARY_VERSION,
                "setting": setting,
                "trials": len(block),
                "median_auroc": block["auroc"].median(),
                "median_neff": block["calib_ess"].median(),
                "median_max_normalized_weight": block["max_normalized_calib_weight"].median(),
                "median_p_min_anom": block["p_min_anom"].median(),
                "median_alpha_over_m": (ALPHA / block["m"]).median(),
                "median_delta_min": block["delta_min"].median(),
                "certified_no_first_discovery_rate": block[
                    "certified_no_first_discovery"
                ].mean(),
                "certified_no_rank_discovery_rate": block[
                    "certified_no_rank_discovery"
                ].mean(),
                "exact_discovery_rate": block["exact_any_discovery"].mean(),
                "exact_power": block["exact_power"].mean(),
                "exact_fdr": block["exact_fdr"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_PATH, index=False)


def validate_summary(summary: pd.DataFrame) -> None:
    observed_m = set(summary["m"].unique())
    if observed_m != set(M_VALUES):
        raise RuntimeError(f"Unexpected m values: {observed_m}")
    if not bool(summary["perfect_separation_from_calibration"].all()):
        raise RuntimeError("Anomaly scores must exceed every calibration score.")
    if not np.isfinite(summary["p_min_anom"]).all():
        raise RuntimeError("All exact WEDF anomaly p-value floors must be finite.")
    if not ((summary["p_min_anom"] >= 0.0) & (summary["p_min_anom"] <= 1.0)).all():
        raise RuntimeError("Exact WEDF anomaly p-value floors must lie in [0, 1].")
    if not np.allclose(
        summary["certified_no_first_discovery"],
        summary["delta_min"] > 1.0,
    ):
        raise RuntimeError("First-discovery certificate does not match delta_min.")


def plot_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.6), constrained_layout=True)
    plot_score_schematic(axes[0, 0])
    plot_pvalue_floor(axes[0, 1], summary)
    plot_power_vs_ess(axes[1, 0], summary)
    plot_detectability_ratio(axes[1, 1], summary)
    fig.suptitle(
        "Perfect-score stress test: finite-sample resolution prevents discovery",
        fontsize=14,
    )
    fig.savefig(FIGURE_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_power_config_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    scenario: str,
    title: str,
) -> None:
    block = summary[summary["scenario"].eq(scenario)]
    for m in PANEL_C_M_VALUES:
        line = block[block["m"].eq(m)].sort_values("mean_neff")
        if line.empty:
            continue
        plotted = ax.plot(
            line["mean_neff"],
            line["power"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"m={m}",
        )[0]
        certified = line["certified_no_rank_rate"] >= 0.5
        if bool(certified.any()):
            ax.scatter(
                line.loc[certified, "mean_neff"],
                line.loc[certified, "power"],
                s=42,
                facecolors="none",
                edgecolors=plotted.get_color(),
                linewidths=1.2,
            )

    median_auc = float(block["mean_auroc"].median())
    ax.text(
        0.03,
        0.08,
        f"mean AUROC = {median_auc:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.grid(alpha=0.18, linewidth=0.6)


def plot_power_configurations(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.8, 9.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for ax, config in zip(axes.ravel(), POWER_CONFIGS, strict=True):
        plot_power_config_panel(
            ax,
            summary,
            str(config["name"]),
            str(config["label"]),
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("exact WEDF power")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"mean calibration $N_{eff}$")

    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Perfect-score power collapse across alpha and anomaly-rate settings",
        fontsize=14,
    )
    fig.savefig(POWER_CONFIG_FIGURE_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def export_panel_a_tikz() -> None:
    rng = rng_for(0, 0, 0, 9)
    n_cal = 80
    n_inlier = 70
    n_anomaly = 12
    calib_scores = rng.normal(0.0, 1.0, n_cal)
    inlier_scores = rng.normal(0.0, 1.0, n_inlier)
    max_calib = float(np.max(calib_scores))
    anomaly_scores = max_calib + DELTA_SCORE + rng.uniform(0.0, 0.12, n_anomaly)

    score_rows = []
    for group, label, style_key, scores, y_values, plot_order in [
        (
            "calibration",
            "calibration scores",
            "calibration",
            calib_scores,
            rng.normal(0.20, 0.018, n_cal),
            1,
        ),
        (
            "inlier",
            "test inliers",
            "inlier",
            inlier_scores,
            rng.normal(0.34, 0.018, n_inlier),
            2,
        ),
        (
            "anomaly",
            "test anomalies",
            "anomaly",
            anomaly_scores,
            rng.normal(0.50, 0.018, n_anomaly),
            3,
        ),
    ]:
        for idx, (x, y) in enumerate(zip(scores, y_values, strict=True)):
            score_rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure2",
                    "panel": "A",
                    "panel_title": "Perfect score separation",
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
    pd.DataFrame(score_rows).to_csv(PANEL_A_SCORES_TIKZ_PATH, index=False)

    annotation_rows = [
        {
            "export_version": TIKZ_EXPORT_VERSION,
            "figure": "figure2",
            "panel": "A",
            "panel_title": "Perfect score separation",
            "plot_order": 1,
            "object_type": "vertical_line",
            "x": max_calib,
            "y": 0.1,
            "x_end": max_calib,
            "y_end": 0.68,
            "style_key": "max_calibration_score",
            "label": r"$\max_i S_i^{cal}$",
        },
        {
            "export_version": TIKZ_EXPORT_VERSION,
            "figure": "figure2",
            "panel": "A",
            "panel_title": "Perfect score separation",
            "plot_order": 2,
            "object_type": "text",
            "x": max_calib + 0.06,
            "y": 0.61,
            "x_end": "",
            "y_end": "",
            "style_key": "text",
            "label": r"$\max_i S_i^{cal}$",
        },
    ]
    pd.DataFrame(annotation_rows).to_csv(PANEL_A_ANNOTATIONS_TIKZ_PATH, index=False)


def export_panel_b_tikz(summary: pd.DataFrame) -> None:
    x_values = np.array([np.log10(m / ALPHA) for m in M_VALUES], dtype=float)
    x_midpoints = (x_values[:-1] + x_values[1:]) / 2.0
    x_edges = np.concatenate(
        [
            [x_values[0] - (x_midpoints[0] - x_values[0])],
            x_midpoints,
            [x_values[-1] + (x_values[-1] - x_midpoints[-1])],
        ]
    )
    rows = []
    for x_idx, m in enumerate(M_VALUES):
        block = summary[summary["m"].eq(m)]
        y_idx = np.searchsorted(Y_BINS, block["log10_inverse_p_min"], side="right") - 1
        for bin_idx in range(len(Y_BINS) - 1):
            in_bin = y_idx == bin_idx
            count = int(np.sum(in_bin))
            if count == 0:
                continue
            probability = float(block.loc[in_bin, "exact_any_discovery"].mean())
            x_left = float(x_edges[x_idx])
            x_right = float(x_edges[x_idx + 1])
            y_bottom = float(Y_BINS[bin_idx])
            y_top = float(Y_BINS[bin_idx + 1])
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure2",
                    "panel": "B",
                    "panel_title": "P-value floor versus BH threshold",
                    "plot_order": x_idx + 1,
                    "group": "heatmap_cell",
                    "method": "exact_wedf",
                    "scenario": "baseline",
                    "kappa": "",
                    "m": m,
                    "x": (x_left + x_right) / 2.0,
                    "y": (y_bottom + y_top) / 2.0,
                    "x_left": x_left,
                    "x_right": x_right,
                    "y_bottom": y_bottom,
                    "y_top": y_top,
                    "value": probability,
                    "probability": probability,
                    "count": count,
                    "style_key": "discovery_probability",
                    "label": "Pr(exact WEDF discovers)",
                }
            )
    pd.DataFrame(rows).to_csv(PANEL_B_HEATMAP_TIKZ_PATH, index=False)

    boundary_rows = []
    for point_order, x in enumerate([float(x_edges[0]), float(x_edges[-1])], start=1):
        boundary_rows.append(
            {
                "export_version": TIKZ_EXPORT_VERSION,
                "figure": "figure2",
                "panel": "B",
                "panel_title": "P-value floor versus BH threshold",
                "plot_order": point_order,
                "group": "bh_boundary",
                "method": "",
                "scenario": "baseline",
                "kappa": "",
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
    pd.DataFrame(boundary_rows).to_csv(PANEL_B_BOUNDARY_TIKZ_PATH, index=False)


def panel_c_curve_rows(summary: pd.DataFrame) -> list[dict[str, float | int | str | bool]]:
    rows = []
    for plot_order, m in enumerate(PANEL_C_M_VALUES, start=1):
        block = summary[summary["m"].eq(m)].copy()
        block["log10_ess"] = np.log10(block["calib_ess"])
        bin_idx = np.searchsorted(LOG_ESS_BINS, block["log10_ess"], side="right") - 1
        for idx in range(len(LOG_ESS_BINS) - 1):
            in_bin = bin_idx == idx
            count = int(np.sum(in_bin))
            if count < 20:
                continue
            bin_block = block.loc[in_bin]
            certified = float(np.median(bin_block["rank_delta"])) > 1.0
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure2",
                    "panel": "C",
                    "panel_title": "Power collapse despite perfect scores",
                    "plot_order": plot_order,
                    "group": "power_curve",
                    "method": "exact_wedf",
                    "scenario": "baseline",
                    "kappa": "",
                    "m": m,
                    "x": float(np.median(bin_block["calib_ess"])),
                    "y": float(np.mean(bin_block["exact_power"])),
                    "value": float(np.mean(bin_block["exact_power"])),
                    "probability": "",
                    "count": count,
                    "certified_no_rank": certified,
                    "style_key": f"m_{m}",
                    "label": f"m={m}",
                }
            )
    return rows


def export_panel_c_tikz(summary: pd.DataFrame) -> None:
    pd.DataFrame(panel_c_curve_rows(summary)).to_csv(PANEL_C_POWER_TIKZ_PATH, index=False)


def export_panel_d_tikz(summary: pd.DataFrame) -> None:
    specs = [
        ("exact_wedf", "exact_any_discovery"),
        ("randomized_wedf", "randomized_any_discovery"),
        ("oracle_continuous_tail", "oracle_any_discovery"),
    ]
    rows = []
    for plot_order, (method, column) in enumerate(specs, start=1):
        curve = method_curve(summary, column)
        for idx, row in enumerate(curve.itertuples(index=False)):
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure2",
                    "panel": "D",
                    "panel_title": "Detectability ratio calibration",
                    "plot_order": plot_order,
                    "point_index": idx,
                    "group": "detectability_curve",
                    "method": method,
                    "scenario": "baseline",
                    "kappa": "",
                    "m": "",
                    "x": float(row.x),
                    "y": float(row.probability),
                    "value": float(row.probability),
                    "probability": float(row.probability),
                    "count": int(row.count),
                    "style_key": method,
                    "label": METHOD_LABELS[method],
                }
            )
    pd.DataFrame(rows).to_csv(PANEL_D_DETECTABILITY_TIKZ_PATH, index=False)

    reference_rows = [
        {
            "export_version": TIKZ_EXPORT_VERSION,
            "figure": "figure2",
            "panel": "D",
            "panel_title": "Detectability ratio calibration",
            "plot_order": idx,
            "point_index": idx,
            "group": "reference",
            "method": "",
            "scenario": "baseline",
            "kappa": "",
            "m": "",
            "x": 0.0,
            "y": y,
            "value": "",
            "probability": "",
            "count": "",
            "style_key": "vertical_zero",
            "label": r"$\delta_{min}=1$",
        }
        for idx, y in enumerate([-0.03, 1.03], start=1)
    ]
    pd.DataFrame(reference_rows).to_csv(PANEL_D_REFERENCE_TIKZ_PATH, index=False)


def export_power_configurations_tikz(summary: pd.DataFrame) -> None:
    config_order = {str(config["name"]): idx + 1 for idx, config in enumerate(POWER_CONFIGS)}
    rows = summary.copy()
    rows["export_version"] = TIKZ_EXPORT_VERSION
    rows["figure"] = "figure2"
    rows["panel"] = rows["scenario"]
    rows["panel_title"] = rows["label"]
    rows["plot_order"] = rows["scenario"].map(config_order)
    rows["group"] = "power_configuration_curve"
    rows["method"] = "exact_wedf"
    rows["kappa"] = ""
    rows["x"] = rows["mean_neff"]
    rows["y"] = rows["power"]
    rows["value"] = rows["power"]
    rows["probability"] = rows["discovery_probability"]
    rows["style_key"] = "m_" + rows["m"].astype(str)
    rows = rows.rename(columns={"certified_no_rank_rate": "certified_no_rank"})
    rows[
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
            "log10_neff_left",
            "log10_neff_right",
            "value",
            "probability",
            "count",
            "certified_no_rank",
            "style_key",
            "label",
        ]
    ].to_csv(POWER_CONFIGURATIONS_TIKZ_PATH, index=False)


def export_tikz_csvs(summary: pd.DataFrame, power_config_summary: pd.DataFrame) -> None:
    export_panel_a_tikz()
    export_panel_b_tikz(summary)
    export_panel_c_tikz(summary)
    export_panel_d_tikz(summary)
    export_power_configurations_tikz(power_config_summary)


def validate_outputs() -> None:
    for path in [
        FIGURE_PATH,
        SUMMARY_PATH,
        TABLE_PATH,
        POWER_CONFIG_FIGURE_PATH,
        POWER_CONFIG_SUMMARY_PATH,
        PANEL_A_SCORES_TIKZ_PATH,
        PANEL_A_ANNOTATIONS_TIKZ_PATH,
        PANEL_B_HEATMAP_TIKZ_PATH,
        PANEL_B_BOUNDARY_TIKZ_PATH,
        PANEL_C_POWER_TIKZ_PATH,
        PANEL_D_DETECTABILITY_TIKZ_PATH,
        PANEL_D_REFERENCE_TIKZ_PATH,
        POWER_CONFIGURATIONS_TIKZ_PATH,
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
    summary = load_or_build_summary()
    validate_summary(summary)
    power_config_summary = load_or_build_power_config_summary()
    write_key_table(summary)
    plot_figure(summary)
    plot_power_configurations(power_config_summary)
    export_tikz_csvs(summary, power_config_summary)
    validate_outputs()
    print(SUMMARY_PATH)
    print(TABLE_PATH)
    print(FIGURE_PATH)
    print(POWER_CONFIG_SUMMARY_PATH)
    print(POWER_CONFIG_FIGURE_PATH)
    print(PANEL_A_SCORES_TIKZ_PATH)
    print(PANEL_A_ANNOTATIONS_TIKZ_PATH)
    print(PANEL_B_HEATMAP_TIKZ_PATH)
    print(PANEL_B_BOUNDARY_TIKZ_PATH)
    print(PANEL_C_POWER_TIKZ_PATH)
    print(PANEL_D_DETECTABILITY_TIKZ_PATH)
    print(PANEL_D_REFERENCE_TIKZ_PATH)
    print(POWER_CONFIGURATIONS_TIKZ_PATH)


if __name__ == "__main__":
    main()
