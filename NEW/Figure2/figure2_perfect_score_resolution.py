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
PANEL_C_DETECTABILITY_TIKZ_PATH = OUT_DIR / "figure2_panel_c_detectability_tikz.csv"
PANEL_C_REFERENCE_TIKZ_PATH = OUT_DIR / "figure2_panel_c_reference_tikz.csv"
POWER_CONFIGURATIONS_TIKZ_PATH = OUT_DIR / "figure2_power_configurations_tikz.csv"

SUMMARY_VERSION = "perfect-score-v2"
POWER_CONFIG_SUMMARY_VERSION = "power-config-v2"
TIKZ_EXPORT_VERSION = "tikz-v2"
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
PHASE_VIEW_LIMITS = (2.25, 4.75)
PHASE_X_BINS = 12
PHASE_Y_BINS = 12
PHASE_CELL_TRIALS = 100
PHASE_MAX_ACCEPT_ATTEMPTS_PER_CELL = 5000
PHASE_RHO_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
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


def phase_x_edges() -> np.ndarray:
    return np.linspace(*PHASE_VIEW_LIMITS, int(PHASE_X_BINS) + 1)


def phase_y_edges() -> np.ndarray:
    return np.linspace(*PHASE_VIEW_LIMITS, int(PHASE_Y_BINS) + 1)


def cell_center(edges: np.ndarray, bin_idx: int) -> float:
    return float((edges[bin_idx] + edges[bin_idx + 1]) / 2.0)


def value_in_bin(value: float, edges: np.ndarray, bin_idx: int) -> bool:
    left = float(edges[bin_idx])
    right = float(edges[bin_idx + 1])
    if bin_idx == len(edges) - 2:
        return left <= value <= right
    return left <= value < right


def m_for_phase_x(alpha: float, x_center: float) -> int:
    return max(2, int(round(alpha * 10**x_center)))


def phase_n_candidates_for_y(y_center: float) -> list[int]:
    base = max(2, int(round(10**y_center - 1.0)))
    multipliers = [0.05, 0.1, 0.25, 0.5, 1.0]
    return sorted({max(2, int(round(base * multiplier))) for multiplier in multipliers})


def simulate_perfect_score_trial(
    n_cal: int,
    m: int,
    rho: float,
    seed: int,
    alpha: float = ALPHA,
    pi1: float = PI1,
) -> dict[str, float | int | str | bool]:
    rho_code = int(round(rho * 1000))
    calib_rng = rng_for(n_cal, m, rho_code, seed, 0)
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

    n_anomaly = max(1, int(round(pi1 * m)))
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
    delta_min = p_min_anomaly / (alpha / m)
    rank_delta = float(
        np.min(sorted_p_min_anomaly / (alpha * anomaly_ranks / m))
    )

    exact_any, exact_power, exact_fdr = discovery_metrics(exact_p_values, y_true, alpha)
    random_any, random_power, random_fdr = discovery_metrics(
        random_p_values,
        y_true,
        alpha,
    )
    oracle_any, oracle_power, oracle_fdr = discovery_metrics(
        oracle_p_values,
        y_true,
        alpha,
    )

    return {
        "n_cal": n_cal,
        "m": m,
        "rho": rho,
        "seed": seed,
        "alpha": alpha,
        "pi1": pi1,
        "n_anomaly": n_anomaly,
        "calib_ess": calib_ess,
        "max_normalized_calib_weight": max_normalized_weight,
        "p_min_anom": p_min_anomaly,
        "delta_min": float(delta_min),
        "rank_delta": rank_delta,
        "certified_no_first_discovery": bool(delta_min > 1.0),
        "certified_no_rank_discovery": bool(rank_delta > 1.0),
        "log10_m_over_alpha": float(np.log10(m / alpha)),
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


def summary_is_current() -> bool:
    if not SUMMARY_PATH.exists():
        return False
    header = pd.read_csv(SUMMARY_PATH, nrows=1)
    return not header.empty and str(header["summary_version"].iloc[0]) == SUMMARY_VERSION


def aggregate_phase_trials(
    trial_rows: list[dict[str, float | int | str | bool]],
    *,
    x_bin: int,
    y_bin: int,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
) -> dict[str, float | int | str | bool]:
    block = pd.DataFrame(trial_rows)
    count = int(len(block))
    return {
        "summary_version": SUMMARY_VERSION,
        "phase_x_bin": x_bin,
        "phase_y_bin": y_bin,
        "x_left": x_left,
        "x_right": x_right,
        "y_bottom": y_bottom,
        "y_top": y_top,
        "x": (x_left + x_right) / 2.0,
        "y": (y_bottom + y_top) / 2.0,
        "alpha": ALPHA,
        "pi1": PI1,
        "m": int(round(float(block["m"].median()))),
        "mean_n_cal": float(block["n_cal"].mean()),
        "mean_rho": float(block["rho"].mean()),
        "count": count,
        "n_anomaly": float(block["n_anomaly"].mean()),
        "calib_ess": float(block["calib_ess"].mean()),
        "max_normalized_calib_weight": float(
            block["max_normalized_calib_weight"].mean()
        ),
        "p_min_anom": float(block["p_min_anom"].mean()),
        "delta_min": float(block["delta_min"].mean()),
        "rank_delta": float(block["rank_delta"].mean()),
        "certified_no_first_discovery": float(
            block["certified_no_first_discovery"].mean()
        ),
        "certified_no_rank_discovery": float(
            block["certified_no_rank_discovery"].mean()
        ),
        "log10_m_over_alpha": float(block["log10_m_over_alpha"].mean()),
        "log10_inverse_p_min": float(block["log10_inverse_p_min"].mean()),
        "log10_delta_min": float(block["log10_delta_min"].mean()),
        "log10_rank_delta": float(block["log10_rank_delta"].mean()),
        "auroc": float(block["auroc"].mean()),
        "perfect_separation_from_calibration": bool(
            block["perfect_separation_from_calibration"].all()
        ),
        "anomalies_exceed_all_inliers": bool(block["anomalies_exceed_all_inliers"].all()),
        "exact_any_discovery": float(block["exact_any_discovery"].mean()),
        "exact_power": float(block["exact_power"].mean()),
        "exact_fdr": float(block["exact_fdr"].mean()),
        "randomized_any_discovery": float(block["randomized_any_discovery"].mean()),
        "randomized_power": float(block["randomized_power"].mean()),
        "randomized_fdr": float(block["randomized_fdr"].mean()),
        "oracle_any_discovery": float(block["oracle_any_discovery"].mean()),
        "oracle_power": float(block["oracle_power"].mean()),
        "oracle_fdr": float(block["oracle_fdr"].mean()),
    }


def simulate_phase_cell(task: tuple[int, int]) -> dict[str, float | int | str | bool]:
    x_bin, y_bin = task
    x_edges = phase_x_edges()
    y_edges = phase_y_edges()
    x_center = cell_center(x_edges, x_bin)
    y_center = cell_center(y_edges, y_bin)
    m = m_for_phase_x(ALPHA, x_center)
    configs = [
        (n_cal, float(rho))
        for n_cal in phase_n_candidates_for_y(y_center)
        for rho in PHASE_RHO_CANDIDATES
    ]
    accepted_rows: list[dict[str, float | int | str | bool]] = []
    attempts = 0

    while (
        len(accepted_rows) < int(PHASE_CELL_TRIALS)
        and attempts < int(PHASE_MAX_ACCEPT_ATTEMPTS_PER_CELL)
    ):
        n_cal, rho = configs[attempts % len(configs)]
        row = simulate_perfect_score_trial(
            n_cal,
            m,
            rho,
            x_bin * 100000 + y_bin * 1000 + attempts,
            ALPHA,
            PI1,
        )
        attempts += 1
        if not value_in_bin(float(row["log10_inverse_p_min"]), y_edges, y_bin):
            continue
        accepted_rows.append(row)

    if len(accepted_rows) < int(PHASE_CELL_TRIALS):
        raise RuntimeError(
            "Could not fill Figure 2 phase cell: "
            f"x_bin={x_bin}, y_bin={y_bin}, accepted={len(accepted_rows)}, "
            f"attempts={attempts}."
        )

    return aggregate_phase_trials(
        accepted_rows,
        x_bin=x_bin,
        y_bin=y_bin,
        x_left=float(x_edges[x_bin]),
        x_right=float(x_edges[x_bin + 1]),
        y_bottom=float(y_edges[y_bin]),
        y_top=float(y_edges[y_bin + 1]),
    )


def build_summary() -> pd.DataFrame:
    tasks = [
        (x_bin, y_bin)
        for x_bin in range(int(PHASE_X_BINS))
        for y_bin in range(int(PHASE_Y_BINS))
    ]
    rows: list[dict[str, float | int | str | bool]] = []
    total_tasks = len(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for done, row in enumerate(executor.map(simulate_phase_cell, tasks), start=1):
            rows.append(row)
            if done % int(PHASE_Y_BINS) == 0 or done == total_tasks:
                print(f"completed Figure 2 phase cell {done}/{total_tasks}", flush=True)
    summary = pd.DataFrame(rows)
    validate_phase_summary(summary)
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
        mean_neff = sum_neff / count
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
                "mean_neff": mean_neff,
                "log10_mean_neff": float(np.log10(mean_neff)),
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
    x_edges = phase_x_edges()
    y_edges = phase_y_edges()
    matrix = np.full((int(PHASE_X_BINS), int(PHASE_Y_BINS)), np.nan)
    for row in summary.itertuples(index=False):
        matrix[int(row.phase_x_bin), int(row.phase_y_bin)] = float(
            row.exact_any_discovery
        )

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(matrix).T,
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=1.0),
        shading="flat",
    )
    diagonal_x = np.asarray(PHASE_VIEW_LIMITS, dtype=float)
    ax.plot(diagonal_x, diagonal_x, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlim(*PHASE_VIEW_LIMITS)
    ax.set_ylim(*PHASE_VIEW_LIMITS)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\log_{10}(m/\alpha)$")
    ax.set_ylabel(r"$\log_{10}(1/p^{min}_{anom})$")
    ax.set_title("B. P-value floor versus BH threshold")
    ax.grid(alpha=0.18, linewidth=0.6)
    cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pr(exact WEDF discovers)")


def method_curve(summary: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    bin_idx = np.searchsorted(DELTA_BINS, summary["log10_delta_min"], side="right") - 1
    for idx in range(len(DELTA_BINS) - 1):
        in_bin = bin_idx == idx
        count = int(summary.loc[in_bin, "count"].sum())
        if count <= 0:
            continue
        block = summary.loc[in_bin]
        weights = block["count"].to_numpy(dtype=float)
        rows.append(
            {
                "x": float(np.average(block["log10_delta_min"], weights=weights)),
                "probability": float(np.average(block[column], weights=weights)),
                "count": count,
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
    ax.set_title("C. Detectability ratio calibration")
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
                "phase_cells": len(block),
                "accepted_trials": int(block["count"].sum()),
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


def validate_phase_summary(summary: pd.DataFrame) -> None:
    x_edges = phase_x_edges()
    y_edges = phase_y_edges()
    expected = {
        (x_bin, y_bin)
        for x_bin in range(int(PHASE_X_BINS))
        for y_bin in range(int(PHASE_Y_BINS))
    }
    observed = {
        (int(row.phase_x_bin), int(row.phase_y_bin))
        for row in summary.itertuples(index=False)
    }
    if observed != expected:
        raise RuntimeError(
            f"Unexpected phase cells: missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}."
        )
    if summary.duplicated(["phase_x_bin", "phase_y_bin"]).any():
        raise RuntimeError("Phase summary contains duplicate cells.")
    if (summary["count"] <= 0).any():
        raise RuntimeError("Phase summary contains zero-count cells.")
    for row in summary.itertuples(index=False):
        x_bin = int(row.phase_x_bin)
        y_bin = int(row.phase_y_bin)
        expected_edges = (
            x_edges[x_bin],
            x_edges[x_bin + 1],
            y_edges[y_bin],
            y_edges[y_bin + 1],
        )
        observed_edges = (row.x_left, row.x_right, row.y_bottom, row.y_top)
        if not np.allclose(observed_edges, expected_edges):
            raise RuntimeError(f"Phase cell edge mismatch: x={x_bin}, y={y_bin}.")


def validate_summary(summary: pd.DataFrame) -> None:
    validate_phase_summary(summary)
    if not bool(summary["perfect_separation_from_calibration"].all()):
        raise RuntimeError("Anomaly scores must exceed every calibration score.")
    if not np.isfinite(summary["p_min_anom"]).all():
        raise RuntimeError("All exact WEDF anomaly p-value floors must be finite.")
    if not ((summary["p_min_anom"] >= 0.0) & (summary["p_min_anom"] <= 1.0)).all():
        raise RuntimeError("Exact WEDF anomaly p-value floors must lie in [0, 1].")
    probability_columns = [
        "certified_no_first_discovery",
        "certified_no_rank_discovery",
        "exact_any_discovery",
        "randomized_any_discovery",
        "oracle_any_discovery",
    ]
    for column in probability_columns:
        if not ((summary[column] >= 0.0) & (summary[column] <= 1.0)).all():
            raise RuntimeError(f"Phase probability column out of range: {column}.")


def plot_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.4), constrained_layout=True)
    plot_score_schematic(axes[0])
    plot_pvalue_floor(axes[1], summary)
    plot_detectability_ratio(axes[2], summary)
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
        line = block[block["m"].eq(m)].sort_values("log10_mean_neff")
        if line.empty:
            continue
        plotted = ax.plot(
            line["log10_mean_neff"],
            line["power"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"m={m}",
        )[0]
        certified = line["certified_no_rank_rate"] >= 0.5
        if bool(certified.any()):
            ax.scatter(
                line.loc[certified, "log10_mean_neff"],
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

    x_values = summary["log10_mean_neff"].to_numpy(dtype=float)
    x_span = float(np.max(x_values) - np.min(x_values))
    x_padding = max(0.05 * x_span, 0.05)
    for ax in axes.ravel():
        ax.set_xlim(
            float(np.min(x_values) - x_padding),
            float(np.max(x_values) + x_padding),
        )
    for ax in axes[:, 0]:
        ax.set_ylabel("exact WEDF power")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10}$ mean calibration $N_{eff}$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
    )
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
    rows = []
    for row in summary.sort_values(["phase_x_bin", "phase_y_bin"]).itertuples(index=False):
        probability = float(row.exact_any_discovery)
        rows.append(
            {
                "export_version": TIKZ_EXPORT_VERSION,
                "figure": "figure2",
                "panel": "B",
                "panel_title": "P-value floor versus BH threshold",
                "plot_order": int(row.phase_x_bin) + 1,
                "group": "heatmap_cell",
                "method": "exact_wedf",
                "scenario": "baseline",
                "kappa": "",
                "m": int(row.m),
                "x": float(row.x),
                "y": float(row.y),
                "x_left": float(row.x_left),
                "x_right": float(row.x_right),
                "y_bottom": float(row.y_bottom),
                "y_top": float(row.y_top),
                "value": probability,
                "probability": probability,
                "count": int(row.count),
                "style_key": "discovery_probability",
                "label": "Pr(exact WEDF discovers)",
            }
        )
    pd.DataFrame(rows).to_csv(PANEL_B_HEATMAP_TIKZ_PATH, index=False)

    boundary_rows = []
    for point_order, x in enumerate(PHASE_VIEW_LIMITS, start=1):
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


def export_panel_c_tikz(summary: pd.DataFrame) -> None:
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
                    "panel": "C",
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
    pd.DataFrame(rows).to_csv(PANEL_C_DETECTABILITY_TIKZ_PATH, index=False)

    reference_rows = [
        {
            "export_version": TIKZ_EXPORT_VERSION,
            "figure": "figure2",
            "panel": "C",
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
    pd.DataFrame(reference_rows).to_csv(PANEL_C_REFERENCE_TIKZ_PATH, index=False)


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
    rows["x"] = rows["log10_mean_neff"]
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
        PANEL_C_DETECTABILITY_TIKZ_PATH,
        PANEL_C_REFERENCE_TIKZ_PATH,
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
    print(PANEL_C_DETECTABILITY_TIKZ_PATH)
    print(PANEL_C_REFERENCE_TIKZ_PATH)
    print(POWER_CONFIGURATIONS_TIKZ_PATH)


if __name__ == "__main__":
    main()
