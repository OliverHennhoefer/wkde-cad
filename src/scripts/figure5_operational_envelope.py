from __future__ import annotations

import argparse
import math
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "figure5"
POWER_ATLAS_FIGURE_PATH = OUT_DIR / "figure5_operational_power_atlas.png"
POWER_ATLAS_SUMMARY_PATH = OUT_DIR / "figure5_operational_power_atlas_summary.csv"
POWER_ATLAS_TIKZ_PATH = OUT_DIR / "figure5_operational_power_atlas_tikz.csv"
POWER_ATLAS_REFERENCE_TIKZ_PATH = OUT_DIR / "figure5_operational_power_atlas_reference_tikz.csv"
REQUIRED_ESS_FIGURE_PATH = OUT_DIR / "figure5_required_ess_design_map.png"
REQUIRED_ESS_SUMMARY_PATH = OUT_DIR / "figure5_required_ess_summary.csv"
REQUIRED_ESS_TIKZ_PATH = OUT_DIR / "figure5_required_ess_tikz.csv"
EMPIRICAL_PROJECTION_FIGURE_PATH = OUT_DIR / "figure5_empirical_projection.png"
EMPIRICAL_PROJECTION_SUMMARY_PATH = OUT_DIR / "figure5_empirical_projection_summary.csv"
EMPIRICAL_PROJECTION_TIKZ_PATH = OUT_DIR / "figure5_empirical_projection_tikz.csv"

EMPIRICAL_RESULTS_DIR = (
    REPO_ROOT / "outputs" / "empirical_benchmark" / "results" / "logistic"
)

SUMMARY_VERSION = "operational-envelope-v3"
REQUIRED_ESS_VERSION = "required-ess-v3"
EMPIRICAL_PROJECTION_VERSION = "empirical-projection-v1"
TIKZ_EXPORT_VERSION = "tikz-v1"
BASE_SEED = 20260518

D = 10
FINITE_KAPPAS = {
    "finite_k1": 1.0,
    "finite_k2": 2.0,
    "finite_k3": 3.0,
}
DELTA_SCORE = 1.0
N_EFF_BINS = np.linspace(1.3, 3.7, 13)
RHO_CANDIDATES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
N_CAL_MAX = 8000
MAX_ACCEPT_ATTEMPTS_PER_CELL = 5000

M_VALUES = [50, 100, 200, 500, 1000, 2000]
PI1_VALUES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
ALPHA_VALUES = [0.01, 0.025, 0.05, 0.10, 0.20]
REQUIRED_M_VALUES = M_VALUES
REQUIRED_PI1_VALUES = PI1_VALUES
REQUIRED_ALPHA_VALUES = [0.05, 0.10]

BASELINE_ALPHA = 0.10
BASELINE_PI1 = 0.05
BASELINE_M = 1000
BASELINE_N_ANOMALY = 10
POWER_TARGET = 0.80
RANK_BOUNDARY_COLOR = "#E53935"
RANK_BOUNDARY_LINEWIDTH = 2.4

DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)
DEFAULT_WCS_BATCH_SIZE = 512
DEFAULT_CELL_TRIALS = 100
SCORE_REGIMES = ("perfect", "finite_k1", "finite_k2", "finite_k3")


@dataclass(frozen=True)
class CellTask:
    collection: str
    sweep: str
    x_index: int
    x_value: float
    x_plot: float
    x_left: float
    x_right: float
    y_bin: int
    y_left: float
    y_right: float
    alpha: float
    m: int
    pi1: float | None
    n_anomaly_fixed: int | None
    cell_trials: int
    wcs_batch_size: int


def rng_for(*values: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *values]))


def effective_sample_size(weights: np.ndarray) -> float:
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


def _bh_rejection_counts_by_row(p_values_by_row: np.ndarray, alpha: float) -> np.ndarray:
    if p_values_by_row.size == 0:
        return np.zeros(p_values_by_row.shape[0], dtype=int)
    m = p_values_by_row.shape[1]
    sorted_p = np.sort(p_values_by_row, axis=1)
    thresholds = alpha * np.arange(1, m + 1) / m
    passed = sorted_p <= thresholds
    return np.where(passed.any(axis=1), m - np.argmax(passed[:, ::-1], axis=1), 0)


def _wcs_rejection_counts_by_candidate(
    candidate_idx: np.ndarray,
    test_scores: np.ndarray,
    calib_mass_strictly_above: np.ndarray,
    test_weights: np.ndarray,
    total_calib_weight: float,
    alpha: float,
    batch_size: int,
) -> np.ndarray:
    rejection_sizes = np.empty(len(candidate_idx), dtype=int)
    batch_size = max(1, int(batch_size))
    for start in range(0, len(candidate_idx), batch_size):
        batch_idx = candidate_idx[start : start + batch_size]
        auxiliary_p_values = (
            calib_mass_strictly_above[None, :]
            + test_weights[None, :]
            * (test_scores[None, :] < test_scores[batch_idx, None])
        ) / (total_calib_weight + test_weights[batch_idx])[:, None]
        auxiliary_p_values[np.arange(len(batch_idx)), batch_idx] = 0.0
        rejection_sizes[start : start + len(batch_idx)] = _bh_rejection_counts_by_row(
            auxiliary_p_values,
            alpha,
        )
    return rejection_sizes


def _select_by_pruning_metrics(indices: np.ndarray, metrics: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.array([], dtype=int)
    sorted_metrics = np.sort(metrics, kind="mergesort")
    passed = sorted_metrics <= np.arange(1, len(sorted_metrics) + 1)
    if not np.any(passed):
        return np.array([], dtype=int)
    cutoff = int(np.flatnonzero(passed)[-1] + 1)
    return np.sort(indices[metrics <= cutoff], kind="mergesort")


def homogeneous_wcs_decisions(
    p_values: np.ndarray,
    test_scores: np.ndarray,
    sorted_calib_scores: np.ndarray,
    sorted_calib_weights: np.ndarray,
    total_calib_weight: float,
    test_weights: np.ndarray,
    alpha: float,
    *,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    test_weights = np.asarray(test_weights, dtype=float)
    m = len(p_values)
    candidate_idx = np.flatnonzero(p_values <= alpha)
    if len(candidate_idx) == 0:
        return np.zeros(m, dtype=bool)

    cumulative_calib_weights = np.concatenate(([0.0], np.cumsum(sorted_calib_weights)))
    strict_tail_start = np.searchsorted(sorted_calib_scores, test_scores, side="right")
    calib_mass_strictly_above = total_calib_weight - cumulative_calib_weights[strict_tail_start]
    candidate_rejection_sizes = _wcs_rejection_counts_by_candidate(
        candidate_idx,
        test_scores,
        calib_mass_strictly_above,
        test_weights,
        total_calib_weight,
        alpha,
        batch_size,
    )
    first_step_mask = p_values[candidate_idx] <= alpha * candidate_rejection_sizes / m
    first_step_idx = candidate_idx[first_step_mask]
    if len(first_step_idx) == 0:
        return np.zeros(m, dtype=bool)

    selected_sizes = candidate_rejection_sizes[first_step_mask]
    rng = np.random.default_rng(seed)
    metrics = rng.uniform() * selected_sizes
    final_idx = _select_by_pruning_metrics(first_step_idx, metrics)
    decisions = np.zeros(m, dtype=bool)
    decisions[final_idx] = True
    return decisions


def empirical_auc(anomaly_scores: np.ndarray, inlier_scores: np.ndarray) -> float:
    sorted_inliers = np.sort(inlier_scores, kind="mergesort")
    less = np.searchsorted(sorted_inliers, anomaly_scores, side="left")
    less_equal = np.searchsorted(sorted_inliers, anomaly_scores, side="right")
    ties = less_equal - less
    return float(np.mean((less + 0.5 * ties) / len(sorted_inliers)))


def discovery_metrics(decisions: np.ndarray, y_true: np.ndarray) -> tuple[bool, float, float]:
    n_rejections = int(np.sum(decisions))
    true_rejections = int(np.sum(decisions & y_true))
    false_rejections = n_rejections - true_rejections
    n_anomaly = int(np.sum(y_true))
    power = true_rejections / n_anomaly if n_anomaly else 0.0
    fdr = false_rejections / n_rejections if n_rejections else 0.0
    return bool(true_rejections > 0), float(power), float(fdr)


def edges_from_centers(centers: list[float] | np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        width = max(abs(float(centers[0])) * 0.1, 0.5)
        return np.array([centers[0] - width, centers[0] + width])
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    return np.concatenate(
        [
            [centers[0] - (midpoints[0] - centers[0])],
            midpoints,
            [centers[-1] + (centers[-1] - midpoints[-1])],
        ]
    )


def sweep_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "m_fixed_pi",
            "title": r"$N_{eff}$ by $m$; fixed $\pi_1=0.05$",
            "x_label": r"test batch size $m$",
            "tick_values": [float(value) for value in M_VALUES],
            "x_plots": [math.log10(float(value)) for value in M_VALUES],
            "values": [float(value) for value in M_VALUES],
        },
        {
            "name": "m_fixed_count",
            "title": r"$N_{eff}$ by $m$; fixed 10 anomalies",
            "x_label": r"test batch size $m$",
            "tick_values": [float(value) for value in M_VALUES],
            "x_plots": [math.log10(float(value)) for value in M_VALUES],
            "values": [float(value) for value in M_VALUES],
        },
        {
            "name": "pi1_sweep",
            "title": r"$N_{eff}$ by anomaly rate",
            "x_label": r"anomaly rate $\pi_1$",
            "tick_values": [float(value) for value in PI1_VALUES],
            "x_plots": [math.log10(float(value)) for value in PI1_VALUES],
            "values": [float(value) for value in PI1_VALUES],
        },
        {
            "name": "alpha_sweep",
            "title": r"$N_{eff}$ by FDR level",
            "x_label": r"nominal FDR $\alpha$",
            "tick_values": [float(value) for value in ALPHA_VALUES],
            "x_plots": [math.log10(float(value)) for value in ALPHA_VALUES],
            "values": [float(value) for value in ALPHA_VALUES],
        },
    ]


def params_for_sweep(sweep: str, value: float) -> tuple[float, int, float | None, int | None]:
    if sweep == "m_fixed_pi":
        return BASELINE_ALPHA, int(round(value)), BASELINE_PI1, None
    if sweep == "m_fixed_count":
        return BASELINE_ALPHA, int(round(value)), None, BASELINE_N_ANOMALY
    if sweep == "pi1_sweep":
        return BASELINE_ALPHA, BASELINE_M, float(value), None
    if sweep == "alpha_sweep":
        return float(value), BASELINE_M, BASELINE_PI1, None
    raise ValueError(f"Unknown sweep: {sweep}")


def neff_config_candidates(y_center: float) -> list[tuple[int, float]]:
    target_neff = 10**y_center
    multipliers = [0.7, 0.9, 1.0, 1.15, 1.35]
    candidates: set[tuple[int, float]] = set()
    for rho in RHO_CANDIDATES:
        ideal_n = target_neff * math.exp(float(rho) ** 2)
        for multiplier in multipliers:
            n_cal = max(2, int(round(ideal_n * multiplier)))
            if n_cal <= N_CAL_MAX:
                candidates.add((n_cal, float(rho)))
    if not candidates:
        candidates.add((min(N_CAL_MAX, max(2, int(round(target_neff)))), 0.0))
    return sorted(candidates)


def n_anomaly_for(m: int, pi1: float | None, n_anomaly_fixed: int | None) -> int:
    if n_anomaly_fixed is not None:
        value = int(n_anomaly_fixed)
    else:
        value = int(round(float(pi1) * m))
    return min(max(1, value), max(1, m - 1))


def score_label(score_regime: str) -> str:
    if score_regime == "perfect":
        return "perfect score"
    if score_regime in FINITE_KAPPAS:
        return rf"finite score ($\kappa={FINITE_KAPPAS[score_regime]:g}$)"
    raise ValueError(f"Unknown score regime: {score_regime}")


def simulate_trial(
    *,
    alpha: float,
    m: int,
    pi1: float | None,
    n_anomaly_fixed: int | None,
    n_cal: int,
    rho: float,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    rho_code = int(round(rho * 1000))
    rng = rng_for(n_cal, m, rho_code, seed)

    calib_x = rng.normal(0.0, 1.0, size=(n_cal, D))
    calib_scores = rng.normal(0.0, 1.0, n_cal)
    calib_weights = np.exp(rho * calib_x[:, 0] - 0.5 * rho**2)
    total_calib_weight = float(np.sum(calib_weights))
    calib_ess = effective_sample_size(calib_weights)
    log10_neff = float(np.log10(calib_ess))

    order = np.argsort(calib_scores, kind="mergesort")
    sorted_calib_scores = calib_scores[order]
    sorted_calib_weights = calib_weights[order]
    suffix_calib_weights = np.concatenate(
        ([0.0], np.cumsum(sorted_calib_weights[::-1]))
    )[::-1]
    max_calib_score = float(np.max(calib_scores))

    n_anomaly = n_anomaly_for(m, pi1, n_anomaly_fixed)
    n_inlier = m - n_anomaly
    inlier_x = rng.normal(0.0, 1.0, size=(n_inlier, D))
    anomaly_x = rng.normal(0.0, 1.0, size=(n_anomaly, D))
    inlier_x[:, 0] += rho
    anomaly_x[:, 0] += rho
    test_x = np.vstack([inlier_x, anomaly_x])
    test_weights = np.exp(rho * test_x[:, 0] - 0.5 * rho**2)

    inlier_scores = rng.normal(0.0, 1.0, n_inlier)
    anomaly_noise = rng.normal(0.0, 1.0, n_anomaly)
    y_true = np.concatenate(
        [np.zeros(n_inlier, dtype=bool), np.ones(n_anomaly, dtype=bool)]
    )

    anomaly_weights = test_weights[n_inlier:]
    p_min_anomaly_values = anomaly_weights / (anomaly_weights + total_calib_weight)
    sorted_p_min = np.sort(p_min_anomaly_values, kind="mergesort")
    ranks = np.arange(1, n_anomaly + 1)
    rank_delta = float(np.min(sorted_p_min / (alpha * ranks / m)))
    log10_rank_delta = float(np.log10(rank_delta))

    metrics: dict[str, dict[str, Any]] = {}
    for regime in SCORE_REGIMES:
        if regime == "perfect":
            anomaly_score_value = max(max_calib_score, float(np.max(inlier_scores))) + DELTA_SCORE
            anomaly_scores = np.full(n_anomaly, anomaly_score_value)
        else:
            anomaly_scores = FINITE_KAPPAS[regime] + anomaly_noise
        test_scores = np.concatenate([inlier_scores, anomaly_scores])
        p_values = weighted_tail_p_values(
            sorted_calib_scores,
            suffix_calib_weights,
            total_calib_weight,
            test_scores,
            test_weights,
        )
        decisions = homogeneous_wcs_decisions(
            p_values,
            test_scores,
            sorted_calib_scores,
            sorted_calib_weights,
            total_calib_weight,
            test_weights,
            alpha,
            seed=BASE_SEED + seed,
            batch_size=batch_size,
        )
        any_discovery, power, fdr = discovery_metrics(decisions, y_true)
        metrics[regime] = {
            "any_discovery": any_discovery,
            "power": power,
            "fdr": fdr,
            "auroc": empirical_auc(anomaly_scores, inlier_scores),
            "perfect_separation_from_calibration": bool(
                float(np.min(anomaly_scores)) > max_calib_score
            ),
        }

    return {
        "n_cal": n_cal,
        "rho": rho,
        "calib_ess": calib_ess,
        "log10_neff": log10_neff,
        "n_anomaly": n_anomaly,
        "actual_pi1": n_anomaly / m,
        "rank_delta": rank_delta,
        "log10_rank_delta": log10_rank_delta,
        "certified_no_rank_discovery": bool(rank_delta > 1.0),
        "metrics": metrics,
    }


def aggregate_cell(task: CellTask, trial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for regime in SCORE_REGIMES:
        count = len(trial_rows)
        metrics = [row["metrics"][regime] for row in trial_rows]
        rows.append(
            {
                "summary_version": SUMMARY_VERSION,
                "collection": task.collection,
                "sweep": task.sweep,
                "score_regime": regime,
                "x_index": task.x_index,
                "x_value": task.x_value,
                "x_plot": task.x_plot,
                "x_left": task.x_left,
                "x_right": task.x_right,
                "log10_neff_bin": task.y_bin,
                "log10_neff_left": task.y_left,
                "log10_neff_right": task.y_right,
                "log10_neff_center": (task.y_left + task.y_right) / 2.0,
                "alpha": task.alpha,
                "m": task.m,
                "target_pi1": np.nan if task.pi1 is None else task.pi1,
                "n_anomaly_fixed": (
                    np.nan if task.n_anomaly_fixed is None else task.n_anomaly_fixed
                ),
                "count": count,
                "mean_n_cal": float(np.mean([row["n_cal"] for row in trial_rows])),
                "mean_rho": float(np.mean([row["rho"] for row in trial_rows])),
                "mean_neff": float(np.mean([row["calib_ess"] for row in trial_rows])),
                "log10_mean_neff": float(
                    np.log10(np.mean([row["calib_ess"] for row in trial_rows]))
                ),
                "mean_n_anomaly": float(
                    np.mean([row["n_anomaly"] for row in trial_rows])
                ),
                "actual_pi1": float(np.mean([row["actual_pi1"] for row in trial_rows])),
                "discovery_probability": float(
                    np.mean([metric["any_discovery"] for metric in metrics])
                ),
                "power": float(np.mean([metric["power"] for metric in metrics])),
                "fdr": float(np.mean([metric["fdr"] for metric in metrics])),
                "mean_log10_rank_delta": float(
                    np.mean([row["log10_rank_delta"] for row in trial_rows])
                ),
                "median_log10_rank_delta": float(
                    np.median([row["log10_rank_delta"] for row in trial_rows])
                ),
                "certified_no_rank_rate": float(
                    np.mean([row["certified_no_rank_discovery"] for row in trial_rows])
                ),
                "mean_auroc": float(np.mean([metric["auroc"] for metric in metrics])),
                "perfect_separation_rate": float(
                    np.mean(
                        [
                            metric["perfect_separation_from_calibration"]
                            for metric in metrics
                        ]
                    )
                ),
            }
        )
    return rows


def simulate_cell(task: CellTask) -> list[dict[str, Any]]:
    y_center = (task.y_left + task.y_right) / 2.0
    configs = neff_config_candidates(y_center)
    trial_rows: list[dict[str, Any]] = []
    attempts = 0
    seed_prefix = (
        (0 if task.collection == "atlas" else 50)
        + 100 * sweep_code(task.sweep)
        + task.x_index
        + 1000 * task.y_bin
    )

    while (
        len(trial_rows) < int(task.cell_trials)
        and attempts < int(MAX_ACCEPT_ATTEMPTS_PER_CELL)
    ):
        n_cal, rho = configs[attempts % len(configs)]
        row = simulate_trial(
            alpha=task.alpha,
            m=task.m,
            pi1=task.pi1,
            n_anomaly_fixed=task.n_anomaly_fixed,
            n_cal=n_cal,
            rho=rho,
            seed=seed_prefix * 100000 + attempts,
            batch_size=task.wcs_batch_size,
        )
        attempts += 1
        if not task.y_left <= float(row["log10_neff"]) < task.y_right:
            if task.y_bin == len(N_EFF_BINS) - 2 and float(row["log10_neff"]) <= task.y_right:
                pass
            else:
                continue
        trial_rows.append(row)

    if len(trial_rows) < int(task.cell_trials):
        raise RuntimeError(
            "Could not fill Figure 5 cell: "
            f"collection={task.collection}, sweep={task.sweep}, "
            f"x_index={task.x_index}, y_bin={task.y_bin}, "
            f"accepted={len(trial_rows)}, attempts={attempts}."
        )
    return aggregate_cell(task, trial_rows)


def sweep_code(sweep: str) -> int:
    names = [spec["name"] for spec in sweep_specs()] + ["required_design"]
    return names.index(sweep)


def atlas_tasks(cell_trials: int, wcs_batch_size: int) -> list[CellTask]:
    tasks = []
    for spec in sweep_specs():
        x_edges = edges_from_centers(spec["x_plots"])
        for x_index, value in enumerate(spec["values"]):
            alpha, m, pi1, n_anomaly_fixed = params_for_sweep(spec["name"], value)
            for y_bin in range(len(N_EFF_BINS) - 1):
                tasks.append(
                    CellTask(
                        collection="atlas",
                        sweep=spec["name"],
                        x_index=x_index,
                        x_value=float(value),
                        x_plot=float(spec["x_plots"][x_index]),
                        x_left=float(x_edges[x_index]),
                        x_right=float(x_edges[x_index + 1]),
                        y_bin=y_bin,
                        y_left=float(N_EFF_BINS[y_bin]),
                        y_right=float(N_EFF_BINS[y_bin + 1]),
                        alpha=alpha,
                        m=m,
                        pi1=pi1,
                        n_anomaly_fixed=n_anomaly_fixed,
                        cell_trials=cell_trials,
                        wcs_batch_size=wcs_batch_size,
                    )
                )
    return tasks


def required_tasks(cell_trials: int, wcs_batch_size: int) -> list[CellTask]:
    m_plots = [math.log10(float(value)) for value in REQUIRED_M_VALUES]
    m_edges = edges_from_centers(m_plots)
    tasks = []
    x_index = 0
    for alpha in REQUIRED_ALPHA_VALUES:
        for pi1 in REQUIRED_PI1_VALUES:
            for m_idx, m in enumerate(REQUIRED_M_VALUES):
                for y_bin in range(len(N_EFF_BINS) - 1):
                    tasks.append(
                        CellTask(
                            collection="required",
                            sweep="required_design",
                            x_index=x_index,
                            x_value=float(m),
                            x_plot=float(m_plots[m_idx]),
                            x_left=float(m_edges[m_idx]),
                            x_right=float(m_edges[m_idx + 1]),
                            y_bin=y_bin,
                            y_left=float(N_EFF_BINS[y_bin]),
                            y_right=float(N_EFF_BINS[y_bin + 1]),
                            alpha=float(alpha),
                            m=int(m),
                            pi1=float(pi1),
                            n_anomaly_fixed=None,
                            cell_trials=cell_trials,
                            wcs_batch_size=wcs_batch_size,
                        )
                    )
                x_index += 1
    return tasks


def run_cells(tasks: list[CellTask], workers: int, description: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        iterator = map(simulate_cell, tasks)
        for cell_rows in tqdm(iterator, total=len(tasks), desc=description, unit="cell"):
            rows.extend(cell_rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for cell_rows in tqdm(
                executor.map(simulate_cell, tasks),
                total=len(tasks),
                desc=description,
                unit="cell",
            ):
                rows.extend(cell_rows)
    return pd.DataFrame(rows)


def summary_is_current(path: Path, version: str) -> bool:
    if not path.exists():
        return False
    header = pd.read_csv(path, nrows=1)
    return not header.empty and str(header["summary_version"].iloc[0]) == version


def build_power_atlas_summary(
    *, workers: int, cell_trials: int, wcs_batch_size: int
) -> pd.DataFrame:
    summary = run_cells(
        atlas_tasks(cell_trials, wcs_batch_size),
        workers,
        "Figure 5 power-atlas cells",
    )
    validate_power_atlas_summary(summary)
    summary.to_csv(POWER_ATLAS_SUMMARY_PATH, index=False)
    return summary


def load_or_build_power_atlas_summary(
    *, workers: int, cell_trials: int, wcs_batch_size: int, force: bool
) -> pd.DataFrame:
    if not force and summary_is_current(POWER_ATLAS_SUMMARY_PATH, SUMMARY_VERSION):
        print("loading existing Figure 5 power-atlas summary", flush=True)
        return pd.read_csv(POWER_ATLAS_SUMMARY_PATH)
    return build_power_atlas_summary(
        workers=workers,
        cell_trials=cell_trials,
        wcs_batch_size=wcs_batch_size,
    )


def derive_required_ess_summary(required_power: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["score_regime", "alpha", "m", "target_pi1"]
    for group_key, block in required_power.groupby(group_cols, dropna=False):
        score_regime, alpha, m, pi1 = group_key
        line = block.sort_values("log10_mean_neff")
        passing = line[line["power"] >= POWER_TARGET]
        if passing.empty:
            status = "out_of_range"
            required_log10_neff = math.nan
            required_neff = math.nan
        else:
            status = "attained"
            first = passing.iloc[0]
            required_log10_neff = float(first["log10_mean_neff"])
            required_neff = float(first["mean_neff"])
        rows.append(
            {
                "summary_version": REQUIRED_ESS_VERSION,
                "score_regime": score_regime,
                "alpha": float(alpha),
                "m": int(m),
                "pi1": float(pi1),
                "power_target": POWER_TARGET,
                "status": status,
                "required_log10_neff": required_log10_neff,
                "required_neff": required_neff,
                "max_power": float(line["power"].max()),
                "min_certified_no_rank_rate": float(
                    line["certified_no_rank_rate"].min()
                ),
                "count": int(line["count"].sum()),
            }
        )
    summary = pd.DataFrame(rows)
    validate_required_ess_summary(summary)
    return summary


def build_required_ess_summary(
    *, workers: int, cell_trials: int, wcs_batch_size: int
) -> pd.DataFrame:
    required_power = run_cells(
        required_tasks(cell_trials, wcs_batch_size),
        workers,
        "Figure 5 required-ESS cells",
    )
    validate_power_atlas_summary(required_power, expected_collection="required")
    summary = derive_required_ess_summary(required_power)
    summary.to_csv(REQUIRED_ESS_SUMMARY_PATH, index=False)
    return summary


def load_or_build_required_ess_summary(
    *, workers: int, cell_trials: int, wcs_batch_size: int, force: bool
) -> pd.DataFrame:
    if not force and summary_is_current(REQUIRED_ESS_SUMMARY_PATH, REQUIRED_ESS_VERSION):
        print("loading existing Figure 5 required-ESS summary", flush=True)
        return pd.read_csv(REQUIRED_ESS_SUMMARY_PATH)
    return build_required_ess_summary(
        workers=workers,
        cell_trials=cell_trials,
        wcs_batch_size=wcs_batch_size,
    )


def validate_power_atlas_summary(
    summary: pd.DataFrame,
    *,
    expected_collection: str = "atlas",
) -> None:
    if summary.empty:
        raise RuntimeError("Figure 5 power summary is empty.")
    if set(summary["collection"]) != {expected_collection}:
        raise RuntimeError("Figure 5 summary contains an unexpected collection.")
    if (summary["count"] <= 0).any():
        raise RuntimeError("Figure 5 summary contains zero-count cells.")
    probability_columns = [
        "discovery_probability",
        "power",
        "fdr",
        "certified_no_rank_rate",
        "mean_auroc",
        "perfect_separation_rate",
    ]
    for column in probability_columns:
        if not ((summary[column] >= 0.0) & (summary[column] <= 1.0)).all():
            raise RuntimeError(f"Probability-like column out of range: {column}.")
    if not np.isfinite(summary["median_log10_rank_delta"]).all():
        raise RuntimeError("Rank-aware detectability diagnostics must be finite.")
    perfect = summary[summary["score_regime"].eq("perfect")]
    if not (perfect["perfect_separation_rate"] == 1.0).all():
        raise RuntimeError("Perfect-score cells must exceed every calibration score.")

    if expected_collection == "atlas":
        expected = {
            (spec["name"], score, x_idx, y_bin)
            for spec in sweep_specs()
            for score in SCORE_REGIMES
            for x_idx in range(len(spec["values"]))
            for y_bin in range(len(N_EFF_BINS) - 1)
        }
        observed = {
            (
                str(row.sweep),
                str(row.score_regime),
                int(row.x_index),
                int(row.log10_neff_bin),
            )
            for row in summary.itertuples(index=False)
        }
        if observed != expected:
            raise RuntimeError(
                f"Figure 5 atlas cells incomplete: missing={len(expected - observed)}, "
                f"unexpected={len(observed - expected)}."
            )
    else:
        expected_required = {
            (score, float(alpha), int(m), float(pi1), y_bin)
            for score in SCORE_REGIMES
            for alpha in REQUIRED_ALPHA_VALUES
            for m in REQUIRED_M_VALUES
            for pi1 in REQUIRED_PI1_VALUES
            for y_bin in range(len(N_EFF_BINS) - 1)
        }
        observed_required = {
            (
                str(row.score_regime),
                float(row.alpha),
                int(row.m),
                float(row.target_pi1),
                int(row.log10_neff_bin),
            )
            for row in summary.itertuples(index=False)
        }
        if observed_required != expected_required:
            raise RuntimeError(
                "Figure 5 required-ESS cells incomplete: "
                f"missing={len(expected_required - observed_required)}, "
                f"unexpected={len(observed_required - expected_required)}."
            )


def validate_required_ess_summary(summary: pd.DataFrame) -> None:
    expected = {
        (score, float(alpha), int(m), float(pi1))
        for score in SCORE_REGIMES
        for alpha in REQUIRED_ALPHA_VALUES
        for m in REQUIRED_M_VALUES
        for pi1 in REQUIRED_PI1_VALUES
    }
    observed = {
        (str(row.score_regime), float(row.alpha), int(row.m), float(row.pi1))
        for row in summary.itertuples(index=False)
    }
    if observed != expected:
        raise RuntimeError(
            f"Required-ESS summary incomplete: missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}."
        )
    if not summary["status"].isin(["attained", "out_of_range"]).all():
        raise RuntimeError("Required-ESS summary has invalid status values.")
    attained = summary[summary["status"].eq("attained")]
    if not np.isfinite(attained["required_log10_neff"]).all():
        raise RuntimeError("Attained required-ESS rows must have finite thresholds.")
    out_of_range = summary[summary["status"].eq("out_of_range")]
    if out_of_range["required_log10_neff"].notna().any():
        raise RuntimeError("Out-of-range required-ESS rows must not have thresholds.")
    if not ((summary["max_power"] >= 0.0) & (summary["max_power"] <= 1.0)).all():
        raise RuntimeError("Required-ESS max power must lie in [0, 1].")


def heatmap_matrix(
    summary: pd.DataFrame,
    *,
    sweep: str,
    score_regime: str,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block = summary[
        summary["sweep"].eq(sweep) & summary["score_regime"].eq(score_regime)
    ]
    x_edges = np.unique(np.concatenate([block["x_left"].to_numpy(), block["x_right"].to_numpy()]))
    y_edges = np.unique(
        np.concatenate(
            [
                block["log10_neff_left"].to_numpy(),
                block["log10_neff_right"].to_numpy(),
            ]
        )
    )
    matrix = np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan)
    x_index_by_left = {float(edge): idx for idx, edge in enumerate(x_edges[:-1])}
    y_index_by_left = {float(edge): idx for idx, edge in enumerate(y_edges[:-1])}
    for row in block.itertuples(index=False):
        x_idx = x_index_by_left[float(row.x_left)]
        y_idx = y_index_by_left[float(row.log10_neff_left)]
        matrix[x_idx, y_idx] = float(getattr(row, value_column))
    return x_edges, y_edges, matrix


def rank_boundary_points(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    rank_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    points: list[tuple[float, float]] = []
    for x_value, column in zip(x_centers, rank_delta):
        values = np.asarray(column, dtype=float)
        finite = np.isfinite(values)
        if int(np.sum(finite)) < 2:
            continue
        y_values = y_centers[finite]
        values = values[finite]
        crossings = []
        exact = np.flatnonzero(np.isclose(values, 0.0, atol=1e-12))
        crossings.extend(float(y_values[idx]) for idx in exact)
        for idx in range(len(values) - 1):
            lower = float(values[idx])
            upper = float(values[idx + 1])
            if lower == upper:
                continue
            if (lower < 0.0 < upper) or (upper < 0.0 < lower):
                y0 = float(y_values[idx])
                y1 = float(y_values[idx + 1])
                crossings.append(y0 + (0.0 - lower) * (y1 - y0) / (upper - lower))
        if crossings:
            points.append((float(x_value), float(np.mean(crossings))))

    if len(points) < 2:
        return None

    x_line = np.array([point[0] for point in points], dtype=float)
    y_line = np.array([point[1] for point in points], dtype=float)
    order = np.argsort(x_line, kind="mergesort")
    x_line = x_line[order]
    y_line = y_line[order]

    x_pad = 0.04 * float(x_edges[-1] - x_edges[0])
    x_start = float(x_edges[0] - x_pad)
    x_end = float(x_edges[-1] + x_pad)
    if x_line[1] != x_line[0]:
        slope = (y_line[1] - y_line[0]) / (x_line[1] - x_line[0])
        first_x = float(x_line[0])
        first_y = float(y_line[0])
        x_line = np.insert(x_line, 0, x_start)
        y_line = np.insert(y_line, 0, first_y + slope * (x_start - first_x))
    if x_line[-1] != x_line[-2]:
        slope = (y_line[-1] - y_line[-2]) / (x_line[-1] - x_line[-2])
        last_x = float(x_line[-1])
        last_y = float(y_line[-1])
        x_line = np.append(x_line, x_end)
        y_line = np.append(y_line, last_y + slope * (x_end - last_x))
    return x_line, y_line


def draw_rank_boundary(
    ax: plt.Axes,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    rank_delta: np.ndarray,
) -> None:
    points = rank_boundary_points(x_edges, y_edges, rank_delta)
    if points is None:
        return
    x_line, y_line = points
    ax.plot(
        x_line,
        y_line,
        color=RANK_BOUNDARY_COLOR,
        linewidth=RANK_BOUNDARY_LINEWIDTH,
        solid_capstyle="round",
        zorder=4,
    )


def format_tick(value: float) -> str:
    if value >= 1:
        return f"{int(value)}"
    return f"{value:g}"


def add_certified_outlines(
    ax: plt.Axes,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    certified_matrix: np.ndarray,
) -> None:
    for x_idx in range(certified_matrix.shape[0]):
        for y_idx in range(certified_matrix.shape[1]):
            if certified_matrix[x_idx, y_idx] < 0.5:
                continue
            ax.add_patch(
                Rectangle(
                    (x_edges[x_idx], y_edges[y_idx]),
                    x_edges[x_idx + 1] - x_edges[x_idx],
                    y_edges[y_idx + 1] - y_edges[y_idx],
                    fill=False,
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.8,
                )
            )


def plot_power_atlas(summary: pd.DataFrame) -> None:
    specs = sweep_specs()
    fig, axes = plt.subplots(
        len(SCORE_REGIMES),
        len(specs),
        figsize=(4.2 * len(specs), 4.2 * len(SCORE_REGIMES)),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(len(SCORE_REGIMES), len(specs))
    mesh = None
    for col_idx, spec in enumerate(specs):
        x_tick_positions = [float(value) for value in spec["x_plots"]]
        x_tick_labels = [format_tick(float(value)) for value in spec["tick_values"]]
        for row_idx, score_regime in enumerate(SCORE_REGIMES):
            ax = axes[row_idx, col_idx]
            x_edges, y_edges, power = heatmap_matrix(
                summary,
                sweep=spec["name"],
                score_regime=score_regime,
                value_column="power",
            )
            _, _, rank_delta = heatmap_matrix(
                summary,
                sweep=spec["name"],
                score_regime=score_regime,
                value_column="median_log10_rank_delta",
            )
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                power.T,
                cmap="viridis",
                norm=Normalize(vmin=0.0, vmax=1.0),
                shading="flat",
            )
            if min(rank_delta.shape) >= 2:
                draw_rank_boundary(ax, x_edges, y_edges, rank_delta)
            ax.set_title(spec["title"] if row_idx == 0 else "")
            ax.set_xlabel(spec["x_label"])
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(x_tick_labels, rotation=35, ha="right")
            ax.grid(False)
            ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
            ax.set_ylim(float(N_EFF_BINS[0]), float(N_EFF_BINS[-1]))
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}$ calibration $N_{eff}$")

    for row_idx, score_regime in enumerate(SCORE_REGIMES):
        axes[row_idx, 0].annotate(
            score_label(score_regime),
            xy=(-0.24, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="right",
            va="center",
            fontsize=11,
        )
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("statistical power")
    fig.suptitle("Operational power atlas for weighted conformal CAD", fontsize=14)
    fig.savefig(POWER_ATLAS_FIGURE_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def required_matrix(
    summary: pd.DataFrame,
    *,
    score_regime: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray, pd.DataFrame]:
    block = summary[
        summary["score_regime"].eq(score_regime)
        & np.isclose(summary["alpha"].astype(float), float(alpha))
    ].copy()
    x_centers = np.array([math.log10(float(value)) for value in REQUIRED_M_VALUES])
    y_centers = np.array([math.log10(float(value)) for value in REQUIRED_PI1_VALUES])
    x_edges = edges_from_centers(x_centers)
    y_edges = edges_from_centers(y_centers)
    matrix = np.full((len(x_centers), len(y_centers)), np.nan)
    for row in block.itertuples(index=False):
        x_idx = REQUIRED_M_VALUES.index(int(row.m))
        y_idx = REQUIRED_PI1_VALUES.index(float(row.pi1))
        matrix[x_idx, y_idx] = float(row.required_log10_neff)
    return x_edges, y_edges, np.ma.masked_invalid(matrix), block


def plot_required_ess(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(SCORE_REGIMES),
        len(REQUIRED_ALPHA_VALUES),
        figsize=(6.0 * len(REQUIRED_ALPHA_VALUES), 3.8 * len(SCORE_REGIMES)),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    if len(SCORE_REGIMES) == 1 or len(REQUIRED_ALPHA_VALUES) == 1:
        axes = np.asarray(axes).reshape(len(SCORE_REGIMES), len(REQUIRED_ALPHA_VALUES))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e7eb")
    mesh = None
    for row_idx, score_regime in enumerate(SCORE_REGIMES):
        for col_idx, alpha in enumerate(REQUIRED_ALPHA_VALUES):
            ax = axes[row_idx, col_idx]
            x_edges, y_edges, matrix, block = required_matrix(
                summary,
                score_regime=score_regime,
                alpha=float(alpha),
            )
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                matrix.T,
                cmap=cmap,
                norm=Normalize(vmin=float(N_EFF_BINS[0]), vmax=float(N_EFF_BINS[-1])),
                shading="flat",
            )
            for row in block[block["status"].eq("out_of_range")].itertuples(index=False):
                ax.text(
                    math.log10(float(row.m)),
                    math.log10(float(row.pi1)),
                    f">{N_EFF_BINS[-1]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#111111",
                )
            ax.set_title(rf"$\alpha={float(alpha):g}$")
            ax.set_xticks([math.log10(float(value)) for value in REQUIRED_M_VALUES])
            ax.set_xticklabels([format_tick(float(value)) for value in REQUIRED_M_VALUES])
            ax.set_yticks([math.log10(float(value)) for value in REQUIRED_PI1_VALUES])
            ax.set_yticklabels([format_tick(float(value)) for value in REQUIRED_PI1_VALUES])
            ax.grid(alpha=0.16, linewidth=0.5)
        axes[row_idx, 0].set_ylabel(
            f"{score_label(score_regime)}\nanomaly rate"
        )
    for ax in axes[-1, :]:
        ax.set_xlabel(r"test batch size $m$")
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label(r"minimum $\log_{10} N_{eff}$ for 80% power")
    fig.suptitle("Required effective calibration size design map", fontsize=14)
    fig.savefig(REQUIRED_ESS_FIGURE_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def export_power_atlas_tikz(summary: pd.DataFrame) -> None:
    spec_by_name = {str(spec["name"]): spec for spec in sweep_specs()}
    sweep_order = {str(spec["name"]): idx for idx, spec in enumerate(sweep_specs())}
    score_order = {score: idx for idx, score in enumerate(SCORE_REGIMES)}

    rows = []
    for row in summary.sort_values(
        ["score_regime", "sweep", "x_index", "log10_neff_bin"],
    ).itertuples(index=False):
        spec = spec_by_name[str(row.sweep)]
        panel_row = score_order[str(row.score_regime)] + 1
        panel_col = sweep_order[str(row.sweep)] + 1
        rows.append(
            {
                "export_version": TIKZ_EXPORT_VERSION,
                "figure": "figure5",
                "panel": f"power_atlas_r{panel_row}_c{panel_col}",
                "panel_title": spec["title"],
                "plot_order": (panel_row - 1) * len(sweep_specs()) + panel_col,
                "group": "heatmap_cell",
                "method": "weighted_wcs_homogeneous",
                "score_regime": row.score_regime,
                "sweep": row.sweep,
                "x": float(row.x_plot),
                "y": float(row.log10_neff_center),
                "x_left": float(row.x_left),
                "x_right": float(row.x_right),
                "y_bottom": float(row.log10_neff_left),
                "y_top": float(row.log10_neff_right),
                "x_value": float(row.x_value),
                "x_label": spec["x_label"],
                "alpha": float(row.alpha),
                "m": int(row.m),
                "target_pi1": row.target_pi1,
                "n_anomaly_fixed": row.n_anomaly_fixed,
                "value": float(row.power),
                "power": float(row.power),
                "discovery_probability": float(row.discovery_probability),
                "fdr": float(row.fdr),
                "median_log10_rank_delta": float(row.median_log10_rank_delta),
                "certified_no_rank_rate": float(row.certified_no_rank_rate),
                "mean_auroc": float(row.mean_auroc),
                "count": int(row.count),
                "style_key": f"{row.score_regime}_{row.sweep}",
                "label": "statistical power",
            }
        )
    pd.DataFrame(rows).to_csv(POWER_ATLAS_TIKZ_PATH, index=False)


def export_power_atlas_reference_tikz() -> None:
    rows = []
    for spec in sweep_specs():
        panel_col = [item["name"] for item in sweep_specs()].index(spec["name"]) + 1
        for score_regime in SCORE_REGIMES:
            panel_row = SCORE_REGIMES.index(score_regime) + 1
            panel = f"power_atlas_r{panel_row}_c{panel_col}"
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure5",
                    "panel": panel,
                    "panel_title": spec["title"],
                    "plot_order": panel_row * 100 + panel_col,
                    "group": "contour_reference",
                    "object_type": "contour_level",
                    "score_regime": score_regime,
                    "sweep": spec["name"],
                    "x": "",
                    "y": "",
                    "x_end": "",
                    "y_end": "",
                    "value": 0.0,
                    "style_key": "median_log10_rank_delta_zero",
                    "label": "rank-aware boundary",
                }
            )
    pd.DataFrame(rows).to_csv(POWER_ATLAS_REFERENCE_TIKZ_PATH, index=False)


def export_required_ess_tikz(summary: pd.DataFrame) -> None:
    x_centers = np.array([math.log10(float(value)) for value in REQUIRED_M_VALUES])
    y_centers = np.array([math.log10(float(value)) for value in REQUIRED_PI1_VALUES])
    x_edges = edges_from_centers(x_centers)
    y_edges = edges_from_centers(y_centers)
    m_index = {int(value): idx for idx, value in enumerate(REQUIRED_M_VALUES)}
    pi_index = {float(value): idx for idx, value in enumerate(REQUIRED_PI1_VALUES)}
    alpha_order = {float(alpha): idx + 1 for idx, alpha in enumerate(REQUIRED_ALPHA_VALUES)}

    rows = []
    for row in summary.sort_values(["score_regime", "alpha", "m", "pi1"]).itertuples(
        index=False,
    ):
        score_idx = SCORE_REGIMES.index(str(row.score_regime)) + 1
        alpha_idx = alpha_order[float(row.alpha)]
        x_idx = m_index[int(row.m)]
        y_idx = pi_index[float(row.pi1)]
        rows.append(
            {
                "export_version": TIKZ_EXPORT_VERSION,
                "figure": "figure5",
                "panel": f"required_ess_r{score_idx}_c{alpha_idx}",
                "panel_title": rf"$\alpha={float(row.alpha):g}$",
                "plot_order": (score_idx - 1) * len(REQUIRED_ALPHA_VALUES) + alpha_idx,
                "group": "heatmap_cell",
                "method": "weighted_wcs_homogeneous",
                "score_regime": row.score_regime,
                "alpha": float(row.alpha),
                "m": int(row.m),
                "pi1": float(row.pi1),
                "x": float(x_centers[x_idx]),
                "y": float(y_centers[y_idx]),
                "x_left": float(x_edges[x_idx]),
                "x_right": float(x_edges[x_idx + 1]),
                "y_bottom": float(y_edges[y_idx]),
                "y_top": float(y_edges[y_idx + 1]),
                "status": row.status,
                "value": row.required_log10_neff,
                "required_log10_neff": row.required_log10_neff,
                "required_neff": row.required_neff,
                "max_power": float(row.max_power),
                "power_target": float(row.power_target),
                "count": int(row.count),
                "style_key": f"{row.score_regime}_alpha_{float(row.alpha):g}",
                "label": "minimum log10 N_eff for target power",
            }
        )
    pd.DataFrame(rows).to_csv(REQUIRED_ESS_TIKZ_PATH, index=False)


def export_empirical_projection_tikz(summary: pd.DataFrame) -> None:
    columns = [
        "export_version",
        "figure",
        "panel",
        "panel_title",
        "plot_order",
        "group",
        "method",
        "dataset",
        "severity",
        "x",
        "y",
        "x_left",
        "x_right",
        "y_bottom",
        "y_top",
        "value",
        "power",
        "fdr",
        "approx_log10_rank_delta",
        "certified_proxy_rate",
        "count",
        "style_key",
        "label",
    ]
    rows = []
    for row in summary.sort_values(["severity", "dataset", "approach"]).itertuples(
        index=False,
    ):
        rows.append(
            {
                "export_version": TIKZ_EXPORT_VERSION,
                "figure": "figure5",
                "panel": "empirical_projection",
                "panel_title": "Empirical projection",
                "plot_order": 1,
                "group": "empirical_point",
                "method": row.approach,
                "dataset": row.dataset,
                "severity": float(row.severity),
                "x": float(row.log10_m_over_alpha),
                "y": float(row.log10_inverse_p_floor_proxy),
                "x_left": "",
                "x_right": "",
                "y_bottom": "",
                "y_top": "",
                "value": float(row.power_mean),
                "power": float(row.power_mean),
                "fdr": float(row.fdr_mean),
                "approx_log10_rank_delta": float(row.approx_log10_rank_delta),
                "certified_proxy_rate": float(row.certified_proxy_rate),
                "count": int(row.count),
                "style_key": f"severity_{float(row.severity):g}",
                "label": "mean empirical power",
            }
        )

    if rows:
        x_values = [float(row["x"]) for row in rows]
        y_values = [float(row["y"]) for row in rows]
        lower = min(x_values + y_values)
        upper = max(x_values + y_values)
        padding = max(0.1 * (upper - lower), 0.1)
        for plot_order, point in enumerate([lower - padding, upper + padding], start=1):
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure5",
                    "panel": "empirical_projection",
                    "panel_title": "Empirical projection",
                    "plot_order": 100 + plot_order,
                    "group": "resolution_boundary",
                    "method": "",
                    "dataset": "",
                    "severity": "",
                    "x": point,
                    "y": point,
                    "x_left": "",
                    "x_right": "",
                    "y_bottom": "",
                    "y_top": "",
                    "value": "",
                    "power": "",
                    "fdr": "",
                    "approx_log10_rank_delta": "",
                    "certified_proxy_rate": "",
                    "count": "",
                    "style_key": "diagonal_y_equals_x",
                    "label": r"$1/p_{\min}^{proxy}=m/\alpha$",
                }
            )
    pd.DataFrame(rows, columns=columns).to_csv(
        EMPIRICAL_PROJECTION_TIKZ_PATH,
        index=False,
    )


def export_tikz_csvs(
    power_summary: pd.DataFrame,
    required_summary: pd.DataFrame,
    empirical_summary: pd.DataFrame,
) -> None:
    export_power_atlas_tikz(power_summary)
    export_power_atlas_reference_tikz()
    export_required_ess_tikz(required_summary)
    export_empirical_projection_tikz(empirical_summary)


EMPIRICAL_COLUMNS = [
    "summary_version",
    "dataset",
    "severity",
    "approach",
    "alpha",
    "m_mean",
    "actual_anomaly_rate_mean",
    "power_mean",
    "fdr_mean",
    "calib_neff_mean",
    "p_floor_proxy_mean",
    "log10_m_over_alpha",
    "log10_inverse_p_floor_proxy",
    "approx_log10_rank_delta",
    "certified_proxy_rate",
    "count",
]


def load_empirical_alpha(results_dir: Path) -> float:
    config_path = results_dir / "config.toml"
    if not config_path.exists():
        return BASELINE_ALPHA
    with open(config_path, "rb") as config_file:
        config = tomllib.load(config_file)
    return float(config.get("conformal", {}).get("fdr_rate", BASELINE_ALPHA))


def empirical_row_diagnostic(row: pd.Series, alpha: float) -> dict[str, Any] | None:
    m = int(row.get("test_size", row.get("n_test", 0)))
    if m <= 1:
        return None
    n_anomaly = int(row.get("n_test_anomaly", round(float(row.get("actual_anomaly_rate", 0.0)) * m)))
    n_anomaly = min(max(1, n_anomaly), m - 1)
    train_size = max(1, int(row.get("train_size", row.get("n_train", 1))))
    weighted = "weighted" in str(row.get("approach", ""))

    if weighted and float(row.get("used_calib_weight_ess", 0.0)) > 0.0:
        calib_neff = float(row.get("used_calib_weight_ess"))
        calib_weight_mean = float(row.get("used_calib_weight_mean", 1.0))
        test_weight = float(row.get("used_test_weight_max", row.get("used_test_weight_mean", 1.0)))
        total_calib_weight = max(train_size * calib_weight_mean, 1e-12)
        p_floor_proxy = test_weight / (test_weight + total_calib_weight)
    else:
        calib_neff = float(train_size)
        p_floor_proxy = 1.0 / (train_size + 1.0)

    rank_scale = alpha * n_anomaly / m
    rank_delta = p_floor_proxy / rank_scale if rank_scale > 0.0 else math.inf
    return {
        "dataset": row.get("dataset", ""),
        "severity": float(row.get("severity", 0.0)),
        "approach": row.get("approach", ""),
        "alpha": alpha,
        "m": m,
        "actual_anomaly_rate": float(row.get("actual_anomaly_rate", n_anomaly / m)),
        "power": float(row.get("power", 0.0)),
        "fdr": float(row.get("fdr", 0.0)),
        "calib_neff": calib_neff,
        "p_floor_proxy": p_floor_proxy,
        "log10_m_over_alpha": math.log10(m / alpha),
        "log10_inverse_p_floor_proxy": math.log10(1.0 / p_floor_proxy),
        "approx_log10_rank_delta": math.log10(rank_delta),
        "certified_proxy": bool(rank_delta > 1.0),
    }


def build_empirical_projection_summary() -> pd.DataFrame:
    rows = []
    if EMPIRICAL_RESULTS_DIR.exists():
        alpha = load_empirical_alpha(EMPIRICAL_RESULTS_DIR)
        for path in sorted(EMPIRICAL_RESULTS_DIR.glob("*.csv")):
            if path.name == "config.csv":
                continue
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            for _, row in frame.iterrows():
                diagnostic = empirical_row_diagnostic(row, alpha)
                if diagnostic is not None:
                    rows.append(diagnostic)

    if not rows:
        summary = pd.DataFrame(columns=EMPIRICAL_COLUMNS)
        summary.to_csv(EMPIRICAL_PROJECTION_SUMMARY_PATH, index=False)
        return summary

    raw = pd.DataFrame(rows)
    grouped = (
        raw.groupby(["dataset", "severity", "approach"], dropna=False)
        .agg(
            alpha=("alpha", "first"),
            m_mean=("m", "mean"),
            actual_anomaly_rate_mean=("actual_anomaly_rate", "mean"),
            power_mean=("power", "mean"),
            fdr_mean=("fdr", "mean"),
            calib_neff_mean=("calib_neff", "mean"),
            p_floor_proxy_mean=("p_floor_proxy", "mean"),
            log10_m_over_alpha=("log10_m_over_alpha", "mean"),
            log10_inverse_p_floor_proxy=("log10_inverse_p_floor_proxy", "mean"),
            approx_log10_rank_delta=("approx_log10_rank_delta", "mean"),
            certified_proxy_rate=("certified_proxy", "mean"),
            count=("power", "count"),
        )
        .reset_index()
    )
    grouped.insert(0, "summary_version", EMPIRICAL_PROJECTION_VERSION)
    grouped[EMPIRICAL_COLUMNS].to_csv(EMPIRICAL_PROJECTION_SUMMARY_PATH, index=False)
    return grouped[EMPIRICAL_COLUMNS]


def load_or_build_empirical_projection_summary(force: bool) -> pd.DataFrame:
    if (
        not force
        and summary_is_current(
            EMPIRICAL_PROJECTION_SUMMARY_PATH,
            EMPIRICAL_PROJECTION_VERSION,
        )
    ):
        print("loading existing Figure 5 empirical projection summary", flush=True)
        return pd.read_csv(EMPIRICAL_PROJECTION_SUMMARY_PATH)
    return build_empirical_projection_summary()


def plot_empirical_projection(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    if summary.empty:
        ax.text(
            0.5,
            0.5,
            "No empirical benchmark CSVs found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(EMPIRICAL_PROJECTION_FIGURE_PATH, bbox_inches="tight", dpi=220)
        plt.close(fig)
        return

    severities = sorted(summary["severity"].unique())
    markers = ["o", "s", "^", "D", "P", "X"]
    scatter = None
    for marker, severity in zip(markers, severities, strict=False):
        block = summary[summary["severity"].eq(severity)]
        scatter = ax.scatter(
            block["log10_m_over_alpha"],
            block["log10_inverse_p_floor_proxy"],
            c=block["power_mean"],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=38,
            marker=marker,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.9,
            label=f"severity={severity:g}",
        )
    lower = min(
        float(summary["log10_m_over_alpha"].min()),
        float(summary["log10_inverse_p_floor_proxy"].min()),
    )
    upper = max(
        float(summary["log10_m_over_alpha"].max()),
        float(summary["log10_inverse_p_floor_proxy"].max()),
    )
    padding = max(0.1 * (upper - lower), 0.1)
    line = np.array([lower - padding, upper + padding])
    ax.plot(line, line, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(line[0], line[1])
    ax.set_ylim(line[0], line[1])
    ax.set_xlabel(r"$\log_{10}(m/\alpha)$")
    ax.set_ylabel(r"$\log_{10}(1 / p_{\min}^{proxy})$")
    ax.set_title("Empirical benchmark projection onto the resolution plane")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("mean empirical power")
    fig.savefig(EMPIRICAL_PROJECTION_FIGURE_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def validate_outputs() -> None:
    for path in [
        POWER_ATLAS_FIGURE_PATH,
        POWER_ATLAS_SUMMARY_PATH,
        POWER_ATLAS_TIKZ_PATH,
        POWER_ATLAS_REFERENCE_TIKZ_PATH,
        REQUIRED_ESS_FIGURE_PATH,
        REQUIRED_ESS_SUMMARY_PATH,
        REQUIRED_ESS_TIKZ_PATH,
        EMPIRICAL_PROJECTION_FIGURE_PATH,
        EMPIRICAL_PROJECTION_SUMMARY_PATH,
        EMPIRICAL_PROJECTION_TIKZ_PATH,
    ]:
        if not path.exists():
            raise RuntimeError(f"Missing output: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of worker processes for simulation cells.",
    )
    parser.add_argument(
        "--wcs-batch-size",
        type=int,
        default=DEFAULT_WCS_BATCH_SIZE,
        help="Candidate rows per WCS auxiliary-p-value batch.",
    )
    parser.add_argument(
        "--cell-trials",
        type=int,
        default=DEFAULT_CELL_TRIALS,
        help="Accepted Monte Carlo trials per designed cell.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild summaries even when matching cached CSVs exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    workers = max(1, int(args.workers))
    wcs_batch_size = max(1, int(args.wcs_batch_size))
    cell_trials = max(1, int(args.cell_trials))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )
    power_summary = load_or_build_power_atlas_summary(
        workers=workers,
        cell_trials=cell_trials,
        wcs_batch_size=wcs_batch_size,
        force=bool(args.force),
    )
    required_summary = load_or_build_required_ess_summary(
        workers=workers,
        cell_trials=cell_trials,
        wcs_batch_size=wcs_batch_size,
        force=bool(args.force),
    )
    empirical_summary = load_or_build_empirical_projection_summary(force=bool(args.force))

    validate_power_atlas_summary(power_summary)
    validate_required_ess_summary(required_summary)
    plot_power_atlas(power_summary)
    plot_required_ess(required_summary)
    plot_empirical_projection(empirical_summary)
    export_tikz_csvs(power_summary, required_summary, empirical_summary)
    validate_outputs()

    print(POWER_ATLAS_SUMMARY_PATH)
    print(POWER_ATLAS_TIKZ_PATH)
    print(POWER_ATLAS_REFERENCE_TIKZ_PATH)
    print(POWER_ATLAS_FIGURE_PATH)
    print(REQUIRED_ESS_SUMMARY_PATH)
    print(REQUIRED_ESS_TIKZ_PATH)
    print(REQUIRED_ESS_FIGURE_PATH)
    print(EMPIRICAL_PROJECTION_SUMMARY_PATH)
    print(EMPIRICAL_PROJECTION_TIKZ_PATH)
    print(EMPIRICAL_PROJECTION_FIGURE_PATH)


if __name__ == "__main__":
    main()
