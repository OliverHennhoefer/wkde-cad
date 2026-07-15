from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "outputs" / "simulations" / Path(__file__).stem
POWER_ATLAS_FIGURE_PATH = OUT_DIR / "figure2_weighted_power_atlas.png"
POWER_ATLAS_SUMMARY_PATH = OUT_DIR / "figure2_weighted_power_atlas_summary.csv"
POWER_ATLAS_TIKZ_PATH = OUT_DIR / "figure2_weighted_power_atlas_tikz.csv"
POWER_ATLAS_REFERENCE_TIKZ_PATH = (
    OUT_DIR / "figure2_weighted_power_atlas_reference_tikz.csv"
)

SUMMARY_VERSION = "operational-envelope-v4"
TIKZ_EXPORT_VERSION = "tikz-v1"
BASE_SEED = 20260518

D = 10
FINITE_KAPPAS = {
    "finite_k1": 1.0,
    "finite_k2": 2.0,
    "finite_k3": 3.0,
}
DELTA_SCORE = 1.0
N_EFF_BINS = np.linspace(1.3, 4.1, 15)
RHO_CANDIDATES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
N_CAL_MAX = 16000
MAX_ACCEPT_ATTEMPTS_PER_CELL = 5000

M_VALUES = [50, 100, 200, 500, 1000, 2000]
PI1_VALUES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
ALPHA_VALUES = [0.01, 0.025, 0.05, 0.10, 0.20]

BASELINE_ALPHA = 0.10
BASELINE_PI1 = 0.05
BASELINE_M = 1000
BASELINE_N_ANOMALY = 10
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
        "rank_delta_above_one": bool(rank_delta > 1.0),
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
                "rank_delta_above_one_rate": float(
                    np.mean([row["rank_delta_above_one"] for row in trial_rows])
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
            "Could not fill Figure 2 cell: "
            f"collection={task.collection}, sweep={task.sweep}, "
            f"x_index={task.x_index}, y_bin={task.y_bin}, "
            f"accepted={len(trial_rows)}, attempts={attempts}."
        )
    return aggregate_cell(task, trial_rows)


def sweep_code(sweep: str) -> int:
    names = [spec["name"] for spec in sweep_specs()]
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


def summary_is_current(path: Path, version: str, cell_trials: int) -> bool:
    if not path.exists():
        return False
    try:
        metadata = pd.read_csv(path, usecols=["summary_version", "count"])
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return (
        not metadata.empty
        and set(metadata["summary_version"].astype(str)) == {version}
        and (pd.to_numeric(metadata["count"], errors="coerce") == cell_trials).all()
    )


def build_power_atlas_summary(
    *, workers: int, cell_trials: int, wcs_batch_size: int
) -> pd.DataFrame:
    summary = run_cells(
        atlas_tasks(cell_trials, wcs_batch_size),
        workers,
        "Figure 2 weighted power-atlas cells",
    )
    validate_power_atlas_summary(summary)
    summary.to_csv(POWER_ATLAS_SUMMARY_PATH, index=False)
    return summary


def load_or_build_power_atlas_summary(
    *, workers: int, cell_trials: int, wcs_batch_size: int, force: bool
) -> pd.DataFrame:
    if not force and summary_is_current(
        POWER_ATLAS_SUMMARY_PATH,
        SUMMARY_VERSION,
        cell_trials,
    ):
        print("loading existing Figure 2 weighted power-atlas summary", flush=True)
        return pd.read_csv(POWER_ATLAS_SUMMARY_PATH)
    return build_power_atlas_summary(
        workers=workers,
        cell_trials=cell_trials,
        wcs_batch_size=wcs_batch_size,
    )


def validate_power_atlas_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        raise RuntimeError("Figure 2 weighted power summary is empty.")
    if set(summary["collection"]) != {"atlas"}:
        raise RuntimeError("Figure 2 summary contains an unexpected collection.")
    if (summary["count"] <= 0).any():
        raise RuntimeError("Figure 2 summary contains zero-count cells.")
    probability_columns = [
        "discovery_probability",
        "power",
        "fdr",
        "rank_delta_above_one_rate",
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
            f"Figure 2 atlas cells incomplete: missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}."
        )


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
                "figure": "figure2",
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
                "rank_delta_above_one_rate": float(row.rank_delta_above_one_rate),
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
                    "figure": "figure2",
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


def export_tikz_csvs(power_summary: pd.DataFrame) -> None:
    export_power_atlas_tikz(power_summary)
    export_power_atlas_reference_tikz()


def validate_outputs() -> None:
    for path in [
        POWER_ATLAS_FIGURE_PATH,
        POWER_ATLAS_SUMMARY_PATH,
        POWER_ATLAS_TIKZ_PATH,
        POWER_ATLAS_REFERENCE_TIKZ_PATH,
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
    validate_power_atlas_summary(power_summary)
    plot_power_atlas(power_summary)
    export_tikz_csvs(power_summary)
    validate_outputs()

    print(POWER_ATLAS_SUMMARY_PATH)
    print(POWER_ATLAS_TIKZ_PATH)
    print(POWER_ATLAS_REFERENCE_TIKZ_PATH)
    print(POWER_ATLAS_FIGURE_PATH)


if __name__ == "__main__":
    main()
