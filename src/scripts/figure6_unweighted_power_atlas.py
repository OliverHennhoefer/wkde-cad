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
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "figure6"
POWER_ATLAS_FIGURE_PATH = OUT_DIR / "figure6_unweighted_power_atlas.png"
POWER_ATLAS_SUMMARY_PATH = OUT_DIR / "figure6_unweighted_power_atlas_summary.csv"
POWER_ATLAS_TIKZ_PATH = OUT_DIR / "figure6_unweighted_power_atlas_tikz.csv"
POWER_ATLAS_REFERENCE_TIKZ_PATH = (
    OUT_DIR / "figure6_unweighted_power_atlas_reference_tikz.csv"
)
FDR_RATIO_ATLAS_FIGURE_PATH = OUT_DIR / "figure6_supp_fdr_ratio_atlas.png"
FDR_RATIO_ATLAS_TIKZ_PATH = OUT_DIR / "figure6_supp_fdr_ratio_atlas_tikz.csv"
FDR_RATIO_ATLAS_REFERENCE_TIKZ_PATH = (
    OUT_DIR / "figure6_supp_fdr_ratio_atlas_reference_tikz.csv"
)

SUMMARY_VERSION = "standard-unweighted-power-atlas-v4"
TIKZ_EXPORT_VERSION = "tikz-v1"
BASE_SEED = 20260603

DELTA_SCORE = 1.0
N_EFF_BINS = np.linspace(1.3, 4.1, 15)
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
DEFAULT_CALIBRATION_REPEATS = 25
DEFAULT_TEST_REPEATS = 100
SCORE_REGIMES = ("perfect", "finite_k1", "finite_k2", "finite_k3")
FINITE_KAPPAS = {
    "finite_k1": 1.0,
    "finite_k2": 2.0,
    "finite_k3": 3.0,
}


@dataclass(frozen=True)
class CellTask:
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
    calibration_repeats: int
    test_repeats: int


def rng_for(*values: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *values]))


def ordinary_conformal_p_values(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
) -> np.ndarray:
    sorted_calibration = np.sort(np.asarray(calibration_scores, dtype=float), kind="mergesort")
    return ordinary_conformal_p_values_from_sorted(sorted_calibration, test_scores)


def ordinary_conformal_p_values_from_sorted(
    sorted_calibration: np.ndarray,
    test_scores: np.ndarray,
) -> np.ndarray:
    sorted_calibration = np.asarray(sorted_calibration, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    tail_start = np.searchsorted(sorted_calibration, test_scores, side="left")
    tail_count = len(sorted_calibration) - tail_start
    return (1.0 + tail_count) / (len(sorted_calibration) + 1.0)


def bh_decisions(p_values: np.ndarray, alpha: float) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p_values, kind="mergesort")
    sorted_p = p_values[order]
    thresholds = alpha * np.arange(1, len(sorted_p) + 1) / len(sorted_p)
    passed = sorted_p <= thresholds
    decisions = np.zeros(len(sorted_p), dtype=bool)
    if not np.any(passed):
        return decisions
    cutoff = sorted_p[int(np.flatnonzero(passed)[-1])]
    decisions[p_values <= cutoff] = True
    return decisions


def bh_decisions_by_row(p_values: np.ndarray, alpha: float) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 2:
        raise ValueError("p_values must be a two-dimensional array.")
    if p_values.shape[1] == 0:
        return np.zeros_like(p_values, dtype=bool)

    order = np.argsort(p_values, axis=1, kind="mergesort")
    sorted_p = np.take_along_axis(p_values, order, axis=1)
    thresholds = alpha * np.arange(1, p_values.shape[1] + 1) / p_values.shape[1]
    passed = sorted_p <= thresholds
    passed_any = np.any(passed, axis=1)
    last_passed = p_values.shape[1] - 1 - np.argmax(passed[:, ::-1], axis=1)
    cutoffs = np.where(
        passed_any,
        sorted_p[np.arange(p_values.shape[0]), last_passed],
        -np.inf,
    )
    return p_values <= cutoffs[:, None]


def empirical_auc(anomaly_scores: np.ndarray, inlier_scores: np.ndarray) -> float:
    sorted_inliers = np.sort(inlier_scores, kind="mergesort")
    less = np.searchsorted(sorted_inliers, anomaly_scores, side="left")
    less_equal = np.searchsorted(sorted_inliers, anomaly_scores, side="right")
    ties = less_equal - less
    return float(np.mean((less + 0.5 * ties) / len(sorted_inliers)))


def empirical_auc_by_row(
    anomaly_scores: np.ndarray,
    inlier_scores: np.ndarray,
) -> np.ndarray:
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)
    inlier_scores = np.asarray(inlier_scores, dtype=float)
    if anomaly_scores.ndim != 2 or inlier_scores.ndim != 2:
        raise ValueError("Score arrays must be two-dimensional.")
    if anomaly_scores.shape[0] != inlier_scores.shape[0]:
        raise ValueError("Score arrays must have the same number of rows.")

    n_rows, n_anomaly = anomaly_scores.shape
    n_inlier = inlier_scores.shape[1]
    combined = np.concatenate([inlier_scores, anomaly_scores], axis=1)
    order = np.argsort(combined, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n_rows)[:, None], order] = np.arange(
        1,
        combined.shape[1] + 1,
        dtype=float,
    )
    anomaly_ranks = ranks[:, n_inlier:]
    rank_sum_null = n_anomaly * (n_anomaly + 1) / 2.0
    return (np.sum(anomaly_ranks, axis=1) - rank_sum_null) / (n_anomaly * n_inlier)


def discovery_metrics(decisions: np.ndarray, y_true: np.ndarray) -> tuple[bool, float, float]:
    n_rejections = int(np.sum(decisions))
    true_rejections = int(np.sum(decisions & y_true))
    false_rejections = n_rejections - true_rejections
    n_anomaly = int(np.sum(y_true))
    power = true_rejections / max(1, n_anomaly)
    fdr = false_rejections / n_rejections if n_rejections else 0.0
    return bool(n_rejections > 0), float(power), float(fdr)


def discovery_metrics_by_row(
    decisions: np.ndarray,
    *,
    n_inlier: int,
    n_anomaly: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decisions = np.asarray(decisions, dtype=bool)
    n_rejections = np.sum(decisions, axis=1)
    true_rejections = np.sum(decisions[:, n_inlier:], axis=1)
    false_rejections = np.sum(decisions[:, :n_inlier], axis=1)
    power = true_rejections / max(1, n_anomaly)
    fdr = np.divide(
        false_rejections,
        n_rejections,
        out=np.zeros_like(false_rejections, dtype=float),
        where=n_rejections > 0,
    )
    return n_rejections > 0, power.astype(float), fdr.astype(float)


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
            "title": r"$N_{cal}$ by $m$; fixed $\pi_1=0.05$",
            "x_label": r"test batch size $m$",
            "tick_values": [float(value) for value in M_VALUES],
            "x_plots": [math.log10(float(value)) for value in M_VALUES],
            "values": [float(value) for value in M_VALUES],
        },
        {
            "name": "m_fixed_count",
            "title": r"$N_{cal}$ by $m$; fixed 10 anomalies",
            "x_label": r"test batch size $m$",
            "tick_values": [float(value) for value in M_VALUES],
            "x_plots": [math.log10(float(value)) for value in M_VALUES],
            "values": [float(value) for value in M_VALUES],
        },
        {
            "name": "pi1_sweep",
            "title": r"$N_{cal}$ by anomaly rate",
            "x_label": r"anomaly rate $\pi_1$",
            "tick_values": [float(value) for value in PI1_VALUES],
            "x_plots": [math.log10(float(value)) for value in PI1_VALUES],
            "values": [float(value) for value in PI1_VALUES],
        },
        {
            "name": "alpha_sweep",
            "title": r"$N_{cal}$ by FDR level",
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


def n_anomaly_for(m: int, pi1: float | None, n_anomaly_fixed: int | None) -> int:
    if n_anomaly_fixed is not None:
        value = int(n_anomaly_fixed)
    else:
        value = int(round(float(pi1) * m))
    return min(max(1, value), max(1, m - 1))


def n_cal_for_bin(y_left: float, y_right: float) -> int:
    return max(2, int(round(10 ** ((float(y_left) + float(y_right)) / 2.0))))


def score_label(score_regime: str) -> str:
    if score_regime == "perfect":
        return "perfect score"
    if score_regime in FINITE_KAPPAS:
        return rf"finite score ($K={FINITE_KAPPAS[score_regime]:g}$)"
    raise ValueError(f"Unknown score regime: {score_regime}")


def simulate_calibration_block(
    *,
    alpha: float,
    m: int,
    pi1: float | None,
    n_anomaly_fixed: int | None,
    n_cal: int,
    seed_prefix: int,
    calibration_index: int,
    test_repeats: int,
) -> list[dict[str, Any]]:
    calibration_rng = rng_for(n_cal, m, seed_prefix, calibration_index, 0)
    calibration_scores = calibration_rng.normal(0.0, 1.0, n_cal)
    sorted_calibration = np.sort(calibration_scores, kind="mergesort")
    max_calibration_score = float(np.max(calibration_scores))

    n_anomaly = n_anomaly_for(m, pi1, n_anomaly_fixed)
    n_inlier = m - n_anomaly
    test_rng = rng_for(n_cal, m, seed_prefix, calibration_index, 1)
    inlier_scores = test_rng.normal(0.0, 1.0, (test_repeats, n_inlier))
    anomaly_noise = test_rng.normal(0.0, 1.0, (test_repeats, n_anomaly))

    p_min = 1.0 / (n_cal + 1.0)
    ranks = np.arange(1, n_anomaly + 1)
    rank_delta = float(np.min(p_min / (alpha * ranks / m)))
    log10_rank_delta = float(np.log10(rank_delta))

    metrics_by_regime: dict[str, dict[str, np.ndarray]] = {}
    for regime in SCORE_REGIMES:
        if regime == "perfect":
            anomaly_score_value = (
                np.maximum(max_calibration_score, np.max(inlier_scores, axis=1))
                + DELTA_SCORE
            )
            anomaly_scores = np.repeat(
                anomaly_score_value[:, None],
                n_anomaly,
                axis=1,
            )
        else:
            anomaly_scores = FINITE_KAPPAS[regime] + anomaly_noise

        test_scores = np.concatenate([inlier_scores, anomaly_scores], axis=1)
        p_values = ordinary_conformal_p_values_from_sorted(
            sorted_calibration,
            test_scores,
        )
        decisions = bh_decisions_by_row(p_values, alpha)
        any_discovery, power, fdr = discovery_metrics_by_row(
            decisions,
            n_inlier=n_inlier,
            n_anomaly=n_anomaly,
        )
        metrics_by_regime[regime] = {
            "any_discovery": any_discovery,
            "power": power,
            "fdr": fdr,
            "auroc": empirical_auc_by_row(anomaly_scores, inlier_scores),
            "perfect_separation_from_calibration": np.min(anomaly_scores, axis=1)
            > max_calibration_score,
        }

    return [
        {
            "n_cal": n_cal,
            "n_anomaly": n_anomaly,
            "actual_pi1": n_anomaly / m,
            "rank_delta": rank_delta,
            "log10_rank_delta": log10_rank_delta,
            "certified_no_rank_discovery": bool(rank_delta > 1.0),
            "calibration_repeat": calibration_index,
            "test_repeat": test_index,
            "metrics": {
                regime: {
                    "any_discovery": bool(
                        metrics_by_regime[regime]["any_discovery"][test_index]
                    ),
                    "power": float(metrics_by_regime[regime]["power"][test_index]),
                    "fdr": float(metrics_by_regime[regime]["fdr"][test_index]),
                    "auroc": float(metrics_by_regime[regime]["auroc"][test_index]),
                    "perfect_separation_from_calibration": bool(
                        metrics_by_regime[regime][
                            "perfect_separation_from_calibration"
                        ][test_index]
                    ),
                }
                for regime in SCORE_REGIMES
            },
        }
        for test_index in range(test_repeats)
    ]


def simulate_trial(
    *,
    alpha: float,
    m: int,
    pi1: float | None,
    n_anomaly_fixed: int | None,
    n_cal: int,
    seed: int,
) -> dict[str, Any]:
    rows = simulate_calibration_block(
        alpha=alpha,
        m=m,
        pi1=pi1,
        n_anomaly_fixed=n_anomaly_fixed,
        n_cal=n_cal,
        seed_prefix=seed,
        calibration_index=0,
        test_repeats=1,
    )
    return rows[0]


def aggregate_cell(task: CellTask, trial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for regime in SCORE_REGIMES:
        count = len(trial_rows)
        metrics = [row["metrics"][regime] for row in trial_rows]
        mean_n_cal = float(np.mean([row["n_cal"] for row in trial_rows]))
        rows.append(
            {
                "summary_version": SUMMARY_VERSION,
                "collection": "atlas",
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
                "calibration_repeats": int(task.calibration_repeats),
                "test_repeats_per_calibration": int(task.test_repeats),
                "count": count,
                "mean_n_cal": mean_n_cal,
                "mean_rho": 0.0,
                "mean_neff": mean_n_cal,
                "log10_mean_neff": float(np.log10(mean_n_cal)),
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
    n_cal = n_cal_for_bin(task.y_left, task.y_right)
    seed_prefix = 100 * sweep_code(task.sweep) + task.x_index + 1000 * task.y_bin
    trial_rows: list[dict[str, Any]] = []
    for calibration_index in range(int(task.calibration_repeats)):
        trial_rows.extend(
            simulate_calibration_block(
                alpha=task.alpha,
                m=task.m,
                pi1=task.pi1,
                n_anomaly_fixed=task.n_anomaly_fixed,
                n_cal=n_cal,
                seed_prefix=seed_prefix,
                calibration_index=calibration_index,
                test_repeats=int(task.test_repeats),
            )
        )
    return aggregate_cell(task, trial_rows)


def sweep_code(sweep: str) -> int:
    order = {
        "m_fixed_pi": 1,
        "m_fixed_count": 2,
        "pi1_sweep": 3,
        "alpha_sweep": 4,
    }
    return order[sweep]


def atlas_tasks(calibration_repeats: int, test_repeats: int) -> list[CellTask]:
    tasks = []
    for spec in sweep_specs():
        x_edges = edges_from_centers(spec["x_plots"])
        for x_index, value in enumerate(spec["values"]):
            alpha, m, pi1, n_anomaly_fixed = params_for_sweep(spec["name"], value)
            for y_bin in range(len(N_EFF_BINS) - 1):
                tasks.append(
                    CellTask(
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
                        calibration_repeats=calibration_repeats,
                        test_repeats=test_repeats,
                    )
                )
    return tasks


def run_cells(tasks: list[CellTask], workers: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        iterator = map(simulate_cell, tasks)
        for cell_rows in tqdm(iterator, total=len(tasks), desc="Figure 6 cells", unit="cell"):
            rows.extend(cell_rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for cell_rows in tqdm(
                executor.map(simulate_cell, tasks),
                total=len(tasks),
                desc="Figure 6 cells",
                unit="cell",
            ):
                rows.extend(cell_rows)
    return pd.DataFrame(rows)


def summary_is_current(
    path: Path,
    *,
    calibration_repeats: int,
    test_repeats: int,
) -> bool:
    if not path.exists():
        return False
    header = pd.read_csv(path, nrows=1)
    if header.empty or str(header["summary_version"].iloc[0]) != SUMMARY_VERSION:
        return False
    if "calibration_repeats" not in header or "test_repeats_per_calibration" not in header:
        return False
    return bool(
        int(header["calibration_repeats"].iloc[0]) == int(calibration_repeats)
        and int(header["test_repeats_per_calibration"].iloc[0]) == int(test_repeats)
    )


def build_power_atlas_summary(
    *,
    workers: int,
    calibration_repeats: int,
    test_repeats: int,
) -> pd.DataFrame:
    summary = run_cells(atlas_tasks(calibration_repeats, test_repeats), workers)
    validate_power_atlas_summary(summary)
    summary.to_csv(POWER_ATLAS_SUMMARY_PATH, index=False)
    return summary


def load_or_build_power_atlas_summary(
    *,
    workers: int,
    calibration_repeats: int,
    test_repeats: int,
    force: bool,
) -> pd.DataFrame:
    if not force and summary_is_current(
        POWER_ATLAS_SUMMARY_PATH,
        calibration_repeats=calibration_repeats,
        test_repeats=test_repeats,
    ):
        print("loading existing Figure 6 power-atlas summary", flush=True)
        return pd.read_csv(POWER_ATLAS_SUMMARY_PATH)
    return build_power_atlas_summary(
        workers=workers,
        calibration_repeats=calibration_repeats,
        test_repeats=test_repeats,
    )


def validate_power_atlas_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        raise RuntimeError("Figure 6 power summary is empty.")
    if set(summary["collection"]) != {"atlas"}:
        raise RuntimeError("Figure 6 summary contains an unexpected collection.")
    if (summary["count"] <= 0).any():
        raise RuntimeError("Figure 6 summary contains zero-count cells.")
    repeat_columns = {"calibration_repeats", "test_repeats_per_calibration"}
    if not repeat_columns.issubset(summary.columns):
        raise RuntimeError("Figure 6 summary must record nested Monte Carlo repeats.")
    expected_count = (
        summary["calibration_repeats"].astype(int)
        * summary["test_repeats_per_calibration"].astype(int)
    )
    if not (summary["count"].astype(int) == expected_count).all():
        raise RuntimeError("Figure 6 summary count does not match nested repeats.")
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
    if not np.allclose(summary["mean_neff"], summary["mean_n_cal"]):
        raise RuntimeError("Unweighted Figure 6 must have N_eff equal to N_cal.")
    if not np.allclose(summary["mean_rho"], 0.0):
        raise RuntimeError("Unweighted Figure 6 must not include covariate shift.")
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
            f"Figure 6 atlas cells incomplete: missing={len(expected - observed)}, "
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


def with_fdr_ratio(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["fdr_ratio"] = result["fdr"].astype(float) / result["alpha"].astype(float)
    return result


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
        ax.set_ylabel(r"$\log_{10}$ calibration $N_{cal}$ $(=N_{eff})$")
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
    fig.suptitle("Operational power atlas for standard conformal CAD", fontsize=14)
    fig.savefig(POWER_ATLAS_FIGURE_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_fdr_ratio_atlas(summary: pd.DataFrame) -> None:
    summary = with_fdr_ratio(summary)
    specs = sweep_specs()
    fig, axes = plt.subplots(
        len(SCORE_REGIMES),
        len(specs),
        figsize=(4.2 * len(specs), 4.2 * len(SCORE_REGIMES)),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(len(SCORE_REGIMES), len(specs))
    cmap = plt.get_cmap("coolwarm").copy()
    mesh = None
    for col_idx, spec in enumerate(specs):
        x_tick_positions = [float(value) for value in spec["x_plots"]]
        x_tick_labels = [format_tick(float(value)) for value in spec["tick_values"]]
        for row_idx, score_regime in enumerate(SCORE_REGIMES):
            ax = axes[row_idx, col_idx]
            x_edges, y_edges, fdr_ratio = heatmap_matrix(
                summary,
                sweep=spec["name"],
                score_regime=score_regime,
                value_column="fdr_ratio",
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
                fdr_ratio.T,
                cmap=cmap,
                norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0),
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
        ax.set_ylabel(r"$\log_{10}$ calibration $N_{cal}$ $(=N_{eff})$")
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

    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.92, extend="max")
    cbar.set_label(r"empirical FDR / nominal $\alpha$")
    fig.suptitle(
        "Supplementary FDR-ratio atlas for standard conformal CAD",
        fontsize=14,
    )
    fig.savefig(FDR_RATIO_ATLAS_FIGURE_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def export_power_atlas_tikz(summary: pd.DataFrame) -> None:
    specs = sweep_specs()
    spec_by_name = {str(spec["name"]): spec for spec in specs}
    sweep_order = {str(spec["name"]): idx for idx, spec in enumerate(specs)}
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
                "figure": "figure6",
                "panel": f"power_atlas_r{panel_row}_c{panel_col}",
                "panel_title": spec["title"],
                "plot_order": (panel_row - 1) * len(specs) + panel_col,
                "group": "heatmap_cell",
                "method": "standard_conformal_bh",
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
                "calibration_repeats": int(row.calibration_repeats),
                "test_repeats_per_calibration": int(
                    row.test_repeats_per_calibration
                ),
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
    specs = sweep_specs()
    rows = []
    for spec in specs:
        panel_col = [item["name"] for item in specs].index(spec["name"]) + 1
        for score_regime in SCORE_REGIMES:
            panel_row = SCORE_REGIMES.index(score_regime) + 1
            panel = f"power_atlas_r{panel_row}_c{panel_col}"
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure6",
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


def export_fdr_ratio_atlas_tikz(summary: pd.DataFrame) -> None:
    summary = with_fdr_ratio(summary)
    specs = sweep_specs()
    spec_by_name = {str(spec["name"]): spec for spec in specs}
    sweep_order = {str(spec["name"]): idx for idx, spec in enumerate(specs)}
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
                "figure": "figure6_supp",
                "panel": f"fdr_ratio_atlas_r{panel_row}_c{panel_col}",
                "panel_title": spec["title"],
                "plot_order": (panel_row - 1) * len(specs) + panel_col,
                "group": "heatmap_cell",
                "method": "standard_conformal_bh",
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
                "calibration_repeats": int(row.calibration_repeats),
                "test_repeats_per_calibration": int(
                    row.test_repeats_per_calibration
                ),
                "value": float(row.fdr_ratio),
                "fdr_ratio": float(row.fdr_ratio),
                "power": float(row.power),
                "discovery_probability": float(row.discovery_probability),
                "fdr": float(row.fdr),
                "median_log10_rank_delta": float(row.median_log10_rank_delta),
                "certified_no_rank_rate": float(row.certified_no_rank_rate),
                "mean_auroc": float(row.mean_auroc),
                "count": int(row.count),
                "style_key": f"{row.score_regime}_{row.sweep}",
                "label": "empirical FDR / nominal alpha",
            }
        )
    pd.DataFrame(rows).to_csv(FDR_RATIO_ATLAS_TIKZ_PATH, index=False)


def export_fdr_ratio_atlas_reference_tikz() -> None:
    specs = sweep_specs()
    rows = []
    for spec in specs:
        panel_col = [item["name"] for item in specs].index(spec["name"]) + 1
        for score_regime in SCORE_REGIMES:
            panel_row = SCORE_REGIMES.index(score_regime) + 1
            panel = f"fdr_ratio_atlas_r{panel_row}_c{panel_col}"
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure6_supp",
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
    pd.DataFrame(rows).to_csv(FDR_RATIO_ATLAS_REFERENCE_TIKZ_PATH, index=False)


def validate_outputs() -> None:
    for path in [
        POWER_ATLAS_FIGURE_PATH,
        POWER_ATLAS_SUMMARY_PATH,
        POWER_ATLAS_TIKZ_PATH,
        POWER_ATLAS_REFERENCE_TIKZ_PATH,
        FDR_RATIO_ATLAS_FIGURE_PATH,
        FDR_RATIO_ATLAS_TIKZ_PATH,
        FDR_RATIO_ATLAS_REFERENCE_TIKZ_PATH,
    ]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected Figure 6 output was not written: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Figure 6 standard conformal CAD power atlas.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of worker processes for atlas simulation.",
    )
    parser.add_argument(
        "--calibration-repeats",
        type=int,
        default=DEFAULT_CALIBRATION_REPEATS,
        help="Independent split-conformal calibration sets per atlas cell.",
    )
    parser.add_argument(
        "--test-repeats",
        type=int,
        default=DEFAULT_TEST_REPEATS,
        help="Independent test batches evaluated against each fixed calibration set.",
    )
    parser.add_argument(
        "--cell-trials",
        type=int,
        default=None,
        help=(
            "Legacy/debug shortcut: use one calibration set per cell and this many "
            "test batches."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached summaries before plotting/exporting.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cell_trials is None:
        calibration_repeats = max(1, int(args.calibration_repeats))
        test_repeats = max(1, int(args.test_repeats))
    else:
        calibration_repeats = 1
        test_repeats = max(1, int(args.cell_trials))
    power_summary = load_or_build_power_atlas_summary(
        workers=max(1, int(args.workers)),
        calibration_repeats=calibration_repeats,
        test_repeats=test_repeats,
        force=bool(args.force),
    )
    validate_power_atlas_summary(power_summary)
    plot_power_atlas(power_summary)
    plot_fdr_ratio_atlas(power_summary)
    export_power_atlas_tikz(power_summary)
    export_power_atlas_reference_tikz()
    export_fdr_ratio_atlas_tikz(power_summary)
    export_fdr_ratio_atlas_reference_tikz()
    validate_outputs()
    print("Figure 6 outputs written:", flush=True)
    print(POWER_ATLAS_SUMMARY_PATH)
    print(POWER_ATLAS_TIKZ_PATH)
    print(POWER_ATLAS_REFERENCE_TIKZ_PATH)
    print(POWER_ATLAS_FIGURE_PATH)
    print(FDR_RATIO_ATLAS_TIKZ_PATH)
    print(FDR_RATIO_ATLAS_REFERENCE_TIKZ_PATH)
    print(FDR_RATIO_ATLAS_FIGURE_PATH)


if __name__ == "__main__":
    main()
