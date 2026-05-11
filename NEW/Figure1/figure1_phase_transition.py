from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


OUT_ROOT = Path(__file__).resolve().parent
MODES = ("unweighted", "weighted")
WCS_PRUNING_METHODS = ("deterministic", "homogeneous", "heterogeneous")
DEFAULT_WEIGHTED_PRUNING = "homogeneous"

SUMMARY_VERSION = "split-v4"
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
INCLUDE_SUPPLEMENT_GRID = False
N_SEEDS = 100
BASE_SEED = 20260509
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)
WORKERS = int(os.environ.get("FIGURE1_WORKERS", DEFAULT_WORKERS))
WCS_CANDIDATE_BATCH_SIZE = int(os.environ.get("FIGURE1_WCS_BATCH_SIZE", 512))

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


@dataclass(frozen=True)
class Figure1Paths:
    out_dir: Path
    schematic_path: Path
    heatmap_path: Path
    collapse_path: Path
    heatmap_summary_path: Path
    collapse_summary_path: Path
    schematic_points_tikz_path: Path
    schematic_annotations_tikz_path: Path
    heatmap_tikz_path: Path
    heatmap_boundary_tikz_path: Path
    collapse_tikz_path: Path
    collapse_reference_tikz_path: Path


def paths_for_mode(mode: str) -> Figure1Paths:
    mode = normalize_mode(mode)
    out_dir = OUT_ROOT / mode
    return Figure1Paths(
        out_dir=out_dir,
        schematic_path=out_dir / "figure1_panel_a_schematic.png",
        heatmap_path=out_dir / "figure1_heatmaps_alpha_pi_sensitivity.png",
        collapse_path=out_dir / "figure1_collapse_diagnostics.png",
        heatmap_summary_path=out_dir / "figure1_heatmap_summary.csv",
        collapse_summary_path=out_dir / "figure1_collapse_summary.csv",
        schematic_points_tikz_path=out_dir / "figure1_schematic_points_tikz.csv",
        schematic_annotations_tikz_path=out_dir / "figure1_schematic_annotations_tikz.csv",
        heatmap_tikz_path=out_dir / "figure1_heatmap_tikz.csv",
        heatmap_boundary_tikz_path=out_dir / "figure1_heatmap_boundary_tikz.csv",
        collapse_tikz_path=out_dir / "figure1_collapse_tikz.csv",
        collapse_reference_tikz_path=out_dir / "figure1_collapse_reference_tikz.csv",
    )


def normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
    return normalized


def normalize_pruning(pruning: str) -> str:
    normalized = str(pruning).strip().lower()
    aliases = {
        "dtm": "deterministic",
        "dete": "deterministic",
        "deterministic": "deterministic",
        "homo": "homogeneous",
        "homogeneous": "homogeneous",
        "hete": "heterogeneous",
        "heterogeneous": "heterogeneous",
    }
    if normalized not in aliases:
        raise ValueError(
            f"pruning must be one of {WCS_PRUNING_METHODS}, got {pruning!r}."
        )
    return aliases[normalized]


def summary_version_for(mode: str, pruning: str) -> str:
    mode = normalize_mode(mode)
    pruning = normalize_pruning(pruning)
    suffix = "bh" if mode == "unweighted" else f"wcs-{pruning}"
    return f"{SUMMARY_VERSION}:{mode}:{suffix}"


def method_key(mode: str, pruning: str) -> str:
    mode = normalize_mode(mode)
    pruning = normalize_pruning(pruning)
    return "bh" if mode == "unweighted" else f"wcs_{pruning}"


def method_label(mode: str, pruning: str) -> str:
    mode = normalize_mode(mode)
    pruning = normalize_pruning(pruning)
    if mode == "unweighted":
        return "BH"
    return f"WCS.{pruning[:4] if pruning != 'deterministic' else 'dtm'}"


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


def standard_tail_p_values(
    sorted_calib_scores: np.ndarray,
    test_scores: np.ndarray,
) -> np.ndarray:
    tail_start = np.searchsorted(sorted_calib_scores, test_scores, side="left")
    tail_count = len(sorted_calib_scores) - tail_start
    return (1.0 + tail_count) / (1.0 + len(sorted_calib_scores))


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
) -> np.ndarray:
    """BH rejection sizes for WCS rows following conformal-selection weighted_CS."""
    rejection_sizes = np.empty(len(candidate_idx), dtype=int)
    batch_size = max(1, int(WCS_CANDIDATE_BATCH_SIZE))
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


def accelerated_wcs_decisions(
    p_values: np.ndarray,
    test_scores: np.ndarray,
    sorted_calib_scores: np.ndarray,
    sorted_calib_weights: np.ndarray,
    total_calib_weight: float,
    test_weights: np.ndarray,
    alpha: float,
    *,
    pruning: str = DEFAULT_WEIGHTED_PRUNING,
    seed: int | None = None,
) -> np.ndarray:
    """Exact WCS decisions with candidate-gated vectorization.

    A hypothesis with p_j > alpha cannot pass the WCS first step because
    q |R_{j->0}| / m <= q. For the remaining candidates, auxiliary p-values use
    the same row formula as conformal-selection's weighted_CS implementation and
    are evaluated in deterministic batches to limit memory pressure.
    """
    pruning = normalize_pruning(pruning)
    p_values = np.asarray(p_values, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    test_weights = np.asarray(test_weights, dtype=float)
    m = len(p_values)
    if m == 0:
        return np.zeros(0, dtype=bool)

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
    )
    first_step_mask = (
        p_values[candidate_idx] <= alpha * candidate_rejection_sizes / m
    )
    first_step_idx = candidate_idx[first_step_mask]
    if len(first_step_idx) == 0:
        return np.zeros(m, dtype=bool)

    selected_sizes = candidate_rejection_sizes[first_step_mask]
    rng = np.random.default_rng(seed)
    if pruning == "heterogeneous":
        metrics = rng.uniform(size=m)[first_step_idx] * selected_sizes
    elif pruning == "homogeneous":
        metrics = rng.uniform() * selected_sizes
    else:
        metrics = selected_sizes.astype(float)

    final_idx = _select_by_pruning_metrics(first_step_idx, metrics)
    decisions = np.zeros(m, dtype=bool)
    decisions[final_idx] = True
    return decisions


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


SimulationTask = tuple[str, str, str, int, float, tuple[int, ...]]


def tasks_for_scenario(mode: str, scenario_name: str) -> list[SimulationTask]:
    mode = normalize_mode(mode)
    base_rho_values = RHO_VALUES if mode == "weighted" else [0.0]
    base_tasks = [
        (mode, "base", scenario_name, n, float(rho), tuple(M_VALUES))
        for n in N_VALUES
        for rho in base_rho_values
    ]

    supplement_tasks = []
    if INCLUDE_SUPPLEMENT_GRID:
        supplement_rho_values = SUPPLEMENT_RHO_VALUES if mode == "weighted" else [0.0]
        # Partial x-axis supplements create unsupported heatmap cells. Keep the
        # default grid rectangular, and require opt-in supplements to cover all m.
        supplement_m_values = tuple(M_VALUES)
        supplement_tasks = [
            (
                mode,
                "high_resolution",
                scenario_name,
                n,
                float(rho),
                supplement_m_values,
            )
            for n in SUPPLEMENT_N_VALUES
            for rho in supplement_rho_values
        ]
    return base_tasks + supplement_tasks


def resolution_block(task: SimulationTask) -> tuple[str, np.ndarray]:
    mode, _, scenario_name, n, rho, m_values = task
    scenario = SCENARIO_BY_NAME[scenario_name]
    pi1 = float(scenario["pi1"])
    rho_code = int(round(rho * 1000))
    values = []

    for seed in range(N_SEEDS):
        if mode == "weighted":
            calib_rng = rng_for(n, rho_code, seed, 0)
            calib_z = calib_rng.normal(0.0, 1.0, n)
            calib_weights = np.exp(rho * calib_z - 0.5 * rho**2)
            total_calib_weight = float(np.sum(calib_weights))
        else:
            total_calib_weight = float(n)

        for m in m_values:
            n_anomaly = max(1, int(round(pi1 * m)))
            n_inlier = m - n_anomaly
            if mode == "weighted":
                test_rng = rng_for(n, m, rho_code, seed, 1)
                _ = test_rng.normal(rho, 1.0, n_inlier)
                anomaly_z = test_rng.normal(rho, 1.0, n_anomaly)
                anomaly_weights = np.exp(rho * anomaly_z - 0.5 * rho**2)
            else:
                anomaly_weights = np.ones(n_anomaly, dtype=float)
            p_min_anomaly = anomaly_weights / (anomaly_weights + total_calib_weight)
            values.append(np.log10(1.0 / float(np.min(p_min_anomaly))))

    return scenario_name, np.asarray(values, dtype=float)


def adaptive_heatmap_y_edges(values: np.ndarray) -> np.ndarray:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) == 0:
        raise ValueError("Cannot build heatmap y-axis edges from no finite values.")

    lower = float(np.quantile(finite_values, HEATMAP_Y_LOWER_QUANTILE))
    upper = float(np.quantile(finite_values, HEATMAP_Y_UPPER_QUANTILE))
    displayed_values = finite_values[
        (finite_values >= lower) & (finite_values <= upper)
    ]
    if len(displayed_values) == 0:
        displayed_values = finite_values

    max_intervals = max(1, int(HEATMAP_Y_BINS) - 1)
    unique_values = np.unique(displayed_values)
    if len(unique_values) <= max_intervals:
        if len(unique_values) == 1:
            width = max(1e-6, abs(float(unique_values[0])) * 1e-6)
            return np.array([unique_values[0] - width, unique_values[0] + width])
        midpoints = (unique_values[:-1] + unique_values[1:]) / 2.0
        return np.concatenate(
            [
                [unique_values[0] - (midpoints[0] - unique_values[0])],
                midpoints,
                [unique_values[-1] + (unique_values[-1] - midpoints[-1])],
            ]
        )

    edges = np.unique(
        np.quantile(displayed_values, np.linspace(0.0, 1.0, max_intervals + 1))
    )
    span = float(edges[-1] - edges[0])
    eps = max(
        np.finfo(float).eps * max(abs(float(edges[0])), abs(float(edges[-1])), 1.0) * 16,
        span * 1e-12,
    )
    edges[0] -= eps
    edges[-1] += eps

    bin_idx = np.searchsorted(edges, displayed_values, side="right") - 1
    valid = (0 <= bin_idx) & (bin_idx < len(edges) - 1)
    counts = np.bincount(bin_idx[valid], minlength=len(edges) - 1)
    if np.all(counts > 0):
        return edges

    compact_edges = [float(edges[0])]
    for idx, count in enumerate(counts):
        if count > 0:
            compact_edges.append(float(edges[idx + 1]))
    return np.asarray(compact_edges, dtype=float)


def compute_y_edges(tasks: list[SimulationTask]) -> dict[str, np.ndarray]:
    y_values = {scenario["name"]: [] for scenario in SCENARIOS}
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for scenario_name, block_values in executor.map(resolution_block, tasks):
            y_values[scenario_name].append(block_values)

    edges = {}
    for scenario in SCENARIOS:
        scenario_name = scenario["name"]
        values = np.concatenate(y_values[scenario_name])
        edges[scenario_name] = adaptive_heatmap_y_edges(values)
    return edges


def add_count(mapping: dict[tuple, list[float]], key: tuple, x: float, success: bool) -> None:
    bucket = mapping.setdefault(key, [0.0, 0.0, 0.0])
    bucket[0] += float(success)
    bucket[1] += 1.0
    bucket[2] += float(x)


def simulate_summary_block(
    task: SimulationTask,
    y_edges_by_scenario: dict[str, np.ndarray],
    pruning: str,
) -> tuple[dict[tuple, list[float]], dict[tuple, list[float]]]:
    mode, grid_part, scenario_name, n, rho, m_values = task
    _ = grid_part
    mode = normalize_mode(mode)
    pruning = normalize_pruning(pruning)
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
        if mode == "weighted":
            calib_z = calib_rng.normal(0.0, 1.0, n)
            calib_weights = np.exp(rho * calib_z - 0.5 * rho**2)
        else:
            calib_weights = np.ones(n, dtype=float)
        calib_t = calib_rng.normal(0.0, 1.0, n)
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

            if mode == "weighted":
                inlier_z = test_rng.normal(rho, 1.0, n_inlier)
                anomaly_z = test_rng.normal(rho, 1.0, n_anomaly)
                test_z = np.concatenate([inlier_z, anomaly_z])
                test_weights = np.exp(rho * test_z - 0.5 * rho**2)
            else:
                test_weights = np.ones(m, dtype=float)
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
                    finite_max = float(
                        np.max(np.concatenate([calib_t, inlier_scores, anomaly_noise]))
                    )
                    anomaly_scores = np.full(n_anomaly, finite_max + 1.0)
                else:
                    anomaly_scores = kappa + anomaly_noise
                test_scores = np.concatenate([inlier_scores, anomaly_scores])
                if mode == "weighted":
                    p_values = weighted_tail_p_values(
                        sorted_calib_scores,
                        suffix_calib_weights,
                        total_calib_weight,
                        test_scores,
                        test_weights,
                    )
                    decisions = accelerated_wcs_decisions(
                        p_values,
                        test_scores,
                        sorted_calib_scores,
                        sorted_calib_weights,
                        total_calib_weight,
                        test_weights,
                        alpha,
                        pruning=pruning,
                        seed=BASE_SEED + n * 100000 + m * 100 + rho_code + seed,
                    )
                else:
                    p_values = standard_tail_p_values(sorted_calib_scores, test_scores)
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


def build_summaries(mode: str, pruning: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mode = normalize_mode(mode)
    pruning = normalize_pruning(pruning)
    summary_version = summary_version_for(mode, pruning)
    tasks = []
    for scenario in SCENARIOS:
        tasks.extend(tasks_for_scenario(mode, scenario["name"]))

    print(f"[{mode}] computing heatmap y-axis support", flush=True)
    y_edges_by_scenario = compute_y_edges(tasks)

    heatmap_counts: dict[tuple, list[float]] = {}
    collapse_counts: dict[tuple, list[float]] = {}
    total_tasks = len(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = executor.map(
            simulate_summary_block,
            tasks,
            [y_edges_by_scenario] * len(tasks),
            [pruning] * len(tasks),
        )
        for done, (heatmap_update, collapse_update) in enumerate(futures, start=1):
            merge_counts(heatmap_counts, heatmap_update)
            merge_counts(collapse_counts, collapse_update)
            progress_interval = len(RHO_VALUES) if mode == "weighted" else 1
            if done % progress_interval == 0 or done == total_tasks:
                print(f"[{mode}] completed summary block {done}/{total_tasks}", flush=True)

    heatmap_rows = []
    for (scenario_name, kappa, x_bin, y_bin), values in heatmap_counts.items():
        scenario = SCENARIO_BY_NAME[scenario_name]
        x_edges, _ = x_edges_for_alpha(float(scenario["alpha"]))
        y_edges = y_edges_by_scenario[scenario_name]
        successes, count, _ = values
        heatmap_rows.append(
            {
                "summary_version": summary_version,
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
                "summary_version": summary_version,
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


def summaries_are_current(paths: Figure1Paths, mode: str, pruning: str) -> bool:
    if (
        not paths.heatmap_summary_path.exists()
        or not paths.collapse_summary_path.exists()
    ):
        return False
    heatmap_header = pd.read_csv(paths.heatmap_summary_path, nrows=1)
    collapse_header = pd.read_csv(paths.collapse_summary_path, nrows=1)
    if heatmap_header.empty or collapse_header.empty:
        return False
    expected_version = summary_version_for(mode, pruning)
    return (
        str(heatmap_header["summary_version"].iloc[0]) == expected_version
        and str(collapse_header["summary_version"].iloc[0]) == expected_version
    )


def normalize_summary_keys(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    if "kappa" in summary.columns:
        summary["kappa"] = summary["kappa"].map(kappa_label)
    return summary


def load_or_build_summaries(
    paths: Figure1Paths,
    mode: str,
    pruning: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summaries_are_current(paths, mode, pruning):
        print(f"[{mode}] loading existing compact summaries", flush=True)
        return (
            normalize_summary_keys(pd.read_csv(paths.heatmap_summary_path)),
            normalize_summary_keys(pd.read_csv(paths.collapse_summary_path)),
        )

    heatmap_summary, collapse_summary = build_summaries(mode, pruning)
    heatmap_summary.to_csv(paths.heatmap_summary_path, index=False)
    collapse_summary.to_csv(paths.collapse_summary_path, index=False)
    return heatmap_summary, collapse_summary


def plot_schematic(ax: plt.Axes, mode: str) -> None:
    mode = normalize_mode(mode)
    rng = np.random.default_rng(BASE_SEED)
    rho = 1.5 if mode == "weighted" else 0.0
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
        label=r"$Q_\rho$ shifted inliers" if mode == "weighted" else r"$P_0$ test inliers",
    )
    ax.scatter(
        anomaly[:, 0],
        anomaly[:, 1],
        s=18,
        alpha=0.78,
        color=COLORS["anomaly"],
        label=r"$A_{\rho,\kappa}$ anomalies",
    )
    if mode == "weighted":
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
    if mode == "weighted":
        ax.text(-3.1, 4.45, r"weights depend on $Z$", fontsize=9)
    else:
        ax.text(-3.1, 4.45, r"unweighted exchangeable inliers", fontsize=9)
    ax.set_title(
        "Controlled Gaussian shift"
        if mode == "weighted"
        else "Unweighted Gaussian baseline"
    )
    ax.set_xlabel(r"benign shift direction $Z$")
    ax.set_ylabel(r"anomaly-score direction $T$")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.18, linewidth=0.6)


def plot_panel_a(paths: Figure1Paths, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    plot_schematic(ax, mode)
    fig.savefig(paths.schematic_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def heatmap_matrix(
    summary: pd.DataFrame,
    scenario: str,
    kappa: str,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    block = summary[(summary["scenario"].eq(scenario)) & (summary["kappa"].eq(kappa))]
    x_edges = np.unique(np.concatenate([block["x_left"].to_numpy(), block["x_right"].to_numpy()]))
    y_edges = np.unique(np.concatenate([block["y_bottom"].to_numpy(), block["y_top"].to_numpy()]))
    matrix = np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan)
    x_index_by_left = {float(edge): idx for idx, edge in enumerate(x_edges[:-1])}
    y_index_by_bottom = {float(edge): idx for idx, edge in enumerate(y_edges[:-1])}
    for row in block.itertuples(index=False):
        x_idx = x_index_by_left[float(row.x_left)]
        y_idx = y_index_by_bottom[float(row.y_bottom)]
        matrix[x_idx, y_idx] = float(row.probability)
    return x_edges, y_edges, np.ma.masked_invalid(matrix)


def heatmap_axis_limits(summary: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(summary["x_left"].min()), float(summary["x_right"].max())),
        (float(summary["y_bottom"].min()), float(summary["y_top"].max())),
    )


def plot_heatmap_grid(
    summary: pd.DataFrame,
    paths: Figure1Paths,
    mode: str,
    pruning: str,
) -> None:
    selection_label = method_label(mode, pruning)
    x_limits, y_limits = heatmap_axis_limits(summary)
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.6, 12.8),
        constrained_layout=True,
        sharey=False,
    )
    column_specs = [
        ("inf", f"Perfect-score {selection_label}"),
        ("3.0", rf"Finite-score {selection_label} ($\kappa=3.0$)"),
    ]
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
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
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
    cbar.set_label(f"Pr({selection_label} finds >=1 anomaly)")
    fig.suptitle("Finite-sample detectability under alpha and anomaly-rate changes", fontsize=14)
    fig.savefig(paths.heatmap_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_collapse_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    diagnostic: str,
    title: str,
    xlabel: str,
    selection_label: str,
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
    ax.set_ylabel(f"Pr({selection_label} finds >=1 anomaly)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")


def plot_collapse_figure(
    summary: pd.DataFrame,
    paths: Figure1Paths,
    mode: str,
    pruning: str,
) -> None:
    selection_label = method_label(mode, pruning)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), constrained_layout=True)
    plot_collapse_panel(
        axes[0],
        summary,
        "first_threshold",
        "D. First-threshold diagnostic",
        r"$\log_{10}\delta$,  $\delta=p^{\min}_{(1)}/(\alpha/m)$",
        selection_label,
    )
    plot_collapse_panel(
        axes[1],
        summary,
        "rank_aware",
        "E. Rank-aware BH-scale diagnostic",
        r"$\log_{10}\Delta_{\mathrm{BH}}$,  "
        r"$\Delta_{\mathrm{BH}}=\min_r p^{\min}_{(r)}/((r/m)\alpha)$",
        selection_label,
    )
    fig.suptitle(r"Baseline collapse diagnostics ($\alpha=0.10,\ \pi_1=0.10$)", fontsize=14)
    fig.savefig(paths.collapse_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def export_schematic_tikz(paths: Figure1Paths, mode: str) -> None:
    mode = normalize_mode(mode)
    rng = np.random.default_rng(BASE_SEED)
    rho = 1.5 if mode == "weighted" else 0.0
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
    shifted_label = (
        r"$Q_\rho$ shifted inliers" if mode == "weighted" else r"$P_0$ test inliers"
    )
    panel_title = (
        "Controlled Gaussian shift" if mode == "weighted" else "Unweighted Gaussian baseline"
    )
    for group, label, style_key, points, plot_order in [
        ("calibration", r"$P_0$ calibration", "calibration", calib, 1),
        ("shifted_inlier", shifted_label, "shifted", shifted, 2),
        ("anomaly", r"$A_{\rho,\kappa}$ anomalies", "anomaly", anomaly, 3),
    ]:
        for idx, (x, y) in enumerate(points):
            point_rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure1",
                    "panel": "A",
                    "panel_title": panel_title,
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
    pd.DataFrame(point_rows).to_csv(paths.schematic_points_tikz_path, index=False)

    annotation_rows = [
        {
            "export_version": TIKZ_EXPORT_VERSION,
            "figure": "figure1",
            "panel": "A",
            "panel_title": panel_title,
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
            "panel_title": panel_title,
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
            "panel_title": panel_title,
            "plot_order": 4,
            "object_type": "text",
            "x": -3.1,
            "y": 4.45,
            "x_end": "",
            "y_end": "",
            "style_key": "text",
            "label": r"weights depend on $Z$"
            if mode == "weighted"
            else r"unweighted exchangeable inliers",
        },
    ]
    if mode == "weighted":
        annotation_rows.insert(
            0,
            {
                "export_version": TIKZ_EXPORT_VERSION,
                "figure": "figure1",
                "panel": "A",
                "panel_title": panel_title,
                "plot_order": 1,
                "object_type": "arrow",
                "x": 0.1,
                "y": -2.7,
                "x_end": rho,
                "y_end": -2.7,
                "style_key": "covariate_shift_arrow",
                "label": "benign covariate shift",
            },
        )
    pd.DataFrame(annotation_rows).to_csv(
        paths.schematic_annotations_tikz_path,
        index=False,
    )


def export_heatmap_tikz(
    summary: pd.DataFrame,
    paths: Figure1Paths,
    mode: str,
    pruning: str,
) -> None:
    selection_label = method_label(mode, pruning)
    method = method_key(mode, pruning)
    column_specs = [
        ("inf", f"Perfect-score {selection_label}"),
        ("3.0", rf"Finite-score {selection_label} ($\kappa=3.0$)"),
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
            block["method"] = method
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

    pd.concat(rows, ignore_index=True).to_csv(paths.heatmap_tikz_path, index=False)
    pd.DataFrame(boundary_rows).to_csv(
        paths.heatmap_boundary_tikz_path,
        index=False,
    )


def export_collapse_tikz(
    summary: pd.DataFrame,
    paths: Figure1Paths,
    mode: str,
    pruning: str,
) -> None:
    method = method_key(mode, pruning)
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
            {
                kappa: idx + 1
                for idx, kappa in enumerate(["inf", "2.0", "2.5", "3.0", "3.5", "4.0"])
            }
        )
        block["group"] = diagnostic
        block["method"] = method
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
    pd.concat(rows, ignore_index=True).to_csv(paths.collapse_tikz_path, index=False)
    pd.DataFrame(reference_rows).to_csv(
        paths.collapse_reference_tikz_path,
        index=False,
    )


def export_tikz_csvs(
    heatmap_summary: pd.DataFrame,
    collapse_summary: pd.DataFrame,
    paths: Figure1Paths,
    mode: str,
    pruning: str,
) -> None:
    export_schematic_tikz(paths, mode)
    export_heatmap_tikz(heatmap_summary, paths, mode, pruning)
    export_collapse_tikz(collapse_summary, paths, mode, pruning)


def validate_outputs(
    heatmap_summary: pd.DataFrame,
    collapse_summary: pd.DataFrame,
    paths: Figure1Paths,
) -> None:
    expected_scenarios = {scenario["name"] for scenario in SCENARIOS}
    observed_scenarios = set(heatmap_summary["scenario"].unique())
    if observed_scenarios != expected_scenarios:
        raise RuntimeError(f"Unexpected heatmap scenarios: {observed_scenarios}")
    if set(collapse_summary["scenario"].unique()) != {BASELINE_SCENARIO}:
        raise RuntimeError("Collapse summary must contain only the baseline scenario.")
    for path in [
        paths.schematic_path,
        paths.heatmap_path,
        paths.collapse_path,
        paths.schematic_points_tikz_path,
        paths.schematic_annotations_tikz_path,
        paths.heatmap_tikz_path,
        paths.heatmap_boundary_tikz_path,
        paths.collapse_tikz_path,
        paths.collapse_reference_tikz_path,
    ]:
        if not path.exists():
            raise RuntimeError(f"Missing output: {path}")


def run_mode(mode: str, pruning: str) -> list[Path]:
    mode = normalize_mode(mode)
    pruning = normalize_pruning(pruning)
    paths = paths_for_mode(mode)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_summary, collapse_summary = load_or_build_summaries(paths, mode, pruning)
    plot_panel_a(paths, mode)
    plot_heatmap_grid(heatmap_summary, paths, mode, pruning)
    plot_collapse_figure(collapse_summary, paths, mode, pruning)
    export_tikz_csvs(heatmap_summary, collapse_summary, paths, mode, pruning)
    validate_outputs(heatmap_summary, collapse_summary, paths)
    return [
        paths.schematic_path,
        paths.heatmap_path,
        paths.collapse_path,
        paths.schematic_points_tikz_path,
        paths.schematic_annotations_tikz_path,
        paths.heatmap_tikz_path,
        paths.heatmap_boundary_tikz_path,
        paths.collapse_tikz_path,
        paths.collapse_reference_tikz_path,
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
        help="Simulation modes to run. Defaults to both unweighted and weighted.",
    )
    parser.add_argument(
        "--weighted-pruning",
        choices=WCS_PRUNING_METHODS,
        default=DEFAULT_WEIGHTED_PRUNING,
        help="WCS pruning method for weighted mode outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help=(
            "Number of worker processes for simulation blocks. Defaults to "
            "FIGURE1_WORKERS or one less than the detected CPU count."
        ),
    )
    parser.add_argument(
        "--wcs-batch-size",
        type=int,
        default=WCS_CANDIDATE_BATCH_SIZE,
        help=(
            "Candidate rows per WCS auxiliary-p-value batch. Lower values reduce "
            "peak memory without changing selections."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global WCS_CANDIDATE_BATCH_SIZE, WORKERS
    args = parse_args(argv)
    WORKERS = max(1, int(args.workers))
    WCS_CANDIDATE_BATCH_SIZE = max(1, int(args.wcs_batch_size))
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )
    for mode in args.modes:
        for path in run_mode(mode, args.weighted_pruning):
            print(path)


if __name__ == "__main__":
    main()
