"""Build Figure 3: operational sufficiency diagnostics.

The top row descriptively collapses the existing Figure 1 and Figure 2 atlas
cells against their median rank-resolution diagnostic.  Each atlas cell has
equal weight; the curves therefore summarize the designed operating grid, not
independent Monte Carlo trials or a causal response curve.

The bottom row compares weighted simulations whose *realized* calibration
effective sample sizes fall in the same narrow log-scale bands.  The comparison
holds ``alpha=0.1``, ``m=500``, and ``pi1=0.05`` fixed and varies the Gaussian
shift parameter ``rho``.  Matching effective sample size does not match the
full normalized-weight distribution, calibration size, or test-point
self-atoms.  The experiment is consequently a sufficiency stress test for
``N_eff``, not a proof that any observed residual difference is caused by one
particular weight functional.

The rank diagnostic compares anomaly-side attainable p-value atoms with BH
rank scales.  In the weighted WCS pipeline it is a resolution marker, not a
complete decision boundary or a no-discovery certificate.

Default run::

    python -m wkde_cad.simulations.figure3_operational_sufficiency

Small smoke run::

    python -m wkde_cad.simulations.figure3_operational_sufficiency --matched-trials 1 --workers 1 --force
"""

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
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wkde_cad.simulations import figure2_weighted_power_atlas as weighted


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "outputs" / "simulations" / Path(__file__).stem
FIGURE_PATH = OUT_DIR / "figure3_operational_sufficiency.png"
COLLAPSE_SUMMARY_PATH = OUT_DIR / "figure3_atlas_collapse_summary.csv"
MATCHED_TRIALS_PATH = OUT_DIR / "figure3_matched_neff_trials.csv"
MATCHED_SUMMARY_PATH = OUT_DIR / "figure3_matched_neff_summary.csv"

UNWEIGHTED_ATLAS_SUMMARY_PATH = (
    REPO_ROOT
    / "outputs"
    / "simulations"
    / "figure1_unweighted_power_atlas"
    / "figure1_unweighted_power_atlas_summary.csv"
)
WEIGHTED_ATLAS_SUMMARY_PATH = (
    REPO_ROOT
    / "outputs"
    / "simulations"
    / "figure2_weighted_power_atlas"
    / "figure2_weighted_power_atlas_summary.csv"
)

SUMMARY_VERSION = "operational-sufficiency-v2"
BASE_SEED = 20260715

SCORE_REGIMES = ("perfect", "finite_k1", "finite_k2", "finite_k3")
MATCHED_SCORE_REGIMES = ("perfect", "finite_k3")
SCORE_LABELS = {
    "perfect": "perfect score",
    "finite_k1": r"finite score ($\kappa=1$)",
    "finite_k2": r"finite score ($\kappa=2$)",
    "finite_k3": r"finite score ($\kappa=3$)",
}
SCORE_COLORS = {
    "perfect": "#111111",
    "finite_k1": "#7E57C2",
    "finite_k2": "#1E88E5",
    "finite_k3": "#00897B",
}

MATCH_ALPHA = 0.10
MATCH_M = 500
MATCH_PI1 = 0.05
TARGET_LOG10_NEFF = (2.1, 2.4, 2.7, 3.0)
MATCH_RHOS = (0.0, 0.5, 1.0, 1.5)
MATCH_N_CAL_MAX = 16_000

DEFAULT_COLLAPSE_BINS = 14
# Match the weighted atlas's default Monte Carlo count so the residual
# between-rho differences are not judged from a smaller simulation budget.
DEFAULT_MATCHED_TRIALS = 100
DEFAULT_NEFF_HALF_WIDTH = 0.08
DEFAULT_MAX_ATTEMPTS_PER_CELL = 1_000
DEFAULT_WCS_BATCH_SIZE = weighted.DEFAULT_WCS_BATCH_SIZE
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)


@dataclass(frozen=True)
class MatchTask:
    target_log10_neff: float
    rho: float
    accepted_trials: int
    neff_half_width: float
    max_attempts: int
    wcs_batch_size: int


def _read_atlas_summary(path: Path, method: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {method} atlas summary: {path}. Run Figures 1 and 2 first."
        )
    summary = pd.read_csv(path)
    required = {"score_regime", "median_log10_rank_delta", "power"}
    missing = required.difference(summary.columns)
    if missing:
        raise RuntimeError(
            f"{method} atlas summary is missing columns: {sorted(missing)}"
        )
    if summary.empty:
        raise RuntimeError(f"{method} atlas summary is empty.")
    if not set(summary["score_regime"]).issubset(SCORE_REGIMES):
        raise RuntimeError(f"{method} atlas summary has unknown score regimes.")
    diagnostic = pd.to_numeric(summary["median_log10_rank_delta"], errors="coerce")
    power = pd.to_numeric(summary["power"], errors="coerce")
    if not np.isfinite(diagnostic).all() or not np.isfinite(power).all():
        raise RuntimeError(f"{method} atlas summary contains non-finite values.")
    if not ((power >= 0.0) & (power <= 1.0)).all():
        raise RuntimeError(f"{method} atlas power must lie in [0, 1].")
    result = summary.copy()
    result["median_log10_rank_delta"] = diagnostic
    result["power"] = power
    if "summary_version" in summary:
        result["source_summary_version"] = summary["summary_version"].astype(str)
    else:
        result["source_summary_version"] = "unversioned"
    result.insert(0, "method", method)
    return result


def collapse_bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Return shared quarter-decade-rounded edges for the two atlas collapses."""

    values = np.asarray(values, dtype=float)
    if bins < 2:
        raise ValueError("collapse bins must be at least 2")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("collapse diagnostics must be finite and non-empty")
    lower = math.floor(float(np.min(values)) * 4.0) / 4.0
    upper = math.ceil(float(np.max(values)) * 4.0) / 4.0
    lower = min(lower, -0.25)
    upper = max(upper, 0.25)
    if math.isclose(lower, upper):
        lower -= 0.25
        upper += 0.25
    return np.linspace(lower, upper, int(bins) + 1)


def build_collapse_summary(
    unweighted_path: Path,
    weighted_path: Path,
    *,
    bins: int,
) -> pd.DataFrame:
    """Bin equal-weighted atlas-cell power by the rank-resolution diagnostic."""

    combined = pd.concat(
        [
            _read_atlas_summary(unweighted_path, "unweighted"),
            _read_atlas_summary(weighted_path, "weighted"),
        ],
        ignore_index=True,
    )
    edges = collapse_bin_edges(
        combined["median_log10_rank_delta"].to_numpy(dtype=float),
        bins,
    )
    diagnostic = combined["median_log10_rank_delta"].to_numpy(dtype=float)
    bin_index = np.searchsorted(edges, diagnostic, side="right") - 1
    bin_index[np.isclose(diagnostic, edges[-1])] = len(edges) - 2
    combined["bin_index"] = bin_index
    combined = combined[
        (combined["bin_index"] >= 0) & (combined["bin_index"] < len(edges) - 1)
    ].copy()

    rows: list[dict[str, Any]] = []
    for (method, regime, index), block in combined.groupby(
        ["method", "score_regime", "bin_index"], sort=True
    ):
        powers = block["power"].to_numpy(dtype=float)
        index = int(index)
        rows.append(
            {
                "summary_version": SUMMARY_VERSION,
                "aggregation_unit": "equal_atlas_cells",
                "method": method,
                "source_summary_version": "|".join(
                    sorted(set(block["source_summary_version"].astype(str)))
                ),
                "score_regime": regime,
                "bin_index": index,
                "diagnostic_left": float(edges[index]),
                "diagnostic_right": float(edges[index + 1]),
                "diagnostic_center": float((edges[index] + edges[index + 1]) / 2.0),
                "cell_count": int(len(block)),
                "diagnostic_mean": float(
                    block["median_log10_rank_delta"].mean()
                ),
                "power_mean": float(np.mean(powers)),
                "power_median": float(np.median(powers)),
                "power_q25": float(np.quantile(powers, 0.25)),
                "power_q75": float(np.quantile(powers, 0.75)),
            }
        )
    result = pd.DataFrame(rows)
    validate_collapse_summary(result)
    return result.sort_values(
        ["method", "score_regime", "bin_index"], ignore_index=True
    )


def validate_collapse_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        raise RuntimeError("Figure 3 atlas collapse summary is empty.")
    if set(summary["method"]) != {"unweighted", "weighted"}:
        raise RuntimeError("Figure 3 collapse must contain both atlas methods.")
    if (summary["cell_count"] <= 0).any():
        raise RuntimeError("Figure 3 collapse contains an empty bin.")
    for column in ["power_mean", "power_median", "power_q25", "power_q75"]:
        if not ((summary[column] >= 0.0) & (summary[column] <= 1.0)).all():
            raise RuntimeError(f"Figure 3 collapse column {column} is outside [0, 1].")


def n_cal_candidates(
    target_log10_neff: float,
    rho: float,
    *,
    n_cal_max: int = MATCH_N_CAL_MAX,
) -> list[int]:
    """Construct calibration-size candidates around the lognormal ESS heuristic."""

    target_neff = 10.0 ** float(target_log10_neff)
    ideal_n_cal = target_neff * math.exp(float(rho) ** 2)
    multipliers = (0.70, 0.85, 1.0, 1.15, 1.35, 1.60)
    candidates = {
        max(2, int(round(ideal_n_cal * multiplier))) for multiplier in multipliers
    }
    candidates = {value for value in candidates if value <= int(n_cal_max)}
    if not candidates:
        raise ValueError(
            "No feasible calibration size for matched ESS target "
            f"log10(N_eff)={target_log10_neff:g}, rho={rho:g}, "
            f"n_cal_max={n_cal_max}."
        )
    return sorted(candidates)


def _match_seed(target_log10_neff: float, rho: float, attempt: int) -> int:
    target_code = int(round(float(target_log10_neff) * 1_000))
    rho_code = int(round(float(rho) * 1_000))
    seed_sequence = np.random.SeedSequence(
        [BASE_SEED, target_code, rho_code, int(attempt)]
    )
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _flatten_accepted_trial(
    row: dict[str, Any],
    task: MatchTask,
    *,
    attempt: int,
    accepted_index: int,
) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "summary_version": SUMMARY_VERSION,
        "base_seed": BASE_SEED,
        "weighted_simulator_version": weighted.SUMMARY_VERSION,
        "target_log10_neff": task.target_log10_neff,
        "neff_band_lower": task.target_log10_neff - task.neff_half_width,
        "neff_band_upper": task.target_log10_neff + task.neff_half_width,
        "neff_half_width": task.neff_half_width,
        "rho": task.rho,
        "alpha": MATCH_ALPHA,
        "m": MATCH_M,
        "pi1": MATCH_PI1,
        "requested_trials": task.accepted_trials,
        "attempt_index": int(attempt),
        "accepted_index": int(accepted_index),
        "n_cal": int(row["n_cal"]),
        "calib_ess": float(row["calib_ess"]),
        "log10_neff": float(row["log10_neff"]),
        "n_anomaly": int(row["n_anomaly"]),
        "actual_pi1": float(row["actual_pi1"]),
        "rank_delta": float(row["rank_delta"]),
        "log10_rank_delta": float(row["log10_rank_delta"]),
        "rank_delta_above_one": bool(float(row["rank_delta"]) > 1.0),
    }
    for regime in MATCHED_SCORE_REGIMES:
        metrics = row["metrics"][regime]
        flat[f"{regime}_any_discovery"] = bool(metrics["any_discovery"])
        flat[f"{regime}_power"] = float(metrics["power"])
        flat[f"{regime}_fdr"] = float(metrics["fdr"])
        flat[f"{regime}_auroc"] = float(metrics["auroc"])
        flat[f"{regime}_perfect_separation_rate"] = float(
            metrics["perfect_separation_from_calibration"]
        )
    return flat


def simulate_matched_cell(task: MatchTask) -> pd.DataFrame:
    """Accept weighted trials inside one declared realized-ESS band."""

    candidates = n_cal_candidates(task.target_log10_neff, task.rho)
    accepted: list[dict[str, Any]] = []
    attempts = 0
    lower = task.target_log10_neff - task.neff_half_width
    upper = task.target_log10_neff + task.neff_half_width

    while len(accepted) < task.accepted_trials and attempts < task.max_attempts:
        n_cal = candidates[attempts % len(candidates)]
        seed = _match_seed(task.target_log10_neff, task.rho, attempts)
        row = weighted.simulate_trial(
            alpha=MATCH_ALPHA,
            m=MATCH_M,
            pi1=MATCH_PI1,
            n_anomaly_fixed=None,
            n_cal=n_cal,
            rho=task.rho,
            seed=seed,
            batch_size=task.wcs_batch_size,
        )
        attempts += 1
        if not lower <= float(row["log10_neff"]) <= upper:
            continue
        accepted.append(
            _flatten_accepted_trial(
                row,
                task,
                attempt=attempts - 1,
                accepted_index=len(accepted),
            )
        )

    if len(accepted) < task.accepted_trials:
        raise RuntimeError(
            "Could not fill Figure 3 matched-ESS cell: "
            f"target={task.target_log10_neff:g}, rho={task.rho:g}, "
            f"band=[{lower:g}, {upper:g}], accepted={len(accepted)}, "
            f"attempts={attempts}."
        )
    result = pd.DataFrame(accepted)
    result["attempts_for_cell"] = attempts
    return result


def matched_tasks(
    *,
    accepted_trials: int,
    neff_half_width: float,
    max_attempts: int,
    wcs_batch_size: int,
) -> list[MatchTask]:
    return [
        MatchTask(
            target_log10_neff=float(target),
            rho=float(rho),
            accepted_trials=int(accepted_trials),
            neff_half_width=float(neff_half_width),
            max_attempts=int(max_attempts),
            wcs_batch_size=int(wcs_batch_size),
        )
        for target in TARGET_LOG10_NEFF
        for rho in MATCH_RHOS
    ]


def run_matched_tasks(tasks: list[MatchTask], workers: int) -> pd.DataFrame:
    if int(workers) <= 1:
        frames = [
            simulate_matched_cell(task)
            for task in tqdm(tasks, desc="Figure 3 matched ESS cells")
        ]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            frames = list(
                tqdm(
                    executor.map(simulate_matched_cell, tasks),
                    total=len(tasks),
                    desc="Figure 3 matched ESS cells",
                )
            )
    return pd.concat(frames, ignore_index=True)


def validate_matched_trials(
    trials: pd.DataFrame,
    *,
    requested_trials: int,
    neff_half_width: float,
) -> None:
    if trials.empty:
        raise RuntimeError("Figure 3 matched-ESS trials are empty.")
    if set(trials["summary_version"]) != {SUMMARY_VERSION}:
        raise RuntimeError("Figure 3 matched-ESS cache has the wrong version.")
    expected_cells = len(TARGET_LOG10_NEFF) * len(MATCH_RHOS)
    counts = trials.groupby(["target_log10_neff", "rho"], sort=False).size()
    if len(counts) != expected_cells or not (counts == int(requested_trials)).all():
        raise RuntimeError("Figure 3 matched-ESS cache has incomplete cells.")
    target = trials["target_log10_neff"].to_numpy(dtype=float)
    achieved = trials["log10_neff"].to_numpy(dtype=float)
    if not (np.abs(achieved - target) <= float(neff_half_width) + 1e-12).all():
        raise RuntimeError("Figure 3 accepted a trial outside the ESS band.")
    for regime in MATCHED_SCORE_REGIMES:
        for suffix in ["any_discovery", "power", "fdr", "auroc"]:
            values = trials[f"{regime}_{suffix}"].to_numpy(dtype=float)
            if not ((values >= 0.0) & (values <= 1.0)).all():
                raise RuntimeError(
                    f"Figure 3 matched metric {regime}_{suffix} is outside [0, 1]."
                )


def matched_cache_is_current(
    path: Path,
    *,
    requested_trials: int,
    neff_half_width: float,
) -> bool:
    if not path.exists():
        return False
    try:
        trials = pd.read_csv(path)
        validate_matched_trials(
            trials,
            requested_trials=requested_trials,
            neff_half_width=neff_half_width,
        )
    except (OSError, ValueError, KeyError, RuntimeError, pd.errors.ParserError):
        return False
    metadata_columns = {"base_seed", "weighted_simulator_version"}
    if not metadata_columns.issubset(trials.columns):
        return False
    if not np.allclose(trials["alpha"], MATCH_ALPHA):
        return False
    if not np.allclose(trials["m"], MATCH_M):
        return False
    if not np.allclose(trials["pi1"], MATCH_PI1):
        return False
    if not np.allclose(trials["neff_half_width"], neff_half_width):
        return False
    if not (trials["base_seed"].astype(int) == BASE_SEED).all():
        return False
    if set(trials["weighted_simulator_version"].astype(str)) != {
        weighted.SUMMARY_VERSION
    }:
        return False
    if set(np.round(trials["target_log10_neff"], 12)) != set(
        np.round(TARGET_LOG10_NEFF, 12)
    ):
        return False
    if set(np.round(trials["rho"], 12)) != set(np.round(MATCH_RHOS, 12)):
        return False
    return True


def load_or_build_matched_trials(
    *,
    force: bool,
    requested_trials: int,
    neff_half_width: float,
    max_attempts: int,
    wcs_batch_size: int,
    workers: int,
) -> pd.DataFrame:
    if not force and matched_cache_is_current(
        MATCHED_TRIALS_PATH,
        requested_trials=requested_trials,
        neff_half_width=neff_half_width,
    ):
        print("loading existing Figure 3 matched-ESS trials", flush=True)
        return pd.read_csv(MATCHED_TRIALS_PATH)
    trials = run_matched_tasks(
        matched_tasks(
            accepted_trials=requested_trials,
            neff_half_width=neff_half_width,
            max_attempts=max_attempts,
            wcs_batch_size=wcs_batch_size,
        ),
        workers,
    )
    validate_matched_trials(
        trials,
        requested_trials=requested_trials,
        neff_half_width=neff_half_width,
    )
    trials.to_csv(MATCHED_TRIALS_PATH, index=False)
    return trials


def _mean_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, 0.0, mean, mean
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return (
        mean,
        standard_error,
        max(0.0, mean - 1.96 * standard_error),
        min(1.0, mean + 1.96 * standard_error),
    )


def aggregate_matched_trials(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, rho), block in trials.groupby(
        ["target_log10_neff", "rho"], sort=True
    ):
        for regime in MATCHED_SCORE_REGIMES:
            power = block[f"{regime}_power"].to_numpy(dtype=float)
            power_mean, power_se, power_low, power_high = _mean_ci(power)
            rows.append(
                {
                    "summary_version": SUMMARY_VERSION,
                    "score_regime": regime,
                    "target_log10_neff": float(target),
                    "rho": float(rho),
                    "alpha": MATCH_ALPHA,
                    "m": MATCH_M,
                    "pi1": MATCH_PI1,
                    "neff_half_width": float(block["neff_half_width"].iloc[0]),
                    "accepted_trials": int(len(block)),
                    "attempts_for_cell": int(block["attempts_for_cell"].iloc[0]),
                    "acceptance_rate": float(
                        len(block) / block["attempts_for_cell"].iloc[0]
                    ),
                    "n_cal_min": int(block["n_cal"].min()),
                    "n_cal_mean": float(block["n_cal"].mean()),
                    "n_cal_max": int(block["n_cal"].max()),
                    "log10_neff_min": float(block["log10_neff"].min()),
                    "log10_neff_q25": float(block["log10_neff"].quantile(0.25)),
                    "log10_neff_mean": float(block["log10_neff"].mean()),
                    "log10_neff_median": float(block["log10_neff"].median()),
                    "log10_neff_q75": float(block["log10_neff"].quantile(0.75)),
                    "log10_neff_max": float(block["log10_neff"].max()),
                    "discovery_probability": float(
                        block[f"{regime}_any_discovery"].mean()
                    ),
                    "power_mean": power_mean,
                    "power_standard_error": power_se,
                    "power_ci95_low": power_low,
                    "power_ci95_high": power_high,
                    "fdr_mean": float(block[f"{regime}_fdr"].mean()),
                    "auroc_mean": float(block[f"{regime}_auroc"].mean()),
                    "mean_log10_rank_delta": float(
                        block["log10_rank_delta"].mean()
                    ),
                    "median_log10_rank_delta": float(
                        block["log10_rank_delta"].median()
                    ),
                    "rank_delta_above_one_rate": float(
                        block["rank_delta_above_one"].mean()
                    ),
                    "perfect_separation_rate": float(
                        block[f"{regime}_perfect_separation_rate"].mean()
                    ),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["score_regime", "rho", "target_log10_neff"], ignore_index=True
    )
    validate_matched_summary(result)
    return result


def validate_matched_summary(summary: pd.DataFrame) -> None:
    expected = len(MATCHED_SCORE_REGIMES) * len(TARGET_LOG10_NEFF) * len(MATCH_RHOS)
    if len(summary) != expected:
        raise RuntimeError(
            f"Figure 3 matched summary has {len(summary)} rows; expected {expected}."
        )
    if set(summary["score_regime"]) != set(MATCHED_SCORE_REGIMES):
        raise RuntimeError("Figure 3 matched summary has unexpected score regimes.")
    for column in [
        "acceptance_rate",
        "discovery_probability",
        "power_mean",
        "power_ci95_low",
        "power_ci95_high",
        "fdr_mean",
        "auroc_mean",
        "rank_delta_above_one_rate",
        "perfect_separation_rate",
    ]:
        if not ((summary[column] >= 0.0) & (summary[column] <= 1.0)).all():
            raise RuntimeError(f"Figure 3 matched column {column} is outside [0, 1].")


def _plot_collapse_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    method: str,
    title: str,
) -> None:
    for regime in SCORE_REGIMES:
        block = summary[
            summary["method"].eq(method) & summary["score_regime"].eq(regime)
        ].sort_values("diagnostic_center")
        if block.empty:
            continue
        x = block["diagnostic_mean"].to_numpy(dtype=float)
        y = block["power_mean"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.5,
            linewidth=1.9,
            color=SCORE_COLORS[regime],
            label=SCORE_LABELS[regime],
        )
        enough = block["cell_count"].to_numpy(dtype=int) >= 2
        if np.any(enough):
            ax.fill_between(
                x[enough],
                block.loc[enough, "power_q25"].to_numpy(dtype=float),
                block.loc[enough, "power_q75"].to_numpy(dtype=float),
                color=SCORE_COLORS[regime],
                alpha=0.10,
                linewidth=0,
            )
    ax.axvline(0.0, color="#D32F2F", linestyle="--", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(r"median rank-resolution margin $\log_{10}\Delta_{BH}$")
    ax.set_ylabel("statistical power")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.20)


def _plot_matched_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    regime: str,
    title: str,
) -> None:
    colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.88, max(2, len(MATCH_RHOS)))
    )
    for color, rho in zip(colors, MATCH_RHOS, strict=False):
        block = summary[
            summary["score_regime"].eq(regime)
            & np.isclose(summary["rho"].astype(float), float(rho))
        ].sort_values("target_log10_neff")
        if block.empty:
            continue
        x = block["log10_neff_mean"].to_numpy(dtype=float)
        y = block["power_mean"].to_numpy(dtype=float)
        lower = y - block["power_ci95_low"].to_numpy(dtype=float)
        upper = block["power_ci95_high"].to_numpy(dtype=float) - y
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            marker="o",
            markersize=5,
            linewidth=1.8,
            capsize=2.5,
            color=color,
            label=rf"$\rho={rho:g}$",
        )
    ax.set_title(title)
    ax.set_xlabel(r"achieved mean $\log_{10}N_{eff}$")
    ax.set_ylabel("statistical power")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.20)


def plot_figure(collapse: pd.DataFrame, matched: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    _plot_collapse_panel(
        axes[0, 0],
        collapse,
        "unweighted",
        "A. Standard CAD: equal-cell mean (IQR)",
    )
    _plot_collapse_panel(
        axes[0, 1],
        collapse,
        "weighted",
        "B. Weighted CAD: equal-cell mean (IQR)",
    )
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].text(
        0.98,
        0.04,
        "red: rank-feasibility boundary",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#B71C1C",
    )

    _plot_matched_panel(
        axes[1, 0],
        matched,
        "perfect",
        "C. Matched effective size: perfect score",
    )
    _plot_matched_panel(
        axes[1, 1],
        matched,
        "finite_k3",
        r"D. Matched effective size: finite score ($\kappa=3$)",
    )
    axes[1, 0].legend(frameon=False, fontsize=9, ncol=2)
    axes[1, 1].text(
        0.98,
        0.04,
        (
            f"{int(matched['accepted_trials'].iloc[0])} accepted trials/point; "
            "bars: normal 95% MC intervals\n"
            rf"accepted within $\pm {float(matched['neff_half_width'].iloc[0]):.2f}$ dex"
        ),
        transform=axes[1, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#424242",
    )

    fig.suptitle(
        "Operational sufficiency: rank feasibility and matched effective calibration size",
        fontsize=16,
    )
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def validate_outputs() -> None:
    for path in [
        FIGURE_PATH,
        COLLAPSE_SUMMARY_PATH,
        MATCHED_TRIALS_PATH,
        MATCHED_SUMMARY_PATH,
    ]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Figure 3 output is missing or empty: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collapse-bins",
        type=int,
        default=DEFAULT_COLLAPSE_BINS,
        help="Shared number of bins for each atlas-cell collapse panel.",
    )
    parser.add_argument(
        "--matched-trials",
        type=int,
        default=DEFAULT_MATCHED_TRIALS,
        help="Accepted weighted trials per target-N_eff/rho cell; use 1 for smoke runs.",
    )
    parser.add_argument(
        "--neff-half-width",
        type=float,
        default=DEFAULT_NEFF_HALF_WIDTH,
        help="Accepted half-width in log10 N_eff around every matched target.",
    )
    parser.add_argument(
        "--max-attempts-per-cell",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS_PER_CELL,
        help="Maximum simulated proposals used to fill each matched cell.",
    )
    parser.add_argument(
        "--wcs-batch-size",
        type=int,
        default=DEFAULT_WCS_BATCH_SIZE,
        help="Candidate batch size forwarded to the Figure 2 WCS implementation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Process workers across matched target/rho cells.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild matched trials instead of using a compatible CSV cache.",
    )
    args = parser.parse_args(argv)
    if args.collapse_bins < 2:
        parser.error("--collapse-bins must be at least 2")
    if args.matched_trials < 1:
        parser.error("--matched-trials must be positive")
    if not 0.0 < args.neff_half_width <= 0.25:
        parser.error("--neff-half-width must lie in (0, 0.25]")
    if args.max_attempts_per_cell < args.matched_trials:
        parser.error("--max-attempts-per-cell must be at least --matched-trials")
    if args.wcs_batch_size < 1 or args.workers < 1:
        parser.error("--wcs-batch-size and --workers must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    collapse = build_collapse_summary(
        UNWEIGHTED_ATLAS_SUMMARY_PATH,
        WEIGHTED_ATLAS_SUMMARY_PATH,
        bins=int(args.collapse_bins),
    )
    collapse.to_csv(COLLAPSE_SUMMARY_PATH, index=False)

    matched_trials = load_or_build_matched_trials(
        force=bool(args.force),
        requested_trials=int(args.matched_trials),
        neff_half_width=float(args.neff_half_width),
        max_attempts=int(args.max_attempts_per_cell),
        wcs_batch_size=int(args.wcs_batch_size),
        workers=int(args.workers),
    )
    matched_summary = aggregate_matched_trials(matched_trials)
    matched_summary.to_csv(MATCHED_SUMMARY_PATH, index=False)

    plot_figure(collapse, matched_summary)
    validate_outputs()


if __name__ == "__main__":
    main()
