from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
FIGURE_PATH = OUT_DIR / "figure3_randomized_pvalue_instability.png"
SUMMARY_PATH = OUT_DIR / "figure3_randomization_summary.csv"
DISTRIBUTION_PATH = OUT_DIR / "figure3_discovery_distribution.csv"
INTERVAL_TIKZ_PATH = OUT_DIR / "figure3_interval_tikz.csv"
DISTRIBUTION_TIKZ_PATH = OUT_DIR / "figure3_distribution_tikz.csv"
RATIONALE_PATH = OUT_DIR / "RATIONALE.md"

SUMMARY_VERSION = "randomized-instability-frontier-v2"
TIKZ_EXPORT_VERSION = "tikz-frontier-v1"
BASE_SEED = 20260512

ALPHA = 0.10
N_CAL = 500
M = 1000
N_ANOMALY = 10
DELTA_SCORE = 1.0
D = 10
RHO_VALUES = np.linspace(0.0, 2.5, 11)
N_WORLD_SEEDS = 40
N_RANDOMIZATIONS = 20_000
SIMULATION_BATCH_SIZE = 10_000
FRONTIER_BINS = 12
INLIER_TEST_WEIGHT_CAP_FRACTION = 0.5

COLORS = {
    "equal": "#111111",
    "moderate": "#7c3aed",
    "high": "#c2410c",
    "threshold": "#111111",
    "observed": "#c2410c",
    "theorem": "#0f766e",
    "points": "#4b5563",
}


@dataclass(frozen=True)
class FixedWorld:
    world_id: str
    rho: float
    world_seed: int
    calib_scores: np.ndarray
    test_scores: np.ndarray
    y_true: np.ndarray
    calib_weights: np.ndarray
    test_weights: np.ndarray
    pvalue_lower: np.ndarray
    pvalue_upper: np.ndarray


def rng_for(*values: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *values]))


def rho_code(rho: float) -> int:
    return int(round(float(rho) * 1000))


def world_id_for(rho: float, world_seed: int) -> str:
    return f"rho_{rho_code(rho):04d}_seed_{int(world_seed):03d}"


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights**2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(weights) ** 2 / denominator)


def bh_discovery_counts_for_rows(
    p_value_rows: np.ndarray,
    alpha: float,
    m_total: int,
) -> np.ndarray:
    p_value_rows = np.asarray(p_value_rows, dtype=float)
    if p_value_rows.ndim != 2:
        raise ValueError("p_value_rows must be a two-dimensional array.")
    n_candidates = p_value_rows.shape[1]
    sorted_p = np.sort(p_value_rows, axis=1, kind="mergesort")
    thresholds = alpha * np.arange(1, n_candidates + 1) / m_total
    passed = sorted_p <= thresholds[None, :]
    any_passed = np.any(passed, axis=1)
    reverse_last = np.argmax(passed[:, ::-1], axis=1)
    counts = n_candidates - reverse_last
    return np.where(any_passed, counts, 0).astype(int)


def randomized_pvalue_intervals(
    sorted_calib_scores: np.ndarray,
    suffix_calib_weights: np.ndarray,
    total_calib_weight: float,
    test_scores: np.ndarray,
    test_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tail_start = np.searchsorted(sorted_calib_scores, test_scores, side="left")
    tail_mass = suffix_calib_weights[tail_start]
    denominators = total_calib_weight + test_weights
    lower = tail_mass / denominators
    upper = (tail_mass + test_weights) / denominators
    return lower, upper


def sorted_calibration(
    calib_scores: np.ndarray,
    calib_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(calib_scores, kind="mergesort")
    sorted_scores = calib_scores[order]
    sorted_weights = calib_weights[order]
    suffix_weights = np.concatenate(([0.0], np.cumsum(sorted_weights[::-1])))[::-1]
    return sorted_scores, sorted_weights, suffix_weights


def build_fixed_world(rho: float, world_seed: int) -> FixedWorld:
    rho = float(rho)
    world_seed = int(world_seed)
    code = rho_code(rho)
    score_rng = rng_for(code, world_seed, 0)
    calib_scores = score_rng.normal(0.0, 1.0, int(N_CAL))
    n_inlier = int(M) - int(N_ANOMALY)
    if n_inlier < 1 or int(N_ANOMALY) < 1:
        raise ValueError("M must exceed N_ANOMALY and N_ANOMALY must be positive.")

    min_calib_score = float(np.min(calib_scores))
    max_calib_score = float(np.max(calib_scores))
    inlier_scores = np.full(n_inlier, min_calib_score - float(DELTA_SCORE))
    anomaly_scores = np.full(int(N_ANOMALY), max_calib_score + float(DELTA_SCORE))
    test_scores = np.concatenate([inlier_scores, anomaly_scores])
    y_true = np.concatenate(
        [np.zeros(n_inlier, dtype=bool), np.ones(int(N_ANOMALY), dtype=bool)]
    )

    if np.isclose(rho, 0.0):
        calib_weights = np.ones(int(N_CAL), dtype=float)
        test_weights = np.ones(int(M), dtype=float)
    else:
        weight_rng = rng_for(code, world_seed, 1)
        calib_x = weight_rng.normal(0.0, 1.0, size=(int(N_CAL), int(D)))
        inlier_x = weight_rng.normal(0.0, 1.0, size=(n_inlier, int(D)))
        anomaly_x = weight_rng.normal(0.0, 1.0, size=(int(N_ANOMALY), int(D)))
        inlier_x[:, 0] += rho
        anomaly_x[:, 0] += rho
        test_x = np.vstack([inlier_x, anomaly_x])
        calib_weights = np.exp(rho * calib_x[:, 0] - 0.5 * rho**2)
        test_weights = np.exp(rho * test_x[:, 0] - 0.5 * rho**2)
        inlier_cap = float(INLIER_TEST_WEIGHT_CAP_FRACTION) * float(
            np.sum(calib_weights)
        )
        test_weights[:n_inlier] = np.minimum(test_weights[:n_inlier], inlier_cap)

    sorted_scores, _, suffix_weights = sorted_calibration(calib_scores, calib_weights)
    lower, upper = randomized_pvalue_intervals(
        sorted_scores,
        suffix_weights,
        float(np.sum(calib_weights)),
        test_scores,
        test_weights,
    )
    return FixedWorld(
        world_id=world_id_for(rho, world_seed),
        rho=rho,
        world_seed=world_seed,
        calib_scores=calib_scores,
        test_scores=test_scores,
        y_true=y_true,
        calib_weights=calib_weights,
        test_weights=test_weights,
        pvalue_lower=lower,
        pvalue_upper=upper,
    )


def build_fixed_worlds() -> list[FixedWorld]:
    return [
        build_fixed_world(float(rho), world_seed)
        for rho in RHO_VALUES
        for world_seed in range(int(N_WORLD_SEEDS))
    ]


def interval_cdf(thresholds: np.ndarray, lower: float, upper: float) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=float)
    lower = float(lower)
    upper = float(upper)
    if upper < lower:
        raise ValueError("Interval upper bound must be at least the lower bound.")
    if np.isclose(upper, lower):
        return (thresholds >= upper).astype(float)
    return np.clip((thresholds - lower) / (upper - lower), 0.0, 1.0)


def theorem_distribution_from_intervals(
    interval_lower: np.ndarray,
    interval_upper: np.ndarray,
    *,
    alpha: float,
    m_total: int,
) -> np.ndarray:
    interval_lower = np.asarray(interval_lower, dtype=float)
    interval_upper = np.asarray(interval_upper, dtype=float)
    if interval_lower.ndim != 1 or interval_upper.ndim != 1:
        raise ValueError("Interval bounds must be one-dimensional arrays.")
    if len(interval_lower) == 0 or len(interval_lower) != len(interval_upper):
        raise ValueError("Interval bounds must be nonempty and have equal length.")
    if np.any(interval_upper < interval_lower):
        raise ValueError("Interval upper bounds must be at least lower bounds.")

    n_anomaly = len(interval_upper)
    thresholds = alpha * np.arange(1, n_anomaly + 1) / m_total
    category_probs = []
    for lower, upper in zip(interval_lower, interval_upper, strict=True):
        cumulative = interval_cdf(thresholds, lower, upper)
        probs = np.empty(n_anomaly + 1, dtype=float)
        probs[0] = cumulative[0]
        probs[1:n_anomaly] = np.diff(cumulative)
        probs[n_anomaly] = 1.0 - cumulative[-1]
        probs = np.clip(probs, 0.0, 1.0)
        total = float(np.sum(probs))
        if total <= 0.0:
            raise ValueError("Invalid interval category probabilities.")
        category_probs.append(probs / total)

    zero_state = (0,) * (n_anomaly + 1)
    state_probs: dict[tuple[int, ...], float] = {zero_state: 1.0}
    for probs in category_probs:
        next_state_probs: dict[tuple[int, ...], float] = {}
        for state, state_prob in state_probs.items():
            for category, category_prob in enumerate(probs):
                if category_prob <= 0.0:
                    continue
                updated = list(state)
                updated[category] += 1
                updated_state = tuple(updated)
                next_state_probs[updated_state] = (
                    next_state_probs.get(updated_state, 0.0)
                    + state_prob * float(category_prob)
                )
        state_probs = next_state_probs

    discovery_probs = np.zeros(n_anomaly + 1, dtype=float)
    for state, state_prob in state_probs.items():
        cumulative_count = 0
        discoveries = 0
        for rank in range(1, n_anomaly + 1):
            cumulative_count += state[rank - 1]
            if cumulative_count >= rank:
                discoveries = rank
        discovery_probs[discoveries] += state_prob

    return discovery_probs / float(np.sum(discovery_probs))


def theorem_distribution_from_interval_upper(
    interval_upper: np.ndarray,
    *,
    alpha: float,
    m_total: int,
) -> np.ndarray:
    interval_upper = np.asarray(interval_upper, dtype=float)
    return theorem_distribution_from_intervals(
        np.zeros_like(interval_upper),
        interval_upper,
        alpha=alpha,
        m_total=m_total,
    )


def simulate_randomized_discovery_counts(
    interval_lower: np.ndarray,
    interval_upper: np.ndarray,
    *,
    alpha: float,
    m_total: int,
    n_randomizations: int,
    seed_offset: int,
) -> np.ndarray:
    interval_lower = np.asarray(interval_lower, dtype=float)
    interval_upper = np.asarray(interval_upper, dtype=float)
    counts = np.empty(int(n_randomizations), dtype=int)
    rng = rng_for(seed_offset, 20)
    start = 0
    while start < int(n_randomizations):
        stop = min(start + int(SIMULATION_BATCH_SIZE), int(n_randomizations))
        uniforms = rng.random((stop - start, len(interval_upper)))
        p_values = interval_lower[None, :] + uniforms * (
            interval_upper - interval_lower
        )[None, :]
        counts[start:stop] = bh_discovery_counts_for_rows(
            p_values,
            alpha=alpha,
            m_total=m_total,
        )
        start = stop
    return counts


def rank_interval_ratio(interval_upper: np.ndarray, alpha: float, m_total: int) -> float:
    sorted_upper = np.sort(np.asarray(interval_upper, dtype=float), kind="mergesort")
    ranks = np.arange(1, len(sorted_upper) + 1)
    return float(np.min(sorted_upper / (alpha * ranks / m_total)))


def distribution_moments(probabilities: np.ndarray) -> tuple[float, float]:
    probabilities = np.asarray(probabilities, dtype=float)
    values = np.arange(len(probabilities), dtype=float)
    mean = float(np.sum(values * probabilities))
    variance = float(np.sum((values - mean) ** 2 * probabilities))
    return mean, variance


def summarize_world(
    world: FixedWorld,
) -> tuple[dict[str, float | int | str | bool], list[dict[str, float | int | str]]]:
    anomaly_lower = world.pvalue_lower[world.y_true]
    anomaly_upper = world.pvalue_upper[world.y_true]
    inlier_lower = world.pvalue_lower[~world.y_true]
    theorem_probs = theorem_distribution_from_intervals(
        anomaly_lower,
        anomaly_upper,
        alpha=float(ALPHA),
        m_total=int(M),
    )
    counts = simulate_randomized_discovery_counts(
        anomaly_lower,
        anomaly_upper,
        alpha=float(ALPHA),
        m_total=int(M),
        n_randomizations=int(N_RANDOMIZATIONS),
        seed_offset=rho_code(world.rho) + 10_000 * int(world.world_seed),
    )
    observed_counts = np.bincount(counts, minlength=len(theorem_probs)).astype(float)
    observed_probs = observed_counts / float(np.sum(observed_counts))
    theorem_mean, theorem_variance = distribution_moments(theorem_probs)
    observed_mean = float(np.mean(counts))
    observed_variance = float(np.var(counts))
    ratio = rank_interval_ratio(anomaly_upper, float(ALPHA), int(M))
    total_calib_weight = float(np.sum(world.calib_weights))

    summary_row = {
        "summary_version": SUMMARY_VERSION,
        "world_id": world.world_id,
        "rho": float(world.rho),
        "world_seed": int(world.world_seed),
        "alpha": float(ALPHA),
        "n_cal": int(N_CAL),
        "m": int(M),
        "n_anomaly": int(np.sum(world.y_true)),
        "n_inlier": int(np.sum(~world.y_true)),
        "n_randomizations": int(N_RANDOMIZATIONS),
        "total_calib_weight": total_calib_weight,
        "calib_ess": effective_sample_size(world.calib_weights),
        "max_normalized_calib_weight": float(
            np.max(world.calib_weights) / total_calib_weight
        ),
        "min_anomaly_interval_lower": float(np.min(anomaly_lower)),
        "max_anomaly_interval_lower": float(np.max(anomaly_lower)),
        "min_anomaly_interval_upper": float(np.min(anomaly_upper)),
        "max_anomaly_interval_upper": float(np.max(anomaly_upper)),
        "mean_anomaly_interval_width": float(np.mean(anomaly_upper - anomaly_lower)),
        "max_anomaly_interval_width": float(np.max(anomaly_upper - anomaly_lower)),
        "rank_interval_ratio": ratio,
        "log10_rank_interval_ratio": float(np.log10(ratio)),
        "min_inlier_interval_lower": float(np.min(inlier_lower)),
        "max_inlier_interval_lower": float(np.max(inlier_lower)),
        "inliers_nonrejectable": bool(np.min(inlier_lower) > float(ALPHA)),
        "theorem_miss_probability": float(theorem_probs[0]),
        "observed_miss_probability": float(observed_probs[0]),
        "miss_probability_error": float(observed_probs[0] - theorem_probs[0]),
        "theorem_mean_discoveries": theorem_mean,
        "observed_mean_discoveries": observed_mean,
        "theorem_variance_discoveries": theorem_variance,
        "observed_variance_discoveries": observed_variance,
    }

    distribution_rows = []
    for discoveries, theorem_probability in enumerate(theorem_probs):
        distribution_rows.append(
            {
                "summary_version": SUMMARY_VERSION,
                "world_id": world.world_id,
                "rho": float(world.rho),
                "world_seed": int(world.world_seed),
                "discoveries": int(discoveries),
                "theorem_probability": float(theorem_probability),
                "observed_probability": float(observed_probs[discoveries]),
                "observed_count": int(observed_counts[discoveries]),
                "n_randomizations": int(N_RANDOMIZATIONS),
            }
        )

    return summary_row, distribution_rows


def build_results(
    worlds: list[FixedWorld],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    distribution_rows = []
    total_worlds = len(worlds)
    for idx, world in enumerate(worlds, start=1):
        summary_row, world_distribution_rows = summarize_world(world)
        summary_rows.append(summary_row)
        distribution_rows.extend(world_distribution_rows)
        if idx % max(1, int(N_WORLD_SEEDS)) == 0 or idx == total_worlds:
            print(f"completed Figure 3 fixed world {idx}/{total_worlds}", flush=True)

    summary = pd.DataFrame(summary_rows)
    distribution = pd.DataFrame(distribution_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    distribution.to_csv(DISTRIBUTION_PATH, index=False)
    return summary, distribution


def summaries_are_current() -> bool:
    if not SUMMARY_PATH.exists() or not DISTRIBUTION_PATH.exists():
        return False
    summary_header = pd.read_csv(SUMMARY_PATH, nrows=1)
    distribution_header = pd.read_csv(DISTRIBUTION_PATH, nrows=1)
    if summary_header.empty or distribution_header.empty:
        return False
    return (
        str(summary_header["summary_version"].iloc[0]) == SUMMARY_VERSION
        and str(distribution_header["summary_version"].iloc[0]) == SUMMARY_VERSION
    )


def load_or_build_results(
    worlds: list[FixedWorld],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summaries_are_current():
        print("loading existing Figure 3 frontier summaries", flush=True)
        return pd.read_csv(SUMMARY_PATH), pd.read_csv(DISTRIBUTION_PATH)
    return build_results(worlds)


def validate_results(summary: pd.DataFrame, distribution: pd.DataFrame) -> None:
    expected_pairs = {
        (float(rho), int(world_seed))
        for rho in RHO_VALUES
        for world_seed in range(int(N_WORLD_SEEDS))
    }
    observed_pairs = {
        (float(row.rho), int(row.world_seed))
        for row in summary.itertuples(index=False)
    }
    if observed_pairs != expected_pairs:
        raise RuntimeError(
            "Unexpected Figure 3 fixed worlds: "
            f"missing={len(expected_pairs - observed_pairs)}, "
            f"unexpected={len(observed_pairs - expected_pairs)}."
        )
    if summary.duplicated(["rho", "world_seed"]).any():
        raise RuntimeError("Figure 3 summary contains duplicate fixed worlds.")
    if not bool(summary["inliers_nonrejectable"].all()):
        raise RuntimeError("Figure 3 inlier p-value intervals must stay above alpha.")
    probability_columns = [
        "theorem_miss_probability",
        "observed_miss_probability",
    ]
    for column in probability_columns:
        if not ((summary[column] >= 0.0) & (summary[column] <= 1.0)).all():
            raise RuntimeError(f"Summary probability column out of range: {column}.")
    for world_id, block in distribution.groupby("world_id"):
        theorem_sum = float(block["theorem_probability"].sum())
        observed_sum = float(block["observed_probability"].sum())
        if not np.isclose(theorem_sum, 1.0, atol=1e-10):
            raise RuntimeError(f"Theorem distribution does not sum to 1: {world_id}.")
        if not np.isclose(observed_sum, 1.0, atol=1e-10):
            raise RuntimeError(f"Observed distribution does not sum to 1: {world_id}.")


def world_lookup(worlds: list[FixedWorld]) -> dict[str, FixedWorld]:
    return {world.world_id: world for world in worlds}


def representative_worlds(
    worlds: list[FixedWorld],
    summary: pd.DataFrame,
) -> list[FixedWorld]:
    lookup = world_lookup(worlds)
    rho_values = sorted(float(rho) for rho in summary["rho"].unique())
    target_rhos = [
        rho_values[0],
        rho_values[len(rho_values) // 2],
        rho_values[-1],
    ]
    reps = []
    for target_rho in target_rhos:
        block = summary[np.isclose(summary["rho"], target_rho)].copy()
        median_x = float(block["log10_rank_interval_ratio"].median())
        idx = (block["log10_rank_interval_ratio"] - median_x).abs().idxmin()
        reps.append(lookup[str(summary.loc[idx, "world_id"])])
    return reps


def frontier_curve(summary: pd.DataFrame, column: str) -> pd.DataFrame:
    x_values = summary["log10_rank_interval_ratio"].to_numpy(dtype=float)
    if np.isclose(float(np.min(x_values)), float(np.max(x_values))):
        return pd.DataFrame(
            [
                {
                    "x": float(np.mean(x_values)),
                    "mean": float(summary[column].mean()),
                    "q10": float(summary[column].quantile(0.10)),
                    "q90": float(summary[column].quantile(0.90)),
                    "count": int(len(summary)),
                }
            ]
        )
    edges = np.linspace(
        float(np.min(x_values)),
        float(np.max(x_values)),
        min(int(FRONTIER_BINS), max(2, len(summary))) + 1,
    )
    bin_idx = np.searchsorted(edges, x_values, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    rows = []
    for idx in range(len(edges) - 1):
        block = summary.loc[bin_idx == idx]
        if block.empty:
            continue
        rows.append(
            {
                "x": float(block["log10_rank_interval_ratio"].mean()),
                "mean": float(block[column].mean()),
                "q10": float(block[column].quantile(0.10)),
                "q90": float(block[column].quantile(0.90)),
                "count": int(len(block)),
            }
        )
    return pd.DataFrame(rows)


def plot_representative_intervals(
    ax: plt.Axes,
    worlds: list[FixedWorld],
    summary: pd.DataFrame,
) -> None:
    reps = representative_worlds(worlds, summary)
    style_keys = ["equal", "moderate", "high"]
    ranks = np.arange(1, int(N_ANOMALY) + 1)
    thresholds = float(ALPHA) * ranks / int(M)
    y_floor = max(float(ALPHA) / int(M) * 0.25, 1e-8)
    offsets = np.linspace(-0.18, 0.18, len(reps))

    for offset, style_key, world in zip(offsets, style_keys, reps, strict=True):
        lower = world.pvalue_lower[world.y_true]
        upper = world.pvalue_upper[world.y_true]
        order = np.argsort(upper, kind="mergesort")
        lower = lower[order]
        upper = upper[order]
        row = summary[summary["world_id"].eq(world.world_id)].iloc[0]
        x = ranks + offset
        color = COLORS[style_key]
        label = (
            rf"$\rho={world.rho:.1f}$, "
            rf"$N_{{eff}}={float(row.calib_ess):.0f}$"
        )
        ax.vlines(
            x,
            np.maximum(lower, y_floor),
            upper,
            color=color,
            alpha=0.72,
            linewidth=2.0,
            label=label,
        )
        ax.scatter(x, upper, color=color, s=22, zorder=3)

    ax.plot(
        ranks,
        thresholds,
        color=COLORS["threshold"],
        linestyle="--",
        linewidth=1.3,
        marker="o",
        markersize=3,
        label="BH thresholds",
    )
    ax.set_yscale("log")
    ax.set_xlim(0.5, int(N_ANOMALY) + 0.5)
    ax.set_ylim(y_floor, 1.05)
    ax.set_xlabel("anomaly rank")
    ax.set_ylabel("randomized p-value interval")
    ax.set_title("A. Fixed-world randomized intervals")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")


def plot_frontier_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    observed_column: str,
    theorem_column: str,
    ylabel: str,
    title: str,
) -> None:
    x = summary["log10_rank_interval_ratio"].to_numpy(dtype=float)
    observed = summary[observed_column].to_numpy(dtype=float)
    theorem_curve = frontier_curve(summary, theorem_column)
    observed_curve = frontier_curve(summary, observed_column)

    scatter = ax.scatter(
        x,
        observed,
        c=summary["rho"].to_numpy(dtype=float),
        cmap="viridis",
        s=16,
        alpha=0.35,
        linewidths=0,
        label="observed fixed worlds",
    )
    ax.fill_between(
        theorem_curve["x"],
        theorem_curve["q10"],
        theorem_curve["q90"],
        color=COLORS["theorem"],
        alpha=0.16,
        linewidth=0,
        label="theorem 10-90%",
    )
    ax.plot(
        theorem_curve["x"],
        theorem_curve["mean"],
        color=COLORS["theorem"],
        linewidth=2.0,
        label="theorem mean",
    )
    ax.plot(
        observed_curve["x"],
        observed_curve["mean"],
        color=COLORS["observed"],
        linewidth=1.6,
        linestyle=":",
        label="observed mean",
    )
    ax.set_xlabel(r"$\log_{10}$ rank interval ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")
    return scatter


def plot_calibration_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    theorem = summary["theorem_miss_probability"].to_numpy(dtype=float)
    observed = summary["observed_miss_probability"].to_numpy(dtype=float)
    ax.scatter(
        theorem,
        observed,
        c=summary["rho"].to_numpy(dtype=float),
        cmap="viridis",
        s=18,
        alpha=0.45,
        linewidths=0,
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1.1)
    max_error = float(np.max(np.abs(observed - theorem)))
    ax.text(
        0.04,
        0.94,
        f"max abs error = {max_error:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("theorem miss probability")
    ax.set_ylabel("observed miss probability")
    ax.set_title("D. Conditional theorem calibration")
    ax.grid(alpha=0.18, linewidth=0.6)


def plot_figure(
    worlds: list[FixedWorld],
    summary: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.0), constrained_layout=True)
    plot_representative_intervals(axes[0, 0], worlds, summary)
    scatter = plot_frontier_panel(
        axes[0, 1],
        summary,
        observed_column="observed_miss_probability",
        theorem_column="theorem_miss_probability",
        ylabel="miss probability",
        title="B. Miss probability frontier",
    )
    plot_frontier_panel(
        axes[1, 0],
        summary,
        observed_column="observed_variance_discoveries",
        theorem_column="theorem_variance_discoveries",
        ylabel="variance of BH discoveries",
        title="C. Discovery variance frontier",
    )
    plot_calibration_panel(axes[1, 1], summary)
    cbar = fig.colorbar(scatter, ax=axes[:, 1], fraction=0.035, pad=0.02)
    cbar.set_label(r"weight shift $\rho$")
    fig.suptitle(
        "Randomized WEDF instability is predicted by fixed interval geometry",
        fontsize=14,
    )
    fig.savefig(FIGURE_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def export_interval_tikz(worlds: list[FixedWorld], summary: pd.DataFrame) -> None:
    rows = []
    reps = representative_worlds(worlds, summary)
    ranks = np.arange(1, int(N_ANOMALY) + 1)
    thresholds = float(ALPHA) * ranks / int(M)
    for plot_order, world in enumerate(reps, start=1):
        lower = world.pvalue_lower[world.y_true]
        upper = world.pvalue_upper[world.y_true]
        order = np.argsort(upper, kind="mergesort")
        for rank, low, high in zip(ranks, lower[order], upper[order], strict=True):
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure3",
                    "panel": "A",
                    "panel_title": "Fixed-world randomized intervals",
                    "plot_order": plot_order,
                    "group": "anomaly_interval",
                    "method": "randomized_wedf",
                    "world_id": world.world_id,
                    "rho": float(world.rho),
                    "world_seed": int(world.world_seed),
                    "rank": int(rank),
                    "x": int(rank),
                    "y": float(high),
                    "y_lower": float(low),
                    "value": float(high),
                    "probability": "",
                    "count": "",
                    "style_key": f"rho_{rho_code(world.rho)}",
                    "label": rf"$\rho={world.rho:.1f}$",
                }
            )
        for rank, threshold in zip(ranks, thresholds, strict=True):
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure3",
                    "panel": "A",
                    "panel_title": "Fixed-world randomized intervals",
                    "plot_order": plot_order,
                    "group": "bh_threshold",
                    "method": "bh",
                    "world_id": world.world_id,
                    "rho": float(world.rho),
                    "world_seed": int(world.world_seed),
                    "rank": int(rank),
                    "x": int(rank),
                    "y": float(threshold),
                    "y_lower": "",
                    "value": float(threshold),
                    "probability": "",
                    "count": "",
                    "style_key": "bh_threshold",
                    "label": r"$\alpha r / m$",
                }
            )
    pd.DataFrame(rows).to_csv(INTERVAL_TIKZ_PATH, index=False)


def export_frontier_tikz(summary: pd.DataFrame) -> None:
    rows = []
    panel_specs = [
        (
            "B",
            "Miss probability frontier",
            "miss_probability",
            "observed_miss_probability",
            "theorem_miss_probability",
        ),
        (
            "C",
            "Discovery variance frontier",
            "discovery_variance",
            "observed_variance_discoveries",
            "theorem_variance_discoveries",
        ),
        (
            "D",
            "Conditional theorem calibration",
            "miss_calibration",
            "observed_miss_probability",
            "theorem_miss_probability",
        ),
    ]
    for panel, title, group, observed_col, theorem_col in panel_specs:
        for row in summary.sort_values(["rho", "world_seed"]).itertuples(index=False):
            if panel == "D":
                x = float(getattr(row, theorem_col))
                y = float(getattr(row, observed_col))
            else:
                x = float(row.log10_rank_interval_ratio)
                y = float(getattr(row, observed_col))
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure3",
                    "panel": panel,
                    "panel_title": title,
                    "plot_order": int(row.world_seed) + 1,
                    "group": group,
                    "method": "observed_randomization",
                    "world_id": str(row.world_id),
                    "rho": float(row.rho),
                    "world_seed": int(row.world_seed),
                    "rank": "",
                    "x": x,
                    "y": y,
                    "value": y,
                    "probability": y if "probability" in observed_col else "",
                    "count": int(row.n_randomizations),
                    "style_key": "observed",
                    "label": "observed randomization",
                }
            )
            if panel != "D":
                rows.append(
                    {
                        "export_version": TIKZ_EXPORT_VERSION,
                        "figure": "figure3",
                        "panel": panel,
                        "panel_title": title,
                        "plot_order": int(row.world_seed) + 1,
                        "group": group,
                        "method": "conditional_interval_theorem",
                        "world_id": str(row.world_id),
                        "rho": float(row.rho),
                        "world_seed": int(row.world_seed),
                        "rank": "",
                        "x": float(row.log10_rank_interval_ratio),
                        "y": float(getattr(row, theorem_col)),
                        "value": float(getattr(row, theorem_col)),
                        "probability": (
                            float(getattr(row, theorem_col))
                            if "probability" in theorem_col
                            else ""
                        ),
                        "count": "",
                        "style_key": "theorem",
                        "label": "conditional interval theorem",
                    }
                )
    pd.DataFrame(rows).to_csv(DISTRIBUTION_TIKZ_PATH, index=False)


def write_rationale() -> None:
    RATIONALE_PATH.write_text(
        """# Figure 3 Rationale: Randomization Instability Frontier

This figure is a theorem-validation experiment, not a benchmark. Each point is
one fixed world: calibration scores, test scores, calibration weights, and test
weights are frozen. Only the uniforms inside the randomized weighted conformal
p-values are resampled.

Perfect score separation is enforced in every fixed world. Inliers are placed
below the calibration range, so their randomized p-values stay above `alpha` and
cannot drive BH discoveries; high-shift inlier weights are capped only to enforce
this guardrail. Anomalies are placed above the calibration range, so their
intervals are determined only by their weighted self-atoms.

The central diagnostic is the rank interval ratio,

```text
min_r U_(r) / (alpha * r / m),
```

where `U_(r)` is the `r`-th smallest anomaly interval upper endpoint. Larger
values mean the randomization intervals are wide relative to the BH scale. The
frontier panels show that the conditional interval theorem predicts both miss
probability and discovery-count variance across fixed weight worlds.
""",
        encoding="utf-8",
    )


def export_tikz_csvs(worlds: list[FixedWorld], summary: pd.DataFrame) -> None:
    export_interval_tikz(worlds, summary)
    export_frontier_tikz(summary)


def validate_outputs() -> None:
    for path in [
        FIGURE_PATH,
        SUMMARY_PATH,
        DISTRIBUTION_PATH,
        INTERVAL_TIKZ_PATH,
        DISTRIBUTION_TIKZ_PATH,
        RATIONALE_PATH,
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
    worlds = build_fixed_worlds()
    summary, distribution = load_or_build_results(worlds)
    validate_results(summary, distribution)
    plot_figure(worlds, summary)
    export_tikz_csvs(worlds, summary)
    write_rationale()
    validate_outputs()
    print(SUMMARY_PATH)
    print(DISTRIBUTION_PATH)
    print(FIGURE_PATH)
    print(INTERVAL_TIKZ_PATH)
    print(DISTRIBUTION_TIKZ_PATH)
    print(RATIONALE_PATH)


if __name__ == "__main__":
    main()
