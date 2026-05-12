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

SUMMARY_VERSION = "randomized-instability-v1"
TIKZ_EXPORT_VERSION = "tikz-v1"
BASE_SEED = 20260512

ALPHA = 0.10
N_CAL = 500
M = 1000
N_ANOMALY = 10
DELTA_SCORE = 1.0
D = 10
SHIFTED_RHO = 1.5
N_RANDOMIZATIONS = 50_000
SIMULATION_BATCH_SIZE = 10_000

SCENARIOS = [
    {
        "name": "equal_weights",
        "label": "Equal weights",
        "panel_prefix": "A",
        "distribution_panel_prefix": "B",
    },
    {
        "name": "shifted_weights",
        "label": "Frozen shifted weights",
        "panel_prefix": "C",
        "distribution_panel_prefix": "D",
    },
]

COLORS = {
    "interval": "#7c3aed",
    "threshold": "#111111",
    "inlier": "#2563eb",
    "observed": "#c2410c",
    "theorem": "#0f766e",
}


@dataclass(frozen=True)
class FixedSetup:
    scenario: str
    label: str
    rho: float
    calib_scores: np.ndarray
    test_scores: np.ndarray
    y_true: np.ndarray
    calib_weights: np.ndarray
    test_weights: np.ndarray
    pvalue_lower: np.ndarray
    pvalue_upper: np.ndarray


def rng_for(*values: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *values]))


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights**2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(weights) ** 2 / denominator)


def bh_discovery_count(p_values: np.ndarray, alpha: float) -> int:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="mergesort")
    sorted_p = p_values[order]
    thresholds = alpha * np.arange(1, len(p_values) + 1) / len(p_values)
    passed = sorted_p <= thresholds
    if not bool(np.any(passed)):
        return 0
    return int(np.flatnonzero(passed)[-1] + 1)


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


def scenario_index(name: str) -> int:
    for idx, scenario in enumerate(SCENARIOS):
        if str(scenario["name"]) == name:
            return idx
    raise ValueError(f"Unknown Figure 3 scenario: {name!r}.")


def build_fixed_setup(scenario: dict[str, str]) -> FixedSetup:
    scenario_name = str(scenario["name"])
    idx = scenario_index(scenario_name)
    score_rng = rng_for(idx, 0)
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

    if scenario_name == "equal_weights":
        rho = 0.0
        calib_weights = np.ones(int(N_CAL), dtype=float)
        test_weights = np.ones(int(M), dtype=float)
    elif scenario_name == "shifted_weights":
        rho = float(SHIFTED_RHO)
        weight_rng = rng_for(idx, 1)
        calib_x = weight_rng.normal(0.0, 1.0, size=(int(N_CAL), int(D)))
        inlier_x = weight_rng.normal(0.0, 1.0, size=(n_inlier, int(D)))
        anomaly_x = weight_rng.normal(0.0, 1.0, size=(int(N_ANOMALY), int(D)))
        inlier_x[:, 0] += rho
        anomaly_x[:, 0] += rho
        test_x = np.vstack([inlier_x, anomaly_x])
        calib_weights = np.exp(rho * calib_x[:, 0] - 0.5 * rho**2)
        test_weights = np.exp(rho * test_x[:, 0] - 0.5 * rho**2)
    else:
        raise ValueError(f"Unsupported scenario: {scenario_name!r}.")

    sorted_scores, _, suffix_weights = sorted_calibration(calib_scores, calib_weights)
    lower, upper = randomized_pvalue_intervals(
        sorted_scores,
        suffix_weights,
        float(np.sum(calib_weights)),
        test_scores,
        test_weights,
    )
    return FixedSetup(
        scenario=scenario_name,
        label=str(scenario["label"]),
        rho=rho,
        calib_scores=calib_scores,
        test_scores=test_scores,
        y_true=y_true,
        calib_weights=calib_weights,
        test_weights=test_weights,
        pvalue_lower=lower,
        pvalue_upper=upper,
    )


def build_fixed_setups() -> list[FixedSetup]:
    return [build_fixed_setup(scenario) for scenario in SCENARIOS]


def theorem_distribution_from_interval_upper(
    interval_upper: np.ndarray,
    *,
    alpha: float,
    m_total: int,
) -> np.ndarray:
    interval_upper = np.asarray(interval_upper, dtype=float)
    if interval_upper.ndim != 1 or len(interval_upper) == 0:
        raise ValueError("interval_upper must be a nonempty one-dimensional array.")
    if np.any(interval_upper <= 0.0):
        raise ValueError("All randomized anomaly interval upper bounds must be positive.")

    n_anomaly = len(interval_upper)
    thresholds = alpha * np.arange(1, n_anomaly + 1) / m_total
    category_probs = []
    for upper in interval_upper:
        cumulative = np.minimum(thresholds, upper) / upper
        probs = np.empty(n_anomaly + 1, dtype=float)
        probs[0] = cumulative[0]
        probs[1:n_anomaly] = np.diff(cumulative)
        probs[n_anomaly] = 1.0 - cumulative[-1]
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / float(np.sum(probs))
        category_probs.append(probs)

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


def simulate_randomized_discovery_counts(
    interval_upper: np.ndarray,
    *,
    alpha: float,
    m_total: int,
    n_randomizations: int,
    seed_offset: int,
) -> np.ndarray:
    interval_upper = np.asarray(interval_upper, dtype=float)
    counts = np.empty(int(n_randomizations), dtype=int)
    rng = rng_for(seed_offset, 20)
    start = 0
    while start < int(n_randomizations):
        stop = min(start + int(SIMULATION_BATCH_SIZE), int(n_randomizations))
        uniforms = rng.random((stop - start, len(interval_upper)))
        p_values = uniforms * interval_upper[None, :]
        counts[start:stop] = bh_discovery_counts_for_rows(
            p_values,
            alpha=alpha,
            m_total=m_total,
        )
        start = stop
    return counts


def build_results(
    setups: list[FixedSetup],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    distribution_rows = []
    for setup in setups:
        idx = scenario_index(setup.scenario)
        anomaly_upper = setup.pvalue_upper[setup.y_true]
        anomaly_lower = setup.pvalue_lower[setup.y_true]
        inlier_lower = setup.pvalue_lower[~setup.y_true]
        theorem_probs = theorem_distribution_from_interval_upper(
            anomaly_upper,
            alpha=float(ALPHA),
            m_total=int(M),
        )
        counts = simulate_randomized_discovery_counts(
            anomaly_upper,
            alpha=float(ALPHA),
            m_total=int(M),
            n_randomizations=int(N_RANDOMIZATIONS),
            seed_offset=idx,
        )
        observed_counts = np.bincount(counts, minlength=len(theorem_probs)).astype(float)
        observed_probs = observed_counts / float(np.sum(observed_counts))
        discovery_values = np.arange(len(theorem_probs), dtype=float)
        theorem_mean = float(np.sum(discovery_values * theorem_probs))
        observed_mean = float(np.mean(counts))
        theorem_variance = float(
            np.sum((discovery_values - theorem_mean) ** 2 * theorem_probs)
        )
        observed_variance = float(np.var(counts))
        theorem_miss = float(theorem_probs[0])
        observed_miss = float(observed_probs[0])

        for discoveries, theorem_probability in enumerate(theorem_probs):
            distribution_rows.append(
                {
                    "summary_version": SUMMARY_VERSION,
                    "scenario": setup.scenario,
                    "label": setup.label,
                    "discoveries": discoveries,
                    "theorem_probability": float(theorem_probability),
                    "observed_probability": float(observed_probs[discoveries]),
                    "observed_count": int(observed_counts[discoveries]),
                    "n_randomizations": int(N_RANDOMIZATIONS),
                }
            )

        summary_rows.append(
            {
                "summary_version": SUMMARY_VERSION,
                "scenario": setup.scenario,
                "label": setup.label,
                "alpha": float(ALPHA),
                "n_cal": int(N_CAL),
                "m": int(M),
                "n_anomaly": int(np.sum(setup.y_true)),
                "n_inlier": int(np.sum(~setup.y_true)),
                "rho": float(setup.rho),
                "n_randomizations": int(N_RANDOMIZATIONS),
                "total_calib_weight": float(np.sum(setup.calib_weights)),
                "calib_ess": effective_sample_size(setup.calib_weights),
                "min_anomaly_interval_lower": float(np.min(anomaly_lower)),
                "max_anomaly_interval_lower": float(np.max(anomaly_lower)),
                "min_anomaly_interval_upper": float(np.min(anomaly_upper)),
                "max_anomaly_interval_upper": float(np.max(anomaly_upper)),
                "min_inlier_interval_lower": float(np.min(inlier_lower)),
                "max_inlier_interval_lower": float(np.max(inlier_lower)),
                "inliers_nonrejectable": bool(np.min(inlier_lower) > float(ALPHA)),
                "theorem_miss_probability": theorem_miss,
                "observed_miss_probability": observed_miss,
                "miss_probability_error": observed_miss - theorem_miss,
                "theorem_mean_discoveries": theorem_mean,
                "observed_mean_discoveries": observed_mean,
                "theorem_variance_discoveries": theorem_variance,
                "observed_variance_discoveries": observed_variance,
            }
        )

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
    setups: list[FixedSetup],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summaries_are_current():
        print("loading existing Figure 3 summaries", flush=True)
        return pd.read_csv(SUMMARY_PATH), pd.read_csv(DISTRIBUTION_PATH)
    return build_results(setups)


def validate_results(summary: pd.DataFrame, distribution: pd.DataFrame) -> None:
    expected_scenarios = {str(scenario["name"]) for scenario in SCENARIOS}
    observed_scenarios = set(summary["scenario"])
    if observed_scenarios != expected_scenarios:
        raise RuntimeError(
            "Unexpected Figure 3 summary scenarios: "
            f"missing={expected_scenarios - observed_scenarios}, "
            f"unexpected={observed_scenarios - expected_scenarios}."
        )
    if not bool(summary["inliers_nonrejectable"].all()):
        raise RuntimeError("Figure 3 inlier p-value intervals must stay above alpha.")
    probability_columns = [
        "theorem_miss_probability",
        "observed_miss_probability",
    ]
    for column in probability_columns:
        if not ((summary[column] >= 0.0) & (summary[column] <= 1.0)).all():
            raise RuntimeError(f"Summary probability column out of range: {column}.")
    for scenario, block in distribution.groupby("scenario"):
        theorem_sum = float(block["theorem_probability"].sum())
        observed_sum = float(block["observed_probability"].sum())
        if not np.isclose(theorem_sum, 1.0, atol=1e-10):
            raise RuntimeError(f"Theorem distribution does not sum to 1: {scenario}.")
        if not np.isclose(observed_sum, 1.0, atol=1e-10):
            raise RuntimeError(f"Observed distribution does not sum to 1: {scenario}.")


def plot_interval_panel(
    ax: plt.Axes,
    setup: FixedSetup,
    panel_prefix: str,
) -> None:
    anomaly_upper = np.sort(setup.pvalue_upper[setup.y_true], kind="mergesort")
    ranks = np.arange(1, len(anomaly_upper) + 1)
    thresholds = float(ALPHA) * ranks / int(M)
    min_inlier_lower = float(np.min(setup.pvalue_lower[~setup.y_true]))
    y_floor = max(float(ALPHA) / int(M) * 0.25, 1e-8)

    ax.vlines(
        ranks,
        y_floor,
        anomaly_upper,
        color=COLORS["interval"],
        alpha=0.75,
        linewidth=2.0,
        label=r"randomized interval $[0,a_j]$",
    )
    ax.scatter(
        ranks,
        anomaly_upper,
        s=32,
        color=COLORS["interval"],
        zorder=3,
    )
    ax.plot(
        ranks,
        thresholds,
        color=COLORS["threshold"],
        linestyle="--",
        linewidth=1.3,
        marker="o",
        markersize=3,
        label="BH rank thresholds",
    )
    ax.axhline(
        min_inlier_lower,
        color=COLORS["inlier"],
        linestyle=":",
        linewidth=1.4,
        label="lowest inlier interval",
    )
    ax.set_yscale("log")
    ax.set_ylim(y_floor, 1.05)
    ax.set_xlim(0.5, len(anomaly_upper) + 0.5)
    ax.set_xlabel("anomaly rank")
    ax.set_ylabel("randomized p-value scale")
    ax.set_title(f"{panel_prefix}. {setup.label}: fixed intervals")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")


def plot_distribution_panel(
    ax: plt.Axes,
    distribution: pd.DataFrame,
    summary: pd.DataFrame,
    setup: FixedSetup,
    panel_prefix: str,
) -> None:
    block = distribution[distribution["scenario"].eq(setup.scenario)].sort_values(
        "discoveries"
    )
    summary_row = summary[summary["scenario"].eq(setup.scenario)].iloc[0]
    x = block["discoveries"].to_numpy(dtype=float)
    ax.bar(
        x,
        block["observed_probability"],
        width=0.75,
        color=COLORS["observed"],
        alpha=0.35,
        label="observed randomization",
    )
    ax.plot(
        x,
        block["theorem_probability"],
        color=COLORS["theorem"],
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="conditional interval theorem",
    )
    ax.text(
        0.97,
        0.92,
        "miss obs/theorem = "
        f"{summary_row.observed_miss_probability:.3f}/"
        f"{summary_row.theorem_miss_probability:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    ax.set_xlim(-0.6, float(np.max(x)) + 0.6)
    y_max = float(
        block[["observed_probability", "theorem_probability"]].max().max()
    )
    ax.set_ylim(0.0, max(0.08, y_max * 1.18))
    ax.set_xlabel("BH anomaly discoveries")
    ax.set_ylabel("probability")
    ax.set_title(f"{panel_prefix}. {setup.label}: discovery distribution")
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")


def plot_figure(
    setups: list[FixedSetup],
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4), constrained_layout=True)
    scenario_meta = {str(scenario["name"]): scenario for scenario in SCENARIOS}
    for row_idx, setup in enumerate(setups):
        meta = scenario_meta[setup.scenario]
        plot_interval_panel(axes[row_idx, 0], setup, str(meta["panel_prefix"]))
        plot_distribution_panel(
            axes[row_idx, 1],
            distribution,
            summary,
            setup,
            str(meta["distribution_panel_prefix"]),
        )
    fig.suptitle(
        "Randomized WEDF p-value instability under fixed scores and weights",
        fontsize=14,
    )
    fig.savefig(FIGURE_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def export_interval_tikz(setups: list[FixedSetup]) -> None:
    rows = []
    scenario_order = {str(scenario["name"]): idx + 1 for idx, scenario in enumerate(SCENARIOS)}
    for setup in setups:
        anomaly_upper = np.sort(setup.pvalue_upper[setup.y_true], kind="mergesort")
        ranks = np.arange(1, len(anomaly_upper) + 1)
        thresholds = float(ALPHA) * ranks / int(M)
        min_inlier_lower = float(np.min(setup.pvalue_lower[~setup.y_true]))
        plot_order = scenario_order[setup.scenario]
        for rank, upper, threshold in zip(ranks, anomaly_upper, thresholds, strict=True):
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure3",
                    "panel": "intervals",
                    "panel_title": "Fixed randomized p-value intervals",
                    "plot_order": plot_order,
                    "group": "anomaly_interval_upper",
                    "method": "randomized_wedf",
                    "scenario": setup.scenario,
                    "rank": int(rank),
                    "x": int(rank),
                    "y": float(upper),
                    "value": float(upper),
                    "probability": "",
                    "count": "",
                    "style_key": "interval_upper",
                    "label": setup.label,
                }
            )
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure3",
                    "panel": "intervals",
                    "panel_title": "Fixed randomized p-value intervals",
                    "plot_order": plot_order,
                    "group": "bh_threshold",
                    "method": "bh",
                    "scenario": setup.scenario,
                    "rank": int(rank),
                    "x": int(rank),
                    "y": float(threshold),
                    "value": float(threshold),
                    "probability": "",
                    "count": "",
                    "style_key": "bh_threshold",
                    "label": r"$\alpha r / m$",
                }
            )
        rows.append(
            {
                "export_version": TIKZ_EXPORT_VERSION,
                "figure": "figure3",
                "panel": "intervals",
                "panel_title": "Fixed randomized p-value intervals",
                "plot_order": plot_order,
                "group": "inlier_lower_reference",
                "method": "randomized_wedf",
                "scenario": setup.scenario,
                "rank": "",
                "x": "",
                "y": min_inlier_lower,
                "value": min_inlier_lower,
                "probability": "",
                "count": "",
                "style_key": "inlier_lower",
                "label": "lowest inlier interval",
            }
        )
    pd.DataFrame(rows).to_csv(INTERVAL_TIKZ_PATH, index=False)


def export_distribution_tikz(distribution: pd.DataFrame) -> None:
    rows = []
    scenario_order = {str(scenario["name"]): idx + 1 for idx, scenario in enumerate(SCENARIOS)}
    for row in distribution.sort_values(["scenario", "discoveries"]).itertuples(index=False):
        plot_order = scenario_order[str(row.scenario)]
        for method, probability in [
            ("observed_randomization", float(row.observed_probability)),
            ("conditional_interval_theorem", float(row.theorem_probability)),
        ]:
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure3",
                    "panel": "distribution",
                    "panel_title": "BH discovery distribution",
                    "plot_order": plot_order,
                    "group": "discovery_distribution",
                    "method": method,
                    "scenario": str(row.scenario),
                    "rank": int(row.discoveries),
                    "x": int(row.discoveries),
                    "y": probability,
                    "value": probability,
                    "probability": probability,
                    "count": int(row.observed_count) if method == "observed_randomization" else "",
                    "style_key": method,
                    "label": str(row.label),
                }
            )
    pd.DataFrame(rows).to_csv(DISTRIBUTION_TIKZ_PATH, index=False)


def write_rationale() -> None:
    RATIONALE_PATH.write_text(
        """# Figure 3 Rationale: Randomized P-Value Instability

This figure conditions on one fixed set of scores and weights, then resamples
only the uniforms used in randomized weighted conformal p-values. It isolates
the variability caused by randomizing the test self-atom.

For the anomaly points, scores are placed above the calibration range, so the
calibration tail mass is zero and the randomized WEDF p-value is uniform on
`[0, w_j / (W_cal + w_j)]`. For inliers, scores are placed below the calibration
range, so their randomized p-values stay above `alpha` and cannot drive BH.

The plotted theorem curve is conditional on the frozen intervals. Each anomaly
p-value is assigned to the BH threshold bins `alpha * k / m`; a dynamic program
over these independent categorical variables gives the exact distribution of
the BH anomaly discovery count. The observed histogram should match this curve
up to Monte Carlo error.

The summary also records observed and theorem means and variances of the BH
discovery count, which makes the randomization-driven variance inflation visible
without changing the fixed score separation or fixed weights.
""",
        encoding="utf-8",
    )


def export_tikz_csvs(setups: list[FixedSetup], distribution: pd.DataFrame) -> None:
    export_interval_tikz(setups)
    export_distribution_tikz(distribution)


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
    setups = build_fixed_setups()
    summary, distribution = load_or_build_results(setups)
    validate_results(summary, distribution)
    plot_figure(setups, summary, distribution)
    export_tikz_csvs(setups, distribution)
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
