from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


OUT_DIR = Path(__file__).resolve().parent
FIGURE_PATH = OUT_DIR / "figure4_clipping_frontier.png"
SUMMARY_PATH = OUT_DIR / "figure4_clipping_frontier_summary.csv"
TIKZ_PATH = OUT_DIR / "figure4_clipping_frontier_tikz.csv"
TABLE_PATH = OUT_DIR / "figure4_key_settings_table.csv"
RATIONALE_PATH = OUT_DIR / "RATIONALE.md"

SUMMARY_VERSION = "clipping-frontier-v2"
TIKZ_EXPORT_VERSION = "tikz-v2"
BASE_SEED = 20260512

RHO_VALUES = [0.5, 1.5, 3.0]
N_CAL = 500
M = 1000
N_SEEDS = 500
ALPHA = 0.10
CLIP_CAPS = [1, 1.5, 2, 3, 5, 8, 13, 21, 34, 55, 89, np.inf]
TAIL_PROBS = np.geomspace(1e-4, 0.5, 80)
EPSILON = 1e-300

RHO_COLORS = {
    0.5: "#2563eb",
    1.5: "#c2410c",
    3.0: "#0f766e",
}
FALLBACK_COLORS = ["#2563eb", "#c2410c", "#0f766e", "#7c3aed", "#111111"]

METRIC_LABELS = {
    "max_test_self_atom": "mean max shifted-null self-atom",
    "calib_ess_fraction": "mean calibration ESS / n",
    "oracle_tail_mismatch": "oracle shifted-null tail mismatch",
    "clipped_target_tv": "TV(Q, Q_c)",
    "frontier_oracle_tail_mismatch": "frontier: mismatch vs atom",
}


def rng_for(*values: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *values]))


def rho_code(rho: float) -> int:
    return int(round(float(rho) * 1000))


def cap_label(cap: float) -> str:
    return "unclipped" if np.isinf(float(cap)) else f"c={float(cap):g}"


def cap_plot_positions(caps: list[float]) -> dict[str, float]:
    finite_caps = [float(cap) for cap in caps if not np.isinf(float(cap))]
    if not finite_caps:
        return {cap_label(cap): float(idx) for idx, cap in enumerate(caps)}
    max_finite_x = float(np.log10(max(finite_caps)))
    return {
        cap_label(cap): (
            max_finite_x + 0.35 if np.isinf(float(cap)) else float(np.log10(float(cap)))
        )
        for cap in caps
    }


def rho_color(rho: float, index: int) -> str:
    return RHO_COLORS.get(float(rho), FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def density_ratio(z_values: np.ndarray, rho: float) -> np.ndarray:
    z_values = np.asarray(z_values, dtype=float)
    rho = float(rho)
    return np.exp(rho * z_values - 0.5 * rho**2)


def clipped_weights(
    z_values: np.ndarray,
    cap: float,
    rho: float,
) -> np.ndarray:
    weights = density_ratio(z_values, rho=rho)
    if np.isinf(float(cap)):
        return weights
    if float(cap) <= 0.0:
        raise ValueError("clip cap must be positive.")
    return np.minimum(weights, float(cap))


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights**2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(weights) ** 2 / denominator)


def tail_score_grid(
    tail_probs: np.ndarray = TAIL_PROBS,
    rho: float = 1.5,
) -> np.ndarray:
    tail_probs = np.asarray(tail_probs, dtype=float)
    if np.any((tail_probs <= 0.0) | (tail_probs >= 1.0)):
        raise ValueError("tail probabilities must lie in (0, 1).")
    return float(rho) + norm.isf(tail_probs)


def shifted_null_tail(scores: np.ndarray, rho: float) -> np.ndarray:
    return norm.sf(np.asarray(scores, dtype=float) - float(rho))


def clipping_threshold_z(cap: float, rho: float) -> float:
    cap = float(cap)
    rho = float(rho)
    if np.isinf(cap):
        return np.inf
    if cap <= 0.0:
        raise ValueError("clip cap must be positive.")
    if np.isclose(rho, 0.0):
        return np.inf if cap >= 1.0 else -np.inf
    if rho < 0.0:
        raise ValueError("clipping diagnostics currently assume rho >= 0.")
    return float((np.log(cap) + 0.5 * rho**2) / rho)


def clipped_weight_normalizer(cap: float, rho: float) -> float:
    cap = float(cap)
    rho = float(rho)
    if np.isinf(cap):
        return 1.0
    z_clip = clipping_threshold_z(cap, rho)
    if np.isposinf(z_clip):
        return 1.0
    if np.isneginf(z_clip):
        return cap
    return float(norm.cdf(z_clip - rho) + cap * norm.sf(z_clip))


def clipped_reference_tail_mass(cap: float, rho: float) -> float:
    z_clip = clipping_threshold_z(cap, rho)
    if np.isposinf(z_clip):
        return 0.0
    if np.isneginf(z_clip):
        return 1.0
    return float(norm.sf(z_clip))


def clipped_shift_tail_mass(cap: float, rho: float) -> float:
    z_clip = clipping_threshold_z(cap, rho)
    if np.isposinf(z_clip):
        return 0.0
    if np.isneginf(z_clip):
        return 1.0
    return float(norm.sf(z_clip - float(rho)))


def clipped_target_tv(cap: float, rho: float) -> float:
    """Exact TV distance between Q and Q_c for rho >= 0 and monotone weights."""
    cap = float(cap)
    rho = float(rho)
    if np.isinf(cap):
        return 0.0
    normalizer = clipped_weight_normalizer(cap, rho)
    if abs(normalizer - 1.0) < 1e-14:
        return 0.0
    if normalizer <= 0.0:
        raise ValueError("clipped target normalizer must be positive.")

    z_clip = clipping_threshold_z(cap, rho)
    z_cross = clipping_threshold_z(cap / normalizer, rho)

    term_left = (1.0 / normalizer - 1.0) * norm.cdf(z_clip - rho)
    term_mid = (cap / normalizer) * (
        norm.cdf(z_cross) - norm.cdf(z_clip)
    ) - (norm.cdf(z_cross - rho) - norm.cdf(z_clip - rho))
    term_right = norm.sf(z_cross - rho) - (cap / normalizer) * norm.sf(z_cross)
    exact_tv = float(0.5 * (term_left + term_mid + term_right))
    if exact_tv > 1e-12 or normalizer < 0.99:
        return max(exact_tv, 0.0)
    return clipped_target_tv_numerical(cap, rho, normalizer)


def clipped_target_tv_numerical(cap: float, rho: float, normalizer: float) -> float:
    lower = min(-10.0, float(rho) - 10.0)
    upper = max(10.0, float(rho) + 10.0)
    grid = np.linspace(lower, upper, 20_001)
    p_density = norm.pdf(grid)
    weights = density_ratio(grid, rho)
    clipped_density_ratio = np.minimum(weights, float(cap)) / float(normalizer)
    integrand = np.abs(weights - clipped_density_ratio) * p_density
    return float(0.5 * np.trapezoid(integrand, grid))


def clipped_target_tail(
    scores: np.ndarray,
    cap: float,
    rho: float,
) -> np.ndarray:
    """Upper tail of the normalized clipped target min(w, c) dP / E[min(w, c)]."""
    scores = np.asarray(scores, dtype=float)
    cap = float(cap)
    rho = float(rho)
    if np.isinf(cap):
        return shifted_null_tail(scores, rho=rho)
    if cap <= 0.0:
        raise ValueError("clip cap must be positive.")
    if np.isclose(rho, 0.0):
        return norm.sf(scores)
    if rho < 0.0:
        raise ValueError("clipped_target_tail currently assumes rho >= 0.")

    z_clip = clipping_threshold_z(cap, rho)
    q_segment = np.where(
        scores < z_clip,
        norm.cdf(z_clip - rho) - norm.cdf(scores - rho),
        0.0,
    )
    p_clipped_tail = cap * norm.sf(np.maximum(scores, z_clip))
    denominator = clipped_weight_normalizer(cap, rho)
    return (q_segment + p_clipped_tail) / denominator


def oracle_tail_metrics(
    cap: float,
    tail_probs: np.ndarray = TAIL_PROBS,
    rho: float = 1.5,
) -> dict[str, float]:
    q_tail = np.asarray(tail_probs, dtype=float)
    scores = tail_score_grid(q_tail, rho=rho)
    clipped_tail = clipped_target_tail(scores, cap=cap, rho=rho)
    ratio = np.maximum(clipped_tail, EPSILON) / q_tail
    abs_log_mismatch = np.abs(np.log10(ratio))
    bias = clipped_tail - q_tail
    return {
        "oracle_tail_mismatch_mean_abs_log10": float(np.mean(abs_log_mismatch)),
        "oracle_tail_mismatch_max_abs_log10": float(np.max(abs_log_mismatch)),
        "oracle_tail_bias_mean": float(np.mean(bias)),
        "oracle_tail_abs_bias_mean": float(np.mean(np.abs(bias))),
        "oracle_tail_bias_max_abs": float(np.max(np.abs(bias))),
        "oracle_tail_bias_signed_max": float(np.max(bias)),
        "oracle_tail_bias_signed_min": float(np.min(bias)),
    }


def clipping_adaptation_metrics(cap: float, rho: float) -> dict[str, float]:
    return {
        "clipped_target_normalizer": clipped_weight_normalizer(cap, rho),
        "clipped_shift_tail_mass": clipped_shift_tail_mass(cap, rho),
        "clipped_reference_tail_mass": clipped_reference_tail_mass(cap, rho),
        "clipped_target_tv": clipped_target_tv(cap, rho),
    }


def empirical_weighted_tail(
    sorted_scores: np.ndarray,
    suffix_weights: np.ndarray,
    total_weight: float,
    scores: np.ndarray,
) -> np.ndarray:
    if total_weight <= 0.0:
        raise ValueError("total_weight must be positive.")
    tail_start = np.searchsorted(sorted_scores, scores, side="left")
    return suffix_weights[tail_start] / float(total_weight)


def sample_tail_bias_metrics(
    calib_z: np.ndarray,
    cap: float,
    tail_probs: np.ndarray = TAIL_PROBS,
    rho: float = 1.5,
) -> dict[str, float]:
    scores = tail_score_grid(tail_probs, rho=rho)
    weights = clipped_weights(calib_z, cap=cap, rho=rho)
    order = np.argsort(calib_z, kind="mergesort")
    sorted_scores = np.asarray(calib_z, dtype=float)[order]
    sorted_weights = weights[order]
    suffix_weights = np.concatenate(([0.0], np.cumsum(sorted_weights[::-1])))[::-1]
    total_weight = float(np.sum(weights))
    sample_tail = empirical_weighted_tail(
        sorted_scores,
        suffix_weights,
        total_weight,
        scores,
    )
    bias = sample_tail - np.asarray(tail_probs, dtype=float)
    return {
        "sample_tail_bias_mean": float(np.mean(bias)),
        "sample_tail_abs_bias_mean": float(np.mean(np.abs(bias))),
        "sample_tail_bias_max_abs": float(np.max(np.abs(bias))),
    }


def resolution_metrics(
    calib_z: np.ndarray,
    test_z: np.ndarray,
    cap: float,
    rho: float,
) -> dict[str, float]:
    calib_weights = clipped_weights(calib_z, cap=cap, rho=rho)
    test_weights = clipped_weights(test_z, cap=cap, rho=rho)
    total_calib_weight = float(np.sum(calib_weights))
    if total_calib_weight <= 0.0:
        raise ValueError("calibration weights must have positive total mass.")

    test_self_atoms = test_weights / (total_calib_weight + test_weights)
    calib_atoms = calib_weights / total_calib_weight
    calib_ess = effective_sample_size(calib_weights)
    max_test_self_atom = float(np.max(test_self_atoms))
    bh_first_threshold = float(ALPHA / M)
    return {
        "max_test_self_atom": max_test_self_atom,
        "test_self_atom_q50": float(np.quantile(test_self_atoms, 0.50)),
        "test_self_atom_q90": float(np.quantile(test_self_atoms, 0.90)),
        "test_self_atom_q95": float(np.quantile(test_self_atoms, 0.95)),
        "test_self_atom_q99": float(np.quantile(test_self_atoms, 0.99)),
        "max_calib_atom": float(np.max(calib_atoms)),
        "calib_ess": calib_ess,
        "calib_ess_fraction": calib_ess / len(calib_weights),
        "total_calib_weight": total_calib_weight,
        "bh_first_threshold": bh_first_threshold,
        "max_test_self_atom_to_bh_threshold": (
            max_test_self_atom / bh_first_threshold
        ),
        "log10_inverse_max_test_self_atom": float(
            np.log10(1.0 / max_test_self_atom)
        ),
    }


def simulate_seed(
    seed: int,
    rho: float,
    caps: list[float],
) -> list[dict[str, float | int | str]]:
    rng = rng_for(rho_code(rho), seed)
    calib_z = rng.normal(0.0, 1.0, int(N_CAL))
    test_z = rng.normal(float(rho), 1.0, int(M))
    rows: list[dict[str, float | int | str]] = []
    for cap_idx, cap in enumerate(caps):
        row: dict[str, float | int | str] = {
            "seed": seed,
            "rho": float(rho),
            "clip_order": cap_idx,
            "clip_cap": float(cap),
            "clip_cap_label": cap_label(float(cap)),
        }
        row.update(resolution_metrics(calib_z, test_z, cap=float(cap), rho=float(rho)))
        row.update(sample_tail_bias_metrics(calib_z, cap=float(cap), rho=float(rho)))
        rows.append(row)
    return rows


def build_summary() -> pd.DataFrame:
    caps = [float(cap) for cap in CLIP_CAPS]
    rhos = [float(rho) for rho in RHO_VALUES]
    plot_x_by_label = cap_plot_positions(caps)
    trial_rows = [
        row
        for rho in rhos
        for seed in range(int(N_SEEDS))
        for row in simulate_seed(seed, rho, caps)
    ]
    trials = pd.DataFrame(trial_rows)

    summary_rows = []
    for rho_idx, rho in enumerate(rhos):
        for cap_idx, cap in enumerate(caps):
            label = cap_label(cap)
            block = trials[
                trials["rho"].eq(float(rho)) & trials["clip_cap_label"].eq(label)
            ]
            if block.empty:
                raise RuntimeError(f"No clipping trials for rho={rho:g}, cap={label}.")

            row: dict[str, float | int | str] = {
                "summary_version": SUMMARY_VERSION,
                "rho_order": rho_idx,
                "rho": float(rho),
                "clip_order": cap_idx,
                "clip_cap": cap,
                "clip_cap_label": label,
                "plot_x": plot_x_by_label[label],
                "alpha": float(ALPHA),
                "n_cal": int(N_CAL),
                "m": int(M),
                "n_seeds": int(N_SEEDS),
                "tail_grid_size": int(len(TAIL_PROBS)),
                "tail_probability_min": float(np.min(TAIL_PROBS)),
                "tail_probability_max": float(np.max(TAIL_PROBS)),
                "bh_first_threshold": float(ALPHA / M),
            }
            for metric in [
                "max_test_self_atom",
                "test_self_atom_q50",
                "test_self_atom_q90",
                "test_self_atom_q95",
                "test_self_atom_q99",
                "max_calib_atom",
                "calib_ess",
                "calib_ess_fraction",
                "total_calib_weight",
                "max_test_self_atom_to_bh_threshold",
                "log10_inverse_max_test_self_atom",
                "sample_tail_bias_mean",
                "sample_tail_abs_bias_mean",
                "sample_tail_bias_max_abs",
            ]:
                row[f"{metric}_mean"] = float(block[metric].mean())
                row[f"{metric}_median"] = float(block[metric].median())
            row.update(oracle_tail_metrics(cap=cap, rho=float(rho)))
            row.update(clipping_adaptation_metrics(cap=cap, rho=float(rho)))
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["rho_order", "clip_order"],
        kind="mergesort",
    )
    validate_summary(summary)
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def expected_pairs() -> list[tuple[float, str]]:
    return [
        (float(rho), cap_label(float(cap)))
        for rho in RHO_VALUES
        for cap in CLIP_CAPS
    ]


def summary_pairs(summary: pd.DataFrame) -> list[tuple[float, str]]:
    ordered = summary.sort_values(["rho_order", "clip_order"], kind="mergesort")
    return [
        (float(row.rho), str(row.clip_cap_label))
        for row in ordered.itertuples(index=False)
    ]


def summary_is_current() -> bool:
    if not SUMMARY_PATH.exists():
        return False
    summary = pd.read_csv(SUMMARY_PATH)
    if summary.empty or "summary_version" not in summary:
        return False
    if str(summary["summary_version"].iloc[0]) != SUMMARY_VERSION:
        return False
    return summary_pairs(summary) == expected_pairs()


def load_or_build_summary() -> pd.DataFrame:
    if summary_is_current():
        print("loading existing Figure 4 summary", flush=True)
        return pd.read_csv(SUMMARY_PATH)
    return build_summary()


def validate_summary(summary: pd.DataFrame) -> None:
    expected = expected_pairs()
    observed = summary_pairs(summary)
    if observed != expected:
        raise RuntimeError(
            f"Unexpected rho/cap pairs: expected={expected}, observed={observed}."
        )
    if summary.duplicated(["rho", "clip_cap_label"]).any():
        raise RuntimeError("Figure 4 summary contains duplicate rho/cap pairs.")
    finite_columns = [
        "plot_x",
        "max_test_self_atom_mean",
        "calib_ess_fraction_mean",
        "max_test_self_atom_to_bh_threshold_mean",
        "log10_inverse_max_test_self_atom_mean",
        "oracle_tail_mismatch_mean_abs_log10",
        "oracle_tail_abs_bias_mean",
        "sample_tail_abs_bias_mean_mean",
        "clipped_target_normalizer",
        "clipped_shift_tail_mass",
        "clipped_reference_tail_mass",
        "clipped_target_tv",
    ]
    for column in finite_columns:
        if not np.isfinite(summary[column]).all():
            raise RuntimeError(f"Summary column must be finite: {column}.")
    unit_interval_columns = [
        "max_test_self_atom_mean",
        "test_self_atom_q50_mean",
        "test_self_atom_q90_mean",
        "test_self_atom_q95_mean",
        "test_self_atom_q99_mean",
        "max_calib_atom_mean",
        "calib_ess_fraction_mean",
        "clipped_target_normalizer",
        "clipped_shift_tail_mass",
        "clipped_reference_tail_mass",
        "clipped_target_tv",
    ]
    for column in unit_interval_columns:
        if not ((summary[column] >= 0.0) & (summary[column] <= 1.0)).all():
            raise RuntimeError(f"Summary unit interval column out of range: {column}.")
    unclipped = summary[summary["clip_cap_label"].eq("unclipped")]
    if not unclipped.empty:
        if (unclipped["oracle_tail_mismatch_mean_abs_log10"].abs() > 1e-10).any():
            raise RuntimeError("Unclipped oracle tail mismatch should be zero.")
        if (unclipped["clipped_target_tv"].abs() > 1e-12).any():
            raise RuntimeError("Unclipped TV should be zero.")


def write_key_table(summary: pd.DataFrame) -> None:
    rows = []
    for rho, block in summary.groupby("rho", sort=True):
        ordered = block.sort_values("clip_order", kind="mergesort").reset_index(drop=True)
        candidate_indices = sorted({0, len(ordered) // 2, len(ordered) - 1})
        setting_names = {
            candidate_indices[0]: "tightest_clip",
            candidate_indices[-1]: "unclipped_or_largest_cap",
        }
        if len(candidate_indices) == 3:
            setting_names[candidate_indices[1]] = "middle_cap"
        for idx in candidate_indices:
            row = ordered.iloc[idx]
            rows.append(
                {
                    "summary_version": SUMMARY_VERSION,
                    "rho": float(rho),
                    "setting": setting_names[idx],
                    "clip_cap_label": row["clip_cap_label"],
                    "clip_cap": row["clip_cap"],
                    "mean_max_test_self_atom": row["max_test_self_atom_mean"],
                    "mean_calib_ess_fraction": row["calib_ess_fraction_mean"],
                    "mean_max_atom_to_bh_threshold": row[
                        "max_test_self_atom_to_bh_threshold_mean"
                    ],
                    "oracle_tail_mismatch_mean_abs_log10": row[
                        "oracle_tail_mismatch_mean_abs_log10"
                    ],
                    "clipped_target_tv": row["clipped_target_tv"],
                    "clipped_shift_tail_mass": row["clipped_shift_tail_mass"],
                    "oracle_tail_abs_bias_mean": row["oracle_tail_abs_bias_mean"],
                    "sample_tail_abs_bias_mean": row["sample_tail_abs_bias_mean_mean"],
                }
            )
    pd.DataFrame(rows).to_csv(TABLE_PATH, index=False)


def apply_cap_axis(ax: plt.Axes, summary: pd.DataFrame) -> None:
    cap_axis = (
        summary[["clip_order", "clip_cap_label", "plot_x"]]
        .drop_duplicates()
        .sort_values("clip_order", kind="mergesort")
    )
    x = cap_axis["plot_x"].to_numpy(dtype=float)
    labels = cap_axis["clip_cap_label"].astype(str).tolist()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    x_span = float(np.max(x) - np.min(x))
    padding = max(0.05 * x_span, 0.05)
    ax.set_xlim(float(np.min(x) - padding), float(np.max(x) + padding))


def plot_resolution_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    for idx, (rho, block) in enumerate(summary.groupby("rho", sort=True)):
        line = block.sort_values("clip_order", kind="mergesort")
        ax.plot(
            line["plot_x"],
            line["max_test_self_atom_mean"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=rho_color(float(rho), idx),
            label=rf"$\rho={float(rho):g}$",
        )
    ax.axhline(
        float(ALPHA / M),
        color="#111111",
        linestyle=":",
        linewidth=1.1,
        label=r"$\alpha/m$",
    )
    ax.set_yscale("log")
    ax.set_ylabel("mean max shifted-null self-atom")
    ax.set_xlabel("upper clipping cap")
    ax.set_title("A. Resolution improves as caps tighten")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")
    apply_cap_axis(ax, summary)


def plot_adaptation_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    tv_ax = ax.twinx()
    for idx, (rho, block) in enumerate(summary.groupby("rho", sort=True)):
        line = block.sort_values("clip_order", kind="mergesort")
        color = rho_color(float(rho), idx)
        ax.plot(
            line["plot_x"],
            line["oracle_tail_mismatch_mean_abs_log10"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=rf"mismatch, $\rho={float(rho):g}$",
        )
        tv_ax.plot(
            line["plot_x"],
            line["clipped_target_tv"],
            marker="s",
            linewidth=1.2,
            markersize=3.3,
            linestyle="--",
            color=color,
            alpha=0.72,
            label=rf"TV, $\rho={float(rho):g}$",
        )
    ax.set_ylabel(r"mean $|\log_{10}(T_c/T_Q)|$")
    tv_ax.set_ylabel(r"$\mathrm{TV}(Q,Q_c)$")
    tv_ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("upper clipping cap")
    ax.set_title("B. Adaptation worsens as caps tighten")
    ax.grid(alpha=0.18, linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    tv_handles, tv_labels = tv_ax.get_legend_handles_labels()
    ax.legend(
        handles + tv_handles,
        labels + tv_labels,
        frameon=False,
        fontsize=7,
        loc="best",
    )
    apply_cap_axis(ax, summary)


def plot_frontier_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    labels_to_annotate = {"c=1", "c=13", "unclipped"}
    for idx, (rho, block) in enumerate(summary.groupby("rho", sort=True)):
        line = block.sort_values("clip_order", kind="mergesort")
        color = rho_color(float(rho), idx)
        ax.plot(
            line["max_test_self_atom_mean"],
            line["oracle_tail_mismatch_mean_abs_log10"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=rf"$\rho={float(rho):g}$",
        )
        for row in line.itertuples(index=False):
            if str(row.clip_cap_label) not in labels_to_annotate:
                continue
            ax.annotate(
                str(row.clip_cap_label),
                (
                    float(row.max_test_self_atom_mean),
                    float(row.oracle_tail_mismatch_mean_abs_log10),
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )
    ax.set_xscale("log")
    ax.set_xlabel("mean max shifted-null self-atom")
    ax.set_ylabel(r"mean $|\log_{10}(T_c/T_Q)|$")
    ax.set_title("C. The clipping frontier")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")


def plot_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.5), constrained_layout=True)
    plot_resolution_panel(axes[0], summary)
    plot_adaptation_panel(axes[1], summary)
    plot_frontier_panel(axes[2], summary)
    fig.suptitle(
        "Clipping frontier: better resolution requires a shifted-null target change",
        fontsize=14,
    )
    fig.savefig(FIGURE_PATH, bbox_inches="tight", dpi=220)
    plt.close(fig)


def export_tikz(summary: pd.DataFrame) -> None:
    specs = [
        (
            "A",
            "resolution",
            "max_test_self_atom",
            "max_test_self_atom_mean",
            "plot_x",
            "atom",
        ),
        (
            "A",
            "resolution",
            "calib_ess_fraction",
            "calib_ess_fraction_mean",
            "plot_x",
            "ess",
        ),
        (
            "B",
            "adaptation",
            "oracle_tail_mismatch",
            "oracle_tail_mismatch_mean_abs_log10",
            "plot_x",
            "mismatch",
        ),
        (
            "B",
            "adaptation",
            "clipped_target_tv",
            "clipped_target_tv",
            "plot_x",
            "tv",
        ),
        (
            "C",
            "frontier",
            "frontier_oracle_tail_mismatch",
            "oracle_tail_mismatch_mean_abs_log10",
            "max_test_self_atom_mean",
            "frontier",
        ),
    ]
    rows = []
    ordered = summary.sort_values(["rho_order", "clip_order"], kind="mergesort")
    for plot_order, (panel, group, metric, y_column, x_column, style_key) in enumerate(
        specs,
        start=1,
    ):
        for point_index, row in enumerate(ordered.itertuples(index=False)):
            value = float(getattr(row, y_column))
            rows.append(
                {
                    "export_version": TIKZ_EXPORT_VERSION,
                    "figure": "figure4",
                    "panel": panel,
                    "panel_title": group,
                    "plot_order": plot_order,
                    "point_index": point_index,
                    "group": group,
                    "metric": metric,
                    "rho": float(row.rho),
                    "clip_cap": row.clip_cap,
                    "clip_cap_label": row.clip_cap_label,
                    "x": float(getattr(row, x_column)),
                    "y": value,
                    "value": value,
                    "count": int(row.n_seeds),
                    "style_key": style_key,
                    "label": METRIC_LABELS[metric],
                }
            )
    pd.DataFrame(rows).to_csv(TIKZ_PATH, index=False)


def write_rationale() -> None:
    RATIONALE_PATH.write_text(
        """# Figure 4 Rationale: Clipping Frontier

This figure is a controlled mechanism experiment, not a new method benchmark.
It asks whether clipping oracle covariate-shift weights removes weighted
conformal resolution collapse or merely changes the target distribution.

Calibration null scores are drawn from `P: Z ~ N(0, 1)`. Shifted null scores are
drawn from `Q_rho: Z ~ N(rho, 1)`, and the anomaly score is `S = Z`. The oracle
density ratio is

```text
w(z) = dQ_rho / dP = exp(rho z - rho^2 / 2).
```

For a clipping cap `c`, the experiment uses

```text
w_c(z) = min(w(z), c)
```

on both calibration and shifted-null test points. This improves resolution by
reducing large self-atoms in weighted conformal p-values and raising calibration
effective sample size.

The cost is that clipping no longer adapts to the true shifted null `Q`. It
instead targets

```text
Q_c(dz) propto min(w(z), c) P(dz).
```

The theorem-facing adaptation cost is therefore the distance between `Q` and
`Q_c`. The summary records the clipped target normalizer, the reference and
shifted mass where `w > c`, and the total variation gap
`TV(Q, Q_c)`.

The plotted frontier makes the trade-off explicit. Tight caps move leftward in
Panel C by lowering max atom mass, but they move upward by increasing oracle
shifted-null tail mismatch. The unclipped point has zero target mismatch but can
have poor finite-sample resolution. Thus clipping changes the
adaptation-resolution frontier; it does not remove it.
""",
        encoding="utf-8",
    )


def validate_outputs() -> None:
    for path in [
        FIGURE_PATH,
        SUMMARY_PATH,
        TIKZ_PATH,
        TABLE_PATH,
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
    summary = load_or_build_summary()
    validate_summary(summary)
    write_key_table(summary)
    plot_figure(summary)
    export_tikz(summary)
    write_rationale()
    validate_outputs()
    print(SUMMARY_PATH)
    print(TABLE_PATH)
    print(FIGURE_PATH)
    print(TIKZ_PATH)
    print(RATIONALE_PATH)


if __name__ == "__main__":
    main()
