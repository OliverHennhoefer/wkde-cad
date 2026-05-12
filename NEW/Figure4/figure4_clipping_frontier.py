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

SUMMARY_VERSION = "clipping-frontier-v1"
TIKZ_EXPORT_VERSION = "tikz-v1"
BASE_SEED = 20260512

RHO = 1.5
N_CAL = 500
M = 1000
N_SEEDS = 500
ALPHA = 0.10
CLIP_CAPS = [1, 1.5, 2, 3, 5, 8, 13, 21, 34, 55, 89, np.inf]
TAIL_PROBS = np.geomspace(1e-4, 0.5, 80)
EPSILON = 1e-300

COLORS = {
    "atom": "#2563eb",
    "ess": "#0f766e",
    "mismatch": "#c2410c",
    "bias": "#7c3aed",
}

METRIC_LABELS = {
    "max_test_self_atom": "mean max shifted-null self-atom",
    "calib_ess_fraction": "mean calibration ESS / n",
    "oracle_tail_mismatch": "oracle shifted-null tail mismatch",
    "oracle_abs_tail_bias": "oracle absolute target-Q tail bias",
}


def rng_for(*values: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *values]))


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


def density_ratio(z_values: np.ndarray, rho: float = RHO) -> np.ndarray:
    z_values = np.asarray(z_values, dtype=float)
    return np.exp(float(rho) * z_values - 0.5 * float(rho) ** 2)


def clipped_weights(
    z_values: np.ndarray,
    cap: float,
    rho: float = RHO,
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
    rho: float = RHO,
) -> np.ndarray:
    tail_probs = np.asarray(tail_probs, dtype=float)
    if np.any((tail_probs <= 0.0) | (tail_probs >= 1.0)):
        raise ValueError("tail probabilities must lie in (0, 1).")
    return float(rho) + norm.isf(tail_probs)


def shifted_null_tail(scores: np.ndarray, rho: float = RHO) -> np.ndarray:
    return norm.sf(np.asarray(scores, dtype=float) - float(rho))


def clipped_target_tail(
    scores: np.ndarray,
    cap: float,
    rho: float = RHO,
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

    z_clip = (np.log(cap) + 0.5 * rho**2) / rho
    q_segment = np.where(
        scores < z_clip,
        norm.cdf(z_clip - rho) - norm.cdf(scores - rho),
        0.0,
    )
    p_clipped_tail = cap * norm.sf(np.maximum(scores, z_clip))
    denominator = norm.cdf(z_clip - rho) + cap * norm.sf(z_clip)
    return (q_segment + p_clipped_tail) / denominator


def oracle_tail_metrics(
    cap: float,
    tail_probs: np.ndarray = TAIL_PROBS,
    rho: float = RHO,
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
    rho: float = RHO,
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
    rho: float = RHO,
) -> dict[str, float]:
    calib_weights = clipped_weights(calib_z, cap=cap, rho=rho)
    test_weights = clipped_weights(test_z, cap=cap, rho=rho)
    total_calib_weight = float(np.sum(calib_weights))
    if total_calib_weight <= 0.0:
        raise ValueError("calibration weights must have positive total mass.")

    test_self_atoms = test_weights / (total_calib_weight + test_weights)
    calib_atoms = calib_weights / total_calib_weight
    calib_ess = effective_sample_size(calib_weights)
    return {
        "max_test_self_atom": float(np.max(test_self_atoms)),
        "test_self_atom_q50": float(np.quantile(test_self_atoms, 0.50)),
        "test_self_atom_q90": float(np.quantile(test_self_atoms, 0.90)),
        "test_self_atom_q95": float(np.quantile(test_self_atoms, 0.95)),
        "test_self_atom_q99": float(np.quantile(test_self_atoms, 0.99)),
        "max_calib_atom": float(np.max(calib_atoms)),
        "calib_ess": calib_ess,
        "calib_ess_fraction": calib_ess / len(calib_weights),
        "total_calib_weight": total_calib_weight,
    }


def simulate_seed(seed: int, caps: list[float]) -> list[dict[str, float | int | str]]:
    rng = rng_for(seed)
    calib_z = rng.normal(0.0, 1.0, int(N_CAL))
    test_z = rng.normal(float(RHO), 1.0, int(M))
    rows: list[dict[str, float | int | str]] = []
    for cap_idx, cap in enumerate(caps):
        row: dict[str, float | int | str] = {
            "seed": seed,
            "clip_order": cap_idx,
            "clip_cap": float(cap),
            "clip_cap_label": cap_label(float(cap)),
        }
        row.update(resolution_metrics(calib_z, test_z, cap=float(cap), rho=float(RHO)))
        row.update(sample_tail_bias_metrics(calib_z, cap=float(cap), rho=float(RHO)))
        rows.append(row)
    return rows


def build_summary() -> pd.DataFrame:
    caps = [float(cap) for cap in CLIP_CAPS]
    plot_x_by_label = cap_plot_positions(caps)
    trial_rows = [
        row
        for seed in range(int(N_SEEDS))
        for row in simulate_seed(seed, caps)
    ]
    trials = pd.DataFrame(trial_rows)

    summary_rows = []
    for cap_idx, cap in enumerate(caps):
        label = cap_label(cap)
        block = trials[trials["clip_cap_label"].eq(label)]
        if block.empty:
            raise RuntimeError(f"No clipping trials for cap={label}.")

        row: dict[str, float | int | str] = {
            "summary_version": SUMMARY_VERSION,
            "clip_order": cap_idx,
            "clip_cap": cap,
            "clip_cap_label": label,
            "plot_x": plot_x_by_label[label],
            "rho": float(RHO),
            "alpha": float(ALPHA),
            "n_cal": int(N_CAL),
            "m": int(M),
            "n_seeds": int(N_SEEDS),
            "tail_grid_size": int(len(TAIL_PROBS)),
            "tail_probability_min": float(np.min(TAIL_PROBS)),
            "tail_probability_max": float(np.max(TAIL_PROBS)),
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
            "sample_tail_bias_mean",
            "sample_tail_abs_bias_mean",
            "sample_tail_bias_max_abs",
        ]:
            row[f"{metric}_mean"] = float(block[metric].mean())
            row[f"{metric}_median"] = float(block[metric].median())
        row.update(oracle_tail_metrics(cap=cap, rho=float(RHO)))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("clip_order", kind="mergesort")
    validate_summary(summary)
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def expected_cap_labels() -> list[str]:
    return [cap_label(float(cap)) for cap in CLIP_CAPS]


def summary_is_current() -> bool:
    if not SUMMARY_PATH.exists():
        return False
    summary = pd.read_csv(SUMMARY_PATH)
    if summary.empty or "summary_version" not in summary:
        return False
    if str(summary["summary_version"].iloc[0]) != SUMMARY_VERSION:
        return False
    return list(summary.sort_values("clip_order")["clip_cap_label"]) == expected_cap_labels()


def load_or_build_summary() -> pd.DataFrame:
    if summary_is_current():
        print("loading existing Figure 4 summary", flush=True)
        return pd.read_csv(SUMMARY_PATH)
    return build_summary()


def validate_summary(summary: pd.DataFrame) -> None:
    expected = expected_cap_labels()
    observed = list(summary.sort_values("clip_order")["clip_cap_label"])
    if observed != expected:
        raise RuntimeError(f"Unexpected clip caps: expected={expected}, observed={observed}.")
    if summary.duplicated("clip_cap_label").any():
        raise RuntimeError("Figure 4 summary contains duplicate clip caps.")
    finite_columns = [
        "plot_x",
        "max_test_self_atom_mean",
        "calib_ess_fraction_mean",
        "oracle_tail_mismatch_mean_abs_log10",
        "oracle_tail_abs_bias_mean",
        "sample_tail_abs_bias_mean_mean",
    ]
    for column in finite_columns:
        if not np.isfinite(summary[column]).all():
            raise RuntimeError(f"Summary column must be finite: {column}.")
    atom_columns = [
        "max_test_self_atom_mean",
        "test_self_atom_q50_mean",
        "test_self_atom_q90_mean",
        "test_self_atom_q95_mean",
        "test_self_atom_q99_mean",
        "max_calib_atom_mean",
        "calib_ess_fraction_mean",
    ]
    for column in atom_columns:
        if not ((summary[column] >= 0.0) & (summary[column] <= 1.0)).all():
            raise RuntimeError(f"Summary atom/ESS fraction column out of range: {column}.")
    if "unclipped" in set(summary["clip_cap_label"]):
        unclipped = summary[summary["clip_cap_label"].eq("unclipped")].iloc[0]
        if abs(float(unclipped["oracle_tail_mismatch_mean_abs_log10"])) > 1e-10:
            raise RuntimeError("Unclipped oracle tail mismatch should be zero.")


def write_key_table(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("clip_order", kind="mergesort").reset_index(drop=True)
    candidate_indices = sorted({0, len(ordered) // 2, len(ordered) - 1})
    setting_names = {
        candidate_indices[0]: "tightest_clip",
        candidate_indices[-1]: "unclipped_or_largest_cap",
    }
    if len(candidate_indices) == 3:
        setting_names[candidate_indices[1]] = "middle_cap"

    rows = []
    for idx in candidate_indices:
        row = ordered.iloc[idx]
        rows.append(
            {
                "summary_version": SUMMARY_VERSION,
                "setting": setting_names[idx],
                "clip_cap_label": row["clip_cap_label"],
                "clip_cap": row["clip_cap"],
                "mean_max_test_self_atom": row["max_test_self_atom_mean"],
                "mean_calib_ess_fraction": row["calib_ess_fraction_mean"],
                "oracle_tail_mismatch_mean_abs_log10": row[
                    "oracle_tail_mismatch_mean_abs_log10"
                ],
                "oracle_tail_abs_bias_mean": row["oracle_tail_abs_bias_mean"],
                "sample_tail_abs_bias_mean": row["sample_tail_abs_bias_mean_mean"],
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_PATH, index=False)


def plot_resolution_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = summary["plot_x"].to_numpy(dtype=float)
    atom = summary["max_test_self_atom_mean"].to_numpy(dtype=float)
    ess = summary["calib_ess_fraction_mean"].to_numpy(dtype=float)

    ax.plot(
        x,
        atom,
        marker="o",
        linewidth=1.8,
        markersize=4,
        color=COLORS["atom"],
        label="max shifted-null self-atom",
    )
    ax.set_yscale("log")
    ax.set_ylabel("mean max self-atom")
    ax.set_xlabel("upper clipping cap")
    ax.set_title("A. Clipping improves p-value resolution")
    ax.grid(alpha=0.18, linewidth=0.6)

    ess_ax = ax.twinx()
    ess_ax.plot(
        x,
        ess,
        marker="s",
        linewidth=1.5,
        markersize=3.5,
        linestyle="--",
        color=COLORS["ess"],
        label="calibration ESS / n",
    )
    ess_ax.set_ylim(0.0, 1.05)
    ess_ax.set_ylabel("mean calibration ESS / n")

    handles, labels = ax.get_legend_handles_labels()
    ess_handles, ess_labels = ess_ax.get_legend_handles_labels()
    ax.legend(handles + ess_handles, labels + ess_labels, frameon=False, fontsize=8)


def plot_adaptation_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = summary["plot_x"].to_numpy(dtype=float)
    mismatch = summary["oracle_tail_mismatch_mean_abs_log10"].to_numpy(dtype=float)
    bias = summary["oracle_tail_abs_bias_mean"].to_numpy(dtype=float)

    ax.plot(
        x,
        mismatch,
        marker="o",
        linewidth=1.8,
        markersize=4,
        color=COLORS["mismatch"],
        label="mean abs log tail mismatch",
    )
    bias_ax = ax.twinx()
    bias_ax.plot(
        x,
        bias,
        marker="s",
        linewidth=1.4,
        markersize=3.5,
        linestyle="--",
        color=COLORS["bias"],
        label="mean abs target-Q tail bias",
    )
    ax.set_ylabel(r"mean $|\log_{10}(T_c/T_Q)|$")
    bias_ax.set_ylabel(r"mean $|T_c - T_Q|$")
    ax.set_xlabel("upper clipping cap")
    ax.set_title("B. Clipping worsens shifted-null adaptation")
    ax.grid(alpha=0.18, linewidth=0.6)

    handles, labels = ax.get_legend_handles_labels()
    bias_handles, bias_labels = bias_ax.get_legend_handles_labels()
    ax.legend(handles + bias_handles, labels + bias_labels, frameon=False, fontsize=8)


def apply_cap_axis(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = summary["plot_x"].to_numpy(dtype=float)
    labels = summary["clip_cap_label"].astype(str).tolist()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    x_span = float(np.max(x) - np.min(x))
    padding = max(0.05 * x_span, 0.05)
    ax.set_xlim(float(np.min(x) - padding), float(np.max(x) + padding))


def plot_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=True)
    plot_resolution_panel(axes[0], summary)
    plot_adaptation_panel(axes[1], summary)
    for ax in axes:
        apply_cap_axis(ax, summary)
    fig.suptitle(
        "Clipping frontier: resolution improves while shifted-null adaptation degrades",
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
            "atom",
        ),
        (
            "A",
            "resolution",
            "calib_ess_fraction",
            "calib_ess_fraction_mean",
            "ess",
        ),
        (
            "B",
            "adaptation",
            "oracle_tail_mismatch",
            "oracle_tail_mismatch_mean_abs_log10",
            "mismatch",
        ),
        (
            "B",
            "adaptation",
            "oracle_abs_tail_bias",
            "oracle_tail_abs_bias_mean",
            "bias",
        ),
    ]
    rows = []
    ordered = summary.sort_values("clip_order", kind="mergesort")
    for plot_order, (panel, group, metric, column, style_key) in enumerate(specs, start=1):
        for point_index, row in enumerate(ordered.itertuples(index=False)):
            value = float(getattr(row, column))
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
                    "clip_cap": row.clip_cap,
                    "clip_cap_label": row.clip_cap_label,
                    "x": float(row.plot_x),
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

This figure is a controlled mechanism experiment. It asks whether clipping
oracle covariate-shift weights removes the weighted conformal resolution
problem, or merely changes the trade-off.

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

on both calibration and shifted-null test points.

The resolution diagnostic is the largest shifted-null self-atom,

```text
max_j w_c(Z_j^test) / (sum_i w_c(Z_i^cal) + w_c(Z_j^test)).
```

Smaller values mean the weighted conformal p-values can attain finer small
p-values. Clipping lowers this atom and raises the calibration effective sample
size because it suppresses extreme weights.

The adaptation diagnostic compares the normalized clipped target tail

```text
T_c(s) = E_P[min(w(Z), c) 1{Z >= s}] / E_P[min(w(Z), c)]
```

against the true shifted-null tail

```text
T_Q(s) = P_Q(Z >= s).
```

The main adaptation curve is the mean absolute log mismatch
`mean_s |log10(T_c(s) / T_Q(s))|` over a fixed shifted-null tail grid. The CSV
also records signed and absolute target-Q tail bias, plus finite-sample
weighted-tail bias from simulated calibration samples.

The point is that clipping improves finite-sample resolution by changing the
target distribution away from `Q`. It changes the frontier; it does not remove
the resolution-adaptation trade-off.
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
