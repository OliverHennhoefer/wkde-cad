from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.covariate_shift import fit_propensity_model, rejection_sample
from src.utils.data_loader import load
from src.utils.registry import get_dataset_enum


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.toml"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _seed_list(seed_count: int) -> list[int]:
    if (
        not isinstance(seed_count, int)
        or isinstance(seed_count, bool)
        or seed_count < 1
    ):
        raise ValueError("experiment.meta_seeds must be a positive integer count.")
    return list(range(1, seed_count + 1))


def _load_dataset_data(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data = load(get_dataset_enum(dataset), setup=False)
    normal = data[data["Class"] == 0].copy()
    anomaly = data[data["Class"] == 1].copy()
    feature_columns = [col for col in normal.columns if col != "Class"]
    return normal, anomaly, feature_columns


def _seeded_uniforms(index: pd.Index, seed: int) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).random(len(index)), index=index)


def _sample_by_priority(
    data: pd.DataFrame,
    n: int,
    priority: pd.Series,
) -> pd.DataFrame:
    if n >= len(data):
        return data.copy()
    selected_index = (
        priority.loc[data.index].sort_values(kind="mergesort").head(n).index
    )
    return data.loc[selected_index].copy()


def _split_anomaly_candidates(
    anomaly: pd.DataFrame,
    feature_columns: list[str],
    propensity_model: Any,
    train_split: float,
    seed: int,
    assignment_uniforms: pd.Series,
) -> pd.DataFrame:
    if len(anomaly) < 2:
        return anomaly.copy()

    sampled = rejection_sample(
        anomaly,
        feature_columns,
        propensity_model,
        seed=seed + 10_000,
        uniforms=assignment_uniforms,
    )
    if len(sampled.accepted) > 0:
        return sampled.accepted

    _, fallback = train_test_split(
        anomaly,
        train_size=train_split,
        random_state=seed,
    )
    return fallback.copy()


def _split_diagnostics(
    *,
    normal: pd.DataFrame,
    anomaly: pd.DataFrame,
    feature_columns: list[str],
    train_split: float,
    test_use_proportion: float,
    test_anomaly_rate: float,
    severity: float,
    propensity_min: float,
    propensity_max: float,
    seeds: list[int],
) -> tuple[pd.DataFrame, Any]:
    model = fit_propensity_model(
        normal[feature_columns],
        train_split=train_split,
        severity=severity,
        propensity_min=propensity_min,
        propensity_max=propensity_max,
    )
    rows = []
    scores = model.score(normal[feature_columns])
    propensities = model.propensity(normal[feature_columns])
    score_by_index = pd.Series(scores, index=normal.index)
    propensity_by_index = pd.Series(propensities, index=normal.index)

    for seed in seeds:
        normal_assignment_uniforms = _seeded_uniforms(normal.index, seed)
        anomaly_assignment_uniforms = _seeded_uniforms(anomaly.index, seed + 10_000)
        normal_test_priority = _seeded_uniforms(normal.index, seed + 20_000)

        sample = rejection_sample(
            normal,
            feature_columns,
            model,
            seed=seed,
            uniforms=normal_assignment_uniforms,
        )
        anomaly_test = _split_anomaly_candidates(
            anomaly,
            feature_columns,
            model,
            train_split,
            seed,
            anomaly_assignment_uniforms,
        )

        total_test_available = len(sample.accepted) + len(anomaly_test)
        target_test_size = round(test_use_proportion * total_test_available)
        if target_test_size < 2:
            n_normal_test = 0
        else:
            n_anomalies_test = round(target_test_size * test_anomaly_rate)
            n_normal_test = target_test_size - n_anomalies_test
            if n_anomalies_test < 1:
                n_anomalies_test = 1
                n_normal_test = target_test_size - 1
            if n_anomalies_test > len(anomaly_test):
                n_anomalies_test = len(anomaly_test)
                n_normal_test = target_test_size - n_anomalies_test
            if n_normal_test > len(sample.accepted):
                n_normal_test = len(sample.accepted)
            if n_normal_test < 1 or n_anomalies_test < 1:
                n_normal_test = 0

        final_normal_test = _sample_by_priority(
            sample.accepted,
            n_normal_test,
            normal_test_priority,
        )
        split_by_index = pd.Series("calibration", index=normal.index)
        split_by_index.loc[final_normal_test.index] = "test"

        seed_rows = pd.DataFrame(
            {
                "seed": seed,
                "split": split_by_index,
                "shift_score": score_by_index,
                "propensity": propensity_by_index,
                "oracle_weight": propensity_by_index / (1.0 - propensity_by_index),
            }
        )
        rows.append(seed_rows)

    return pd.concat(rows, ignore_index=True), model


def _support_table(diagnostics: pd.DataFrame, bins: int) -> pd.DataFrame:
    table = diagnostics.copy()
    table["score_bin"] = pd.cut(table["shift_score"], bins=bins)
    counts = (
        table.groupby(["score_bin", "split"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "test" not in counts:
        counts["test"] = 0
    count_columns = [col for col in counts.columns if col not in {"score_bin"}]
    denominator = counts[count_columns].sum(axis=1)
    counts["test_fraction"] = counts["test"] / denominator
    counts["bin_midpoint"] = counts["score_bin"].apply(lambda interval: interval.mid)
    return counts


def _padded_limits(values: np.ndarray) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    if span <= 0.0:
        return lower - 1.0, upper + 1.0
    padding = 0.06 * span
    return lower - padding, upper + padding


def _propensity_surface(
    *,
    model: Any,
    scaler: StandardScaler,
    pca: PCA,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    resolution: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_x = np.linspace(xlim[0], xlim[1], resolution)
    grid_y = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_embedding = np.column_stack([xx.ravel(), yy.ravel()])
    grid_scaled_features = pca.inverse_transform(grid_embedding)
    grid_features = scaler.inverse_transform(grid_scaled_features)
    propensity = model.propensity(grid_features).reshape(xx.shape)
    return grid_x, grid_y, propensity


def _plot_dataset(
    *,
    dataset: str,
    normal: pd.DataFrame,
    anomaly: pd.DataFrame,
    feature_columns: list[str],
    train_split: float,
    test_use_proportion: float,
    test_anomaly_rate: float,
    severities: list[float],
    propensity_min: float,
    propensity_max: float,
    seeds: list[int],
    output_dir: Path,
    bins: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = normal[feature_columns].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2, random_state=0)
    embedding = pca.fit_transform(scaled)
    embedding_df = pd.DataFrame(
        {
            "pc1": embedding[:, 0],
            "pc2": embedding[:, 1],
        },
        index=normal.index,
    )
    xlim = _padded_limits(embedding[:, 0])
    ylim = _padded_limits(embedding[:, 1])

    n_cols = 2 if len(severities) == 4 else min(3, len(severities))
    n_rows = math.ceil(len(severities) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.6 * n_cols, 4.8 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    all_propensities = []
    panel_data = []
    for severity in severities:
        diagnostics, propensity_model = _split_diagnostics(
            normal=normal,
            anomaly=anomaly,
            feature_columns=feature_columns,
            train_split=train_split,
            test_use_proportion=test_use_proportion,
            test_anomaly_rate=test_anomaly_rate,
            severity=severity,
            propensity_min=propensity_min,
            propensity_max=propensity_max,
            seeds=seeds,
        )
        first_seed = diagnostics[diagnostics["seed"] == seeds[0]].copy()
        first_seed = first_seed.join(embedding_df.reset_index(drop=True))
        support = _support_table(diagnostics, bins=bins)
        all_propensities.append(first_seed["propensity"].to_numpy())
        panel_data.append((severity, first_seed, support, propensity_model))

    prop_min = float(np.min(np.concatenate(all_propensities)))
    prop_max = float(np.max(np.concatenate(all_propensities)))
    propensity_cmap = plt.get_cmap("viridis")
    propensity_norm = colors.Normalize(vmin=prop_min, vmax=prop_max)
    propensity_mappable = plt.cm.ScalarMappable(
        norm=propensity_norm,
        cmap=propensity_cmap,
    )

    for ax, (severity, first_seed, support, propensity_model) in zip(
        axes.ravel(),
        panel_data,
    ):
        calibration = first_seed[first_seed["split"] == "calibration"]
        test = first_seed[first_seed["split"] == "test"]

        grid_x, grid_y, propensity = _propensity_surface(
            model=propensity_model,
            scaler=scaler,
            pca=pca,
            xlim=xlim,
            ylim=ylim,
        )
        ax.imshow(
            propensity,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            cmap=propensity_cmap,
            norm=propensity_norm,
            alpha=0.22,
            aspect="auto",
            interpolation="bilinear",
            zorder=0,
        )
        if np.ptp(propensity) > 0.0:
            ax.contour(
                grid_x,
                grid_y,
                propensity,
                levels=6,
                colors="0.35",
                linewidths=0.35,
                alpha=0.22,
                zorder=1,
            )

        ax.scatter(
            calibration["pc1"],
            calibration["pc2"],
            c="#111111",
            edgecolors="none",
            s=10,
            alpha=0.18,
            marker="o",
            zorder=2,
            label="calibration",
        )
        ax.scatter(
            test["pc1"],
            test["pc2"],
            c="#d62728",
            edgecolors="white",
            s=14,
            alpha=0.9,
            marker="o",
            linewidths=0.25,
            zorder=4,
            label="final test",
        )

        empty_test_bins = int((support["test"] == 0).sum())
        sparse_test_bins = int(
            ((support["test"] > 0) & (support["test"] <= len(seeds))).sum()
        )
        weight_max = float(first_seed["oracle_weight"].max())
        prop_std = float(first_seed["propensity"].std())
        ax.set_title(
            f"severity={severity:g} | prop sd={prop_std:.3f} | max w={weight_max:.1f}\n"
            f"empty final-test bins={empty_test_bins}, sparse={sparse_test_bins}"
        )
        ax.set_xlabel("PCA-2 PC1")
        ax.set_ylabel("PCA-2 PC2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(frameon=False, loc="best")

    for ax in axes.ravel()[len(panel_data) :]:
        ax.axis("off")

    fig.colorbar(
        propensity_mappable,
        ax=axes.ravel().tolist(),
        label="test propensity e(x)",
    )
    fig.suptitle(
        f"{dataset}: rejection-sampling covariate shift "
        f"(red markers show final normal test subset for seed {seeds[0]}; "
        f"support bins aggregate {len(seeds)} seed(s))"
    )

    output_path = output_dir / f"{dataset}_covariate_shift_2d.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)

    experiment_cfg = cfg["experiment"]
    shift_cfg = cfg["covariate_shift"]
    plot_cfg = cfg["plots"]
    datasets = [str(value) for value in _as_list(experiment_cfg["datasets"])]
    severities = [float(value) for value in _as_list(experiment_cfg["severities"])]
    seeds = _seed_list(experiment_cfg["meta_seeds"])
    output_dir = Path(plot_cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    bins = int(plot_cfg["bins"])

    for dataset in datasets:
        normal, anomaly, feature_columns = _load_dataset_data(dataset)
        output_path = _plot_dataset(
            dataset=dataset,
            normal=normal,
            anomaly=anomaly,
            feature_columns=feature_columns,
            train_split=float(cfg["splits"]["train_split"]),
            test_use_proportion=float(cfg["splits"]["test_use_proportion"]),
            test_anomaly_rate=float(cfg["splits"]["test_anomaly_rate"]),
            severities=severities,
            propensity_min=float(shift_cfg["propensity_min"]),
            propensity_max=float(shift_cfg["propensity_max"]),
            seeds=seeds,
            output_dir=output_dir,
            bins=bins,
        )
        print(output_path)


if __name__ == "__main__":
    main()
