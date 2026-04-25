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
from sklearn.preprocessing import StandardScaler

from src.rebuttal.covariate_shift import fit_propensity_model, rejection_sample
from src.utils.data_loader import load
from src.utils.registry import get_dataset_enum


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.toml"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _seed_list(seed_count: int) -> list[int]:
    if not isinstance(seed_count, int) or isinstance(seed_count, bool) or seed_count < 1:
        raise ValueError("global.meta_seeds must be a positive integer count.")
    return list(range(1, seed_count + 1))


def _load_normal_data(dataset: str) -> tuple[pd.DataFrame, list[str]]:
    data = load(get_dataset_enum(dataset), setup=False)
    normal = data[data["Class"] == 0].copy()
    feature_columns = [col for col in normal.columns if col != "Class"]
    return normal, feature_columns


def _split_diagnostics(
    *,
    normal: pd.DataFrame,
    feature_columns: list[str],
    train_split: float,
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
        sample = rejection_sample(
            normal,
            feature_columns,
            model,
            seed=seed,
        )
        split_by_index = pd.Series("calibration", index=normal.index)
        split_by_index.loc[sample.accepted.index] = "test"

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
    if "calibration" not in counts:
        counts["calibration"] = 0
    if "test" not in counts:
        counts["test"] = 0
    counts["test_fraction"] = counts["test"] / (counts["test"] + counts["calibration"])
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
    feature_columns: list[str],
    train_split: float,
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
            feature_columns=feature_columns,
            train_split=train_split,
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
            test["pc1"],
            test["pc2"],
            c="#d62728",
            edgecolors="white",
            s=16,
            alpha=0.88,
            marker="o",
            linewidths=0.25,
            zorder=4,
            label="test",
        )
        ax.scatter(
            calibration["pc1"],
            calibration["pc2"],
            c="#111111",
            edgecolors="white",
            s=12,
            alpha=0.68,
            marker="o",
            linewidths=0.2,
            zorder=5,
            label="calibration",
        )

        empty_calib_bins = int((support["calibration"] == 0).sum())
        sparse_calib_bins = int(((support["calibration"] > 0) & (support["calibration"] <= len(seeds))).sum())
        weight_max = float(first_seed["oracle_weight"].max())
        prop_std = float(first_seed["propensity"].std())
        ax.set_title(
            f"severity={severity:g} | prop sd={prop_std:.3f} | max w={weight_max:.1f}\n"
            f"empty calib bins={empty_calib_bins}, sparse={sparse_calib_bins}"
        )
        ax.set_xlabel("PCA-2 PC1")
        ax.set_ylabel("PCA-2 PC2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(frameon=False, loc="best")

    for ax in axes.ravel()[len(panel_data):]:
        ax.axis("off")

    fig.colorbar(
        propensity_mappable,
        ax=axes.ravel().tolist(),
        label="test propensity e(x)",
    )
    fig.suptitle(
        f"{dataset}: rejection-sampling covariate shift "
        f"(markers show first seed {seeds[0]}; support bins aggregate {len(seeds)} seed(s))"
    )

    output_path = output_dir / f"{dataset}_covariate_shift_2d.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)

    rebuttal_cfg = cfg["rebuttal_covariate_shift"]
    datasets = [str(value) for value in _as_list(cfg["experiments"]["datasets"])]
    severities = [
        float(value) for value in _as_list(rebuttal_cfg["severities"])
    ]
    seeds = _seed_list(cfg["global"]["meta_seeds"])
    output_dir = Path(rebuttal_cfg["plot_output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    bins = int(rebuttal_cfg["plot_bins"])

    for dataset in datasets:
        normal, feature_columns = _load_normal_data(dataset)
        output_path = _plot_dataset(
            dataset=dataset,
            normal=normal,
            feature_columns=feature_columns,
            train_split=float(cfg["global"]["train_split"]),
            severities=severities,
            propensity_min=float(rebuttal_cfg["propensity_min"]),
            propensity_max=float(rebuttal_cfg["propensity_max"]),
            seeds=seeds,
            output_dir=output_dir,
            bins=bins,
        )
        print(output_path)


if __name__ == "__main__":
    main()
