from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
) -> pd.DataFrame:
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

    return pd.concat(rows, ignore_index=True)


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
    scaled = StandardScaler().fit_transform(features)
    embedding = PCA(n_components=2, random_state=0).fit_transform(scaled)
    embedding_df = pd.DataFrame(
        {
            "pc1": embedding[:, 0],
            "pc2": embedding[:, 1],
        },
        index=normal.index,
    )

    n_cols = min(3, len(severities))
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
        diagnostics = _split_diagnostics(
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
        panel_data.append((severity, first_seed, support))

    prop_min = float(np.min(np.concatenate(all_propensities)))
    prop_max = float(np.max(np.concatenate(all_propensities)))

    for ax, (severity, first_seed, support) in zip(axes.ravel(), panel_data):
        calibration = first_seed[first_seed["split"] == "calibration"]
        test = first_seed[first_seed["split"] == "test"]

        ax.scatter(
            calibration["pc1"],
            calibration["pc2"],
            c=calibration["propensity"],
            cmap="viridis",
            vmin=prop_min,
            vmax=prop_max,
            s=18,
            alpha=0.28,
            marker="o",
            linewidths=0,
            label="calibration",
        )
        points = ax.scatter(
            test["pc1"],
            test["pc2"],
            c=test["propensity"],
            cmap="viridis",
            vmin=prop_min,
            vmax=prop_max,
            s=30,
            alpha=0.86,
            marker="^",
            edgecolors="black",
            linewidths=0.25,
            label="test",
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
        ax.legend(frameon=False, loc="best")

    for ax in axes.ravel()[len(panel_data):]:
        ax.axis("off")

    fig.colorbar(points, ax=axes.ravel().tolist(), label="test propensity e(x)")
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
