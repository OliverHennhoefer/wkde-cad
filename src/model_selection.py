import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.data_loader import load
from src.utils.registry import get_dataset_enum, get_model_instance


def process_seed_phase1(
    seed,
    ds_name,
    normal,
    anomaly,
    empirical_anomaly_rate,
    models,
    cfg,
    output_dir,
):
    """Process a single seed for Phase 1: Model Selection."""
    normal_train, _ = train_test_split(
        normal,
        train_size=cfg["splits"]["train_split"],
        random_state=seed,
    )
    anomaly_train, _ = train_test_split(
        anomaly,
        train_size=cfg["splits"]["train_split"],
        random_state=seed,
    )

    model_results = []
    all_fold_data = []

    for model_name in models:
        kf = KFold(
            n_splits=cfg["model_selection"]["folds"],
            shuffle=True,
            random_state=seed,
        )
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(normal_train)):
            x_normal_train = normal_train.iloc[train_idx].drop(columns=["Class"])
            x_normal_test = normal_train.iloc[test_idx].drop(columns=["Class"])

            n_anomalies_needed = int(
                len(x_normal_test)
                * empirical_anomaly_rate
                / (1 - empirical_anomaly_rate)
            )
            n_anomalies_needed = max(n_anomalies_needed, 5)

            rng = np.random.default_rng(seed + fold_idx)
            anomaly_idx = rng.choice(
                len(anomaly_train),
                size=min(n_anomalies_needed, len(anomaly_train)),
                replace=False,
            )
            X_anomaly_test = anomaly_train.iloc[anomaly_idx].drop(columns=["Class"])

            X_train = x_normal_train.values
            X_test = pd.concat([x_normal_test, X_anomaly_test]).values
            y_test = np.concatenate(
                [np.zeros(len(x_normal_test)), np.ones(len(X_anomaly_test))]
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = get_model_instance(model_name, random_state=seed)
            model.fit(X_train)
            y_scores = model.decision_function(X_test)

            prauc = average_precision_score(y_test, y_scores)
            rocauc = roc_auc_score(y_test, y_scores)
            y_probs = (y_scores - y_scores.min()) / (
                y_scores.max() - y_scores.min()
            )
            brier = brier_score_loss(y_test, y_probs)

            fold_metrics.append(
                {
                    "dataset": ds_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "prauc": prauc,
                    "rocauc": rocauc,
                    "brier": brier,
                }
            )

        df_folds = pd.DataFrame(fold_metrics)
        mean_prauc = df_folds["prauc"].mean()
        mean_rocauc = df_folds["rocauc"].mean()
        mean_brier = df_folds["brier"].mean()

        results = {
            "dataset": ds_name,
            "model": model_name,
            "mean_prauc": mean_prauc,
            "std_prauc": df_folds["prauc"].std(),
            "mean_rocauc": mean_rocauc,
            "std_rocauc": df_folds["rocauc"].std(),
            "mean_brier": mean_brier,
            "std_brier": df_folds["brier"].std(),
        }
        model_results.append(results)

        all_fold_data.append(
            {
                "seed": seed,
                "dataset": ds_name,
                "model": model_name,
                "fold": "mean",
                "prauc": mean_prauc,
                "rocauc": mean_rocauc,
                "brier": mean_brier,
            }
        )

    best_model = max(
        model_results,
        key=lambda x: (x["mean_prauc"], x["mean_rocauc"], -x["mean_brier"]),
    )

    # Write to seed-specific CSV (avoids race conditions)
    csv_path = output_dir / f"{ds_name}_seed{seed}.csv"
    df_all = pd.DataFrame(all_fold_data)
    df_all["is_best"] = df_all["model"] == best_model["model"]
    df_all = df_all[
        ["seed", "dataset", "model", "fold", "prauc", "rocauc", "brier", "is_best"]
    ]
    df_all.to_csv(csv_path, mode="w", header=True, index=False)

    return ds_name, seed, best_model["model"]


def run_model_selection(
    *,
    cfg: dict[str, Any],
    datasets: list[str],
    seeds: list[int],
    output_dir: Path,
    jobs: int,
    logger,
) -> None:
    """Run model selection for datasets missing final selection CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in datasets:
        csv_path = output_dir / f"{ds_name}.csv"

        if csv_path.exists():
            logger.info(f"Skipping model selection for {ds_name} (results exist)")
            continue

        dataset_enum = get_dataset_enum(ds_name)
        data = load(dataset_enum, setup=False)

        normal = data[data["Class"] == 0]
        anomaly = data[data["Class"] == 1]

        empirical_anomaly_rate = len(anomaly) / len(data)
        models = cfg["model_selection"]["models"]
        models = models if isinstance(models, list) else [models]

        tasks = [
            (
                seed,
                ds_name,
                normal,
                anomaly,
                empirical_anomaly_rate,
                models,
                cfg,
                output_dir,
            )
            for seed in seeds
        ]

        if jobs == 1:
            results = [process_seed_phase1(*task) for task in tasks]
        else:
            with mp.Pool(jobs) as pool:
                results = pool.starmap(process_seed_phase1, tasks)

        # Log results from parallel execution
        for ds_name_result, seed_result, best_model_name in results:
            logger.info(f"{ds_name_result} (seed={seed_result}) - {best_model_name}")

        # Merge per-seed CSV files into final dataset CSV
        seed_csvs = [output_dir / f"{ds_name}_seed{seed}.csv" for seed in seeds]
        dfs = [pd.read_csv(csv) for csv in seed_csvs if csv.exists()]
        if dfs:
            df_merged = pd.concat(dfs, ignore_index=True)
            df_merged.to_csv(csv_path, index=False)
            # Clean up temporary per-seed files
            for csv in seed_csvs:
                if csv.exists():
                    csv.unlink()
