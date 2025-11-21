import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from nonconform.utils.data import load
from nonconform.strategy import JackknifeBootstrap, Probabilistic, Empirical
from nonconform.detection import ConformalDetector
from nonconform.detection.weight import (
    BootstrapBaggedWeightEstimator,
    LogisticWeightEstimator,
)
from nonconform.utils.stat import (
    false_discovery_rate,
    statistical_power,
    weighted_false_discovery_control,
)
from code.utils.registry import get_dataset_enum, get_model_instance
from code.utils.logger import get_logger

logging.getLogger("nonconform").setLevel(logging.CRITICAL)

####################################################################################################################
# Setup
####################################################################################################################

logger = get_logger()

# Load configuration
config_path = Path(__file__).parent.parent / "code" / "config.toml"
with open(config_path, "rb") as f:
    cfg = tomllib.load(f)

datasets = cfg["experiments"]["datasets"]
datasets = datasets if isinstance(datasets, list) else [datasets]

seed_config = cfg["global"].get("meta_seeds", cfg["global"].get("meta_seed"))
if seed_config is None:
    raise ValueError(
        "Please provide at least one seed via 'meta_seeds' or 'meta_seed' in config."
    )

seeds = seed_config if isinstance(seed_config, list) else [seed_config]

####################################################################################################################
# Phase 1: Model Selection
####################################################################################################################

output_dir = Path(__file__).parent / "model_selection"
output_dir.mkdir(exist_ok=True)

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
    models = cfg["experiments"]["models"]
    models = models if isinstance(models, list) else [models]

    for seed in seeds:
        normal_train, normal_test = train_test_split(
            normal,
            train_size=cfg["global"]["train_split"],
            random_state=seed,
        )
        anomaly_train, anomaly_test = train_test_split(
            anomaly,
            train_size=cfg["global"]["train_split"],
            random_state=seed,
        )

        model_results = []
        all_fold_data = []

        for model_name in models:
            kf = KFold(
                n_splits=cfg["global"]["selection_folds"],
                shuffle=True,
                random_state=seed,
            )
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(normal_train)):
                X_normal_train = normal_train.iloc[train_idx].drop(columns=["Class"])
                X_normal_test = normal_train.iloc[test_idx].drop(columns=["Class"])

                n_anomalies_needed = int(
                    len(X_normal_test)
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

                X_train = X_normal_train.values
                X_test = pd.concat([X_normal_test, X_anomaly_test]).values
                y_test = np.concatenate(
                    [np.zeros(len(X_normal_test)), np.ones(len(X_anomaly_test))]
                )

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = get_model_instance(model_name)
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

        logger.info(f"{ds_name} (seed={seed}) - {best_model['model']}")

        df_all = pd.DataFrame(all_fold_data)
        df_all["is_best"] = df_all["model"] == best_model["model"]
        df_all = df_all[
            ["seed", "dataset", "model", "fold", "prauc", "rocauc", "brier", "is_best"]
        ]
        header = not csv_path.exists()
        df_all.to_csv(csv_path, mode="a", header=header, index=False)

####################################################################################################################
# Phase 2: Experimentation
####################################################################################################################

results_dir = Path(__file__).parent / "results" / "experiment1"
results_dir.mkdir(parents=True, exist_ok=True)

fdr_rate = cfg["global"]["fdr_rate"]
n_bootstraps = cfg["global"]["n_bootstraps"]
n_trials = cfg["global"]["n_trials"]
test_use_proportion = cfg["global"]["test_use_proportion"]
test_anomaly_rate = cfg["global"]["test_anomaly_rate"]

# Load approaches to run
approaches_to_run = cfg["global"]["approaches"]
approaches_to_run = (
    approaches_to_run if isinstance(approaches_to_run, list) else [approaches_to_run]
)

for ds_name in datasets:
    results_csv = results_dir / f"{ds_name}.csv"

    if results_csv.exists():
        logger.info(f"Skipping experimentation for {ds_name} (results exist)")
        continue

    selection_csv = output_dir / f"{ds_name}.csv"
    if not selection_csv.exists():
        logger.warning(f"No model selection results for {ds_name}, skipping")
        continue

    selection_df = pd.read_csv(selection_csv)
    best_models = selection_df[selection_df["is_best"]]

    dataset_enum = get_dataset_enum(ds_name)
    data = load(dataset_enum, setup=False)

    normal = data[data["Class"] == 0]
    anomaly = data[data["Class"] == 1]

    # Track whether CSV header has been written
    csv_header_written = False

    for _, row in best_models.iterrows():
        seed = int(row["seed"])
        model_name = row["model"]

        normal_train, normal_test = train_test_split(
            normal,
            train_size=cfg["global"]["train_split"],
            random_state=seed,
        )
        anomaly_train, anomaly_test = train_test_split(
            anomaly,
            train_size=cfg["global"]["train_split"],
            random_state=seed,
        )

        # Use all training data
        X_train = normal_train.drop(columns=["Class"]).values
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        train_size = len(X_train_scaled)

        # Compute test set sizes from configuration
        total_test_available = len(normal_test) + len(anomaly_test)
        test_size = round(test_use_proportion * total_test_available)
        n_anomalies_test = round(test_size * test_anomaly_rate)
        n_normal_test = test_size - n_anomalies_test

        # Ensure at least 1 anomaly
        if n_anomalies_test < 1:
            logger.warning(
                f"{ds_name}: computed n_anomalies_test={n_anomalies_test} < 1, using 1"
            )
            n_anomalies_test = 1
            n_normal_test = test_size - 1

        # Validate sufficient test samples
        if n_normal_test < 1:
            logger.error(
                f"{ds_name}: invalid configuration resulted in n_normal_test < 1, skipping"
            )
            continue
        if n_normal_test > len(normal_test):
            logger.warning(
                f"{ds_name}: computed n_normal_test {n_normal_test} exceeds available normal test samples ({len(normal_test)}), skipping"
            )
            continue
        if n_anomalies_test > len(anomaly_test):
            logger.warning(
                f"{ds_name}: computed n_anomalies_test {n_anomalies_test} exceeds available anomaly test samples ({len(anomaly_test)}), skipping"
            )
            continue

        # Sample test data with computed sizes
        normal_test_sampled = normal_test.sample(n=n_normal_test, random_state=seed)
        anomaly_test_sampled = anomaly_test.sample(
            n=n_anomalies_test, random_state=seed
        )

        X_test = (
            pd.concat([normal_test_sampled, anomaly_test_sampled])
            .drop(columns=["Class"])
            .values
        )
        y_test = np.concatenate(
            [
                np.zeros(len(normal_test_sampled)),
                np.ones(len(anomaly_test_sampled)),
            ]
        )
        X_test_scaled = scaler.transform(X_test)

        actual_anomaly_rate = n_anomalies_test / test_size

        # Create weight estimator for weighted approaches
        weight_estimator = BootstrapBaggedWeightEstimator(
            base_estimator=LogisticWeightEstimator(),
            n_bootstrap=n_bootstraps,
        )

        # Define all four approaches
        all_approaches = {
            "empirical": {
                "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
                "estimation": Empirical(),
                "weight_estimator": None,
                "weighted": False,
            },
            "probabilistic": {
                "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
                "estimation": Probabilistic(n_trials=n_trials),
                "weight_estimator": None,
                "weighted": False,
            },
            "empirical_weighted": {
                "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
                "estimation": Empirical(),
                "weight_estimator": weight_estimator,
                "weighted": True,
            },
            "probabilistic_weighted": {
                "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
                "estimation": Probabilistic(n_trials=n_trials),
                "weight_estimator": weight_estimator,
                "weighted": True,
            },
        }

        # Filter to only configured approaches
        approaches = {
            k: v for k, v in all_approaches.items() if k in approaches_to_run
        }

        # Validate that all configured approaches are valid
        invalid_approaches = set(approaches_to_run) - set(all_approaches.keys())
        if invalid_approaches:
            raise ValueError(
                f"Invalid approaches in config: {invalid_approaches}. "
                f"Valid options are: {list(all_approaches.keys())}"
            )

        for approach_name, approach_config in approaches.items():
            detector_kwargs = {
                "detector": get_model_instance(model_name),
                "strategy": approach_config["strategy"],
                "seed": seed,
            }
            if approach_config["estimation"] is not None:
                detector_kwargs["estimation"] = approach_config["estimation"]
            if approach_config["weight_estimator"] is not None:
                detector_kwargs["weight_estimator"] = approach_config[
                    "weight_estimator"
                ]

            detector = ConformalDetector(**detector_kwargs)
            detector.fit(X_train_scaled)
            p_values = detector.predict(X_test_scaled)

            # Apply appropriate FDR control
            if approach_config["weighted"]:
                # Use weighted FDR control for weighted approaches
                decisions = weighted_false_discovery_control(
                    detector.last_result, alpha=fdr_rate
                )
            else:
                # Use standard BH FDR control for non-weighted approaches
                decisions = (
                    false_discovery_control(p_values, method="bh") <= fdr_rate
                )

            fdr = false_discovery_rate(y=y_test, y_hat=decisions)
            power = statistical_power(y=y_test, y_hat=decisions)

            # Write result immediately to CSV
            result_row = pd.DataFrame(
                [
                    {
                        "seed": seed,
                        "dataset": ds_name,
                        "model": model_name,
                        "approach": approach_name,
                        "train_size": train_size,
                        "test_size": test_size,
                        "n_train": len(X_train_scaled),
                        "n_test": test_size,
                        "n_test_normal": n_normal_test,
                        "n_test_anomaly": n_anomalies_test,
                        "actual_anomaly_rate": round(actual_anomaly_rate, 3),
                        "fdr": round(fdr, 3),
                        "power": round(power, 3),
                    }
                ]
            )
            result_row.to_csv(
                results_csv,
                mode="a",
                header=not csv_header_written,
                index=False,
            )
            csv_header_written = True

            logger.info(
                f"{ds_name} (seed {seed}, {approach_name}, train={train_size}, test={test_size}, anom_rate={actual_anomaly_rate:.3f}) - FDR: {fdr:.3f}, Power: {power:.3f}"
            )

    # Read the incrementally written results and compute summary rows
    df_results = pd.read_csv(results_csv)

    # Group by approach, train_size, and test_size for summary (aggregate across all models/trials)
    group_cols = ["dataset", "approach", "train_size", "test_size"]
    summary = (
        df_results.groupby(group_cols)
        .agg(
            fdr_mean=("fdr", "mean"),
            fdr_std=("fdr", "std"),
            power_mean=("power", "mean"),
            power_std=("power", "std"),
            n_train=("n_train", "first"),
            n_test=("n_test", "first"),
            n_test_normal=("n_test_normal", "first"),
            n_test_anomaly=("n_test_anomaly", "first"),
            actual_anomaly_rate=("actual_anomaly_rate", "first"),
        )
        .reset_index()
    )

    # Format summary rows
    summary["seed"] = "mean"
    summary["model"] = "all_models"
    summary["fdr"] = summary.apply(
        lambda r: f"{r['fdr_mean']:.3f}±{r['fdr_std']:.3f}", axis=1
    )
    summary["power"] = summary.apply(
        lambda r: f"{r['power_mean']:.3f}±{r['power_std']:.3f}", axis=1
    )

    # Append summary rows to the existing CSV
    summary[df_results.columns].to_csv(results_csv, mode="a", header=False, index=False)
