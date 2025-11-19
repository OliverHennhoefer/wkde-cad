import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from nonconform.utils.data import load
from nonconform.strategy import JackknifeBootstrap, Probabilistic, Empirical
from nonconform.detection import ConformalDetector
from nonconform.utils.stat import false_discovery_rate, statistical_power
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
    raise ValueError("Please provide at least one seed via 'meta_seeds' or 'meta_seed' in config.")

seeds = seed_config if isinstance(seed_config, list) else [seed_config]

####################################################################################################################
# Phase 1: Model Selection
####################################################################################################################

output_dir = Path(__file__).parent / "model_selection"
output_dir.mkdir(exist_ok=True)

for ds_name in tqdm(datasets, desc="Model Selection"):
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

                n_anomalies_needed = int(len(X_normal_test) * empirical_anomaly_rate / (1 - empirical_anomaly_rate))
                n_anomalies_needed = max(n_anomalies_needed, 5)

                rng = np.random.default_rng(seed + fold_idx)
                anomaly_idx = rng.choice(len(anomaly_train), size=min(n_anomalies_needed, len(anomaly_train)), replace=False)
                X_anomaly_test = anomaly_train.iloc[anomaly_idx].drop(columns=["Class"])

                X_train = X_normal_train.values
                X_test = pd.concat([X_normal_test, X_anomaly_test]).values
                y_test = np.concatenate([np.zeros(len(X_normal_test)), np.ones(len(X_anomaly_test))])

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = get_model_instance(model_name)
                model.fit(X_train)
                y_scores = model.decision_function(X_test)

                prauc = average_precision_score(y_test, y_scores)
                rocauc = roc_auc_score(y_test, y_scores)
                y_probs = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
                brier = brier_score_loss(y_test, y_probs)

                fold_metrics.append({"dataset": ds_name, "model": model_name, "fold": fold_idx,
                                    "prauc": prauc, "rocauc": rocauc, "brier": brier})

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

            all_fold_data.append({
                "seed": seed,
                "dataset": ds_name,
                "model": model_name,
                "fold": "mean",
                "prauc": mean_prauc,
                "rocauc": mean_rocauc,
                "brier": mean_brier,
            })

        best_model = max(model_results, key=lambda x: (x["mean_prauc"], x["mean_rocauc"], -x["mean_brier"]))

        logger.info(f"{ds_name} (seed {seed}) - {best_model['model']}")

        df_all = pd.DataFrame(all_fold_data)
        df_all["is_best"] = df_all["model"] == best_model["model"]
        df_all = df_all[["seed", "dataset", "model", "fold", "prauc", "rocauc", "brier", "is_best"]]
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
n_test_fixed = cfg["global"].get("n_test_fixed")
min_anomalies = cfg["global"].get("min_anomalies", 3)

# Normalize target_anomaly_rates to list
rate_config = cfg["global"].get("target_anomaly_rates")
target_rates = rate_config if isinstance(rate_config, list) else [rate_config] if rate_config else [None]

for ds_name in tqdm(datasets, desc="Experimentation"):
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

    experiment_results = []

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

        X_train = normal_train.drop(columns=["Class"]).values
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        for target_rate in target_rates:
            # Subsample test set based on target rate or use empirical
            if target_rate is not None:
                # Use target anomaly rate
                if n_test_fixed:
                    n_anomaly_sample = max(min_anomalies, int(round(n_test_fixed * target_rate)))
                    n_normal_sample = n_test_fixed - n_anomaly_sample
                else:
                    # No fixed test size, scale based on available normals
                    n_normal_sample = len(normal_test)
                    n_anomaly_sample = max(min_anomalies, int(round(n_normal_sample * target_rate / (1 - target_rate))))

                normal_test_sampled = normal_test.sample(n=min(n_normal_sample, len(normal_test)), random_state=seed)
                anomaly_test_sampled = anomaly_test.sample(n=min(n_anomaly_sample, len(anomaly_test)), random_state=seed)
            else:
                # Use empirical rate
                if n_test_fixed and len(normal_test) + len(anomaly_test) > n_test_fixed:
                    total_test = len(normal_test) + len(anomaly_test)
                    empirical_rate = len(anomaly_test) / total_test

                    n_anomaly_sample = max(min_anomalies, int(round(n_test_fixed * empirical_rate)))
                    n_normal_sample = n_test_fixed - n_anomaly_sample

                    normal_test_sampled = normal_test.sample(n=n_normal_sample, random_state=seed)
                    anomaly_test_sampled = anomaly_test.sample(n=n_anomaly_sample, random_state=seed)
                else:
                    normal_test_sampled = normal_test
                    anomaly_test_sampled = anomaly_test

            X_test = pd.concat([normal_test_sampled, anomaly_test_sampled]).drop(columns=["Class"]).values
            y_test = np.concatenate([np.zeros(len(normal_test_sampled)), np.ones(len(anomaly_test_sampled))])
            X_test_scaled = scaler.transform(X_test)

            actual_rate = len(anomaly_test_sampled) / len(X_test)

            # Define approaches
            approaches = {
                "empirical": {
                    "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
                    "estimation": Empirical()
                },
                "probabilistic": {
                    "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
                    "estimation": Probabilistic(n_trials=n_trials)
                }
            }

            for approach_name, approach_config in approaches.items():
                detector_kwargs = {
                    "detector": get_model_instance(model_name),
                    "strategy": approach_config["strategy"],
                    "seed": seed
                }
                if approach_config["estimation"] is not None:
                    detector_kwargs["estimation"] = approach_config["estimation"]

                detector = ConformalDetector(**detector_kwargs)
                detector.fit(X_train_scaled)
                p_values = detector.predict(X_test_scaled)

                decisions = false_discovery_control(p_values, method="bh") <= fdr_rate

                fdr = false_discovery_rate(y=y_test, y_hat=decisions)
                power = statistical_power(y=y_test, y_hat=decisions)

                experiment_results.append({
                    "seed": seed,
                    "dataset": ds_name,
                    "model": model_name,
                    "approach": approach_name,
                    "target_anomaly_rate": round(actual_rate, 3),
                    "n_train": len(X_train_scaled),
                    "n_test": len(X_test_scaled),
                    "n_test_normal": len(normal_test_sampled),
                    "n_test_anomaly": len(anomaly_test_sampled),
                    "fdr": round(fdr, 3),
                    "power": round(power, 3)
                })

                logger.info(f"{ds_name} (seed {seed}, {approach_name}, rate={actual_rate:.3f}) - FDR: {fdr:.3f}, Power: {power:.3f}")

    # Create DataFrame and add summary rows
    df_results = pd.DataFrame(experiment_results)

    # Group by approach and target_anomaly_rate for summary
    group_cols = ["dataset", "model", "approach", "target_anomaly_rate"]
    summary = df_results.groupby(group_cols).agg(
        fdr_mean=("fdr", "mean"),
        fdr_std=("fdr", "std"),
        power_mean=("power", "mean"),
        power_std=("power", "std"),
        n_train=("n_train", "first"),
        n_test=("n_test", "first"),
        n_test_normal=("n_test_normal", "first"),
        n_test_anomaly=("n_test_anomaly", "first"),
    ).reset_index()

    # Format summary rows
    summary["seed"] = "mean"
    summary["fdr"] = summary.apply(lambda r: f"{r['fdr_mean']:.3f}±{r['fdr_std']:.3f}", axis=1)
    summary["power"] = summary.apply(lambda r: f"{r['power_mean']:.3f}±{r['power_std']:.3f}", axis=1)

    # Combine results with summary and save
    df_final = pd.concat([df_results, summary[df_results.columns]], ignore_index=True)
    df_final.to_csv(results_csv, index=False)

