import logging
import multiprocessing as mp
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from src.utils.data_loader import load
from nonconform import (
    ConformalDetector,
    JackknifeBootstrap,
    Probabilistic,
    Empirical,
)
from nonconform.metrics import false_discovery_rate, statistical_power
from nonconform.fdr import weighted_false_discovery_control, Pruning
from nonconform.weighting import (
    BootstrapBaggedWeightEstimator,
    forest_weight_estimator,
)
from src.utils.registry import get_dataset_enum, get_model_instance
from src.utils.logger import get_logger

logging.getLogger("nonconform").setLevel(logging.CRITICAL)

# Auto-detect CPU count: use max 2 cores locally, scale up on servers
N_JOBS = min(mp.cpu_count(), 2) if mp.cpu_count() <= 4 else max(1, mp.cpu_count() - 2)

####################################################################################################################
# Setup
####################################################################################################################

logger = get_logger()

# Load configuration
repo_root = Path(__file__).resolve().parent.parent
config_path = Path(__file__).resolve().parent / "config.toml"
with open(config_path, "rb") as f:
    cfg = tomllib.load(f)

datasets = cfg["experiments"]["datasets"]
datasets = datasets if isinstance(datasets, list) else [datasets]

seed_config = cfg["global"].get("meta_seeds")
if seed_config is None:
    raise ValueError(
        "Please provide 'meta_seeds' as an integer count in config."
    )
if not isinstance(seed_config, int) or isinstance(seed_config, bool):
    raise ValueError(
        "meta_seeds must be an integer count (e.g., 20 for seeds 1..20)."
    )
if seed_config < 1:
    raise ValueError("meta_seeds must be >= 1.")

seeds = list(range(1, seed_config + 1))
pruning_choice = str(cfg["global"].get("pruning", "heterogeneous")).strip().lower()
pruning_method = {
    "deterministic": Pruning.DETERMINISTIC,
    "homogeneous": Pruning.HOMOGENEOUS,
    "heterogeneous": Pruning.HETEROGENEOUS,
}[pruning_choice]

####################################################################################################################
# Function Definitions
####################################################################################################################


def process_seed_phase1(seed, ds_name, normal, anomaly, empirical_anomaly_rate, models, cfg, output_dir):
    """Process a single seed for Phase 1: Model Selection."""
    normal_train, _ = train_test_split(
        normal,
        train_size=cfg["global"]["train_split"],
        random_state=seed,
    )
    anomaly_train, _ = train_test_split(
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


def process_seed_phase2(seed, model_name, ds_name, normal, anomaly, cfg, fdr_rate,
                        n_bootstraps, n_trials, test_use_proportion, test_anomaly_rate,
                        approaches_to_run, results_dir):
    """Process a single seed for Phase 2: Experimentation."""
    normal_train, normal_test = train_test_split(
        normal,
        train_size=cfg["global"]["train_split"],
        random_state=seed,
    )
    _, anomaly_test = train_test_split(
        anomaly,
        train_size=cfg["global"]["train_split"],
        random_state=seed,
    )

    # Use all training data
    X_train = normal_train.drop(columns=["Class"]).values
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    train_size = len(x_train_scaled)

    # Compute test set sizes from configuration
    total_test_available = len(normal_test) + len(anomaly_test)
    test_size = round(test_use_proportion * total_test_available)
    n_anomalies_test = round(test_size * test_anomaly_rate)
    n_normal_test = test_size - n_anomalies_test

    # Ensure at least 1 anomaly
    if n_anomalies_test < 1:
        n_anomalies_test = 1
        n_normal_test = test_size - 1

    # Cap anomalies to available amount (use all if not enough)
    if n_anomalies_test > len(anomaly_test):
        n_anomalies_test = len(anomaly_test)
        # Recalculate normal samples to maintain test_size
        n_normal_test = test_size - n_anomalies_test

    # Cap normal samples to available amount
    if n_normal_test > len(normal_test):
        n_normal_test = len(normal_test)

    # Validate we have at least 1 of each
    if n_normal_test < 1 or n_anomalies_test < 1:
        return None  # Skip this seed

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
    x_test_scaled = scaler.transform(X_test)

    actual_anomaly_rate = n_anomalies_test / test_size

    weight_choice = cfg["global"].get("weight_estimator", "forest")
    weight_choice = str(weight_choice).strip().lower()
    if weight_choice == "forest":
        weight_estimator = forest_weight_estimator()
    elif weight_choice == "forest_bagged":
        weight_estimator = BootstrapBaggedWeightEstimator(
            base_estimator=forest_weight_estimator(),
            n_bootstraps=n_bootstraps,
        )
    else:
        raise ValueError(
            f"Invalid weight_estimator '{weight_choice}'. "
            "Valid options are: 'forest', 'forest_bagged'."
        )

    # Define all four approaches
    all_approaches = {
        "empirical": {
            "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
            "estimation": Empirical(tie_break="classical"),
            "weight_estimator": None,
            "weighted": False,
        },
        "empirical_randomized": {
            "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
            "estimation": Empirical(tie_break="randomized"),
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
            "estimation": Empirical(tie_break="classical"),
            "weight_estimator": weight_estimator,
            "weighted": True,
        },
        "empirical_randomized_weighted": {
            "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
            "estimation": Empirical(tie_break="randomized"),
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

    # Collect all results for this seed
    seed_results = []

    for approach_name, approach_config in approaches.items():
        detector_kwargs = {
            "detector": get_model_instance(model_name, random_state=seed),
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
        detector.fit(x_train_scaled)
        p_values = detector.compute_p_values(x_test_scaled)

        # Apply appropriate FDR control
        if approach_config["weighted"]:
            decisions = weighted_false_discovery_control(
                detector.last_result, alpha=fdr_rate,
                pruning=pruning_method, seed=seed
            )
        else:
            decisions = (
                false_discovery_control(p_values, method="bh") <= fdr_rate
            )

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        seed_results.append({
            "seed": seed,
            "dataset": ds_name,
            "model": model_name,
            "approach": approach_name,
            "train_size": train_size,
            "test_size": test_size,
            "n_train": len(x_train_scaled),
            "n_test": test_size,
            "n_test_normal": n_normal_test,
            "n_test_anomaly": n_anomalies_test,
            "actual_anomaly_rate": round(actual_anomaly_rate, 3),
            "fdr": round(fdr, 3),
            "power": round(power, 3),
        })

    # Write all results for this seed to a seed-specific CSV
    csv_path = results_dir / f"{ds_name}_seed{seed}.csv"
    df = pd.DataFrame(seed_results)
    df.to_csv(csv_path, index=False)

    return seed, seed_results


####################################################################################################################
# Main Execution
####################################################################################################################

if __name__ == '__main__':
    ####################################################################################################################
    # Phase 1: Model Selection
    ####################################################################################################################

    output_dir = repo_root / "outputs" / "model_selection"
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

        # Process seeds in parallel
        with mp.Pool(N_JOBS) as pool:
            results = pool.starmap(
                process_seed_phase1,
                [(seed, ds_name, normal, anomaly, empirical_anomaly_rate, models, cfg, output_dir)
                 for seed in seeds]
            )

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

    ####################################################################################################################
    # Phase 2: Experimentation
    ####################################################################################################################

    results_dir = repo_root / "outputs" / "experiment_results"
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

        # Validate configured approaches before processing
        all_approaches_keys = {
            "empirical",
            "empirical_randomized",
            "probabilistic",
            "empirical_weighted",
            "empirical_randomized_weighted",
            "probabilistic_weighted",
        }
        invalid_approaches = set(approaches_to_run) - all_approaches_keys
        if invalid_approaches:
            raise ValueError(
                f"Invalid approaches in config: {invalid_approaches}. "
                f"Valid options are: {list(all_approaches_keys)}"
            )

        # Process seeds in parallel
        seed_model_pairs = [(int(row["seed"]), row["model"]) for _, row in best_models.iterrows()]

        with mp.Pool(N_JOBS) as pool:
            results = pool.starmap(
                process_seed_phase2,
                [(seed, model_name, ds_name, normal, anomaly, cfg, fdr_rate,
                  n_bootstraps, n_trials, test_use_proportion, test_anomaly_rate,
                  approaches_to_run, results_dir)
                 for seed, model_name in seed_model_pairs]
            )

        # Filter out None results (skipped seeds) and log results
        results = [r for r in results if r is not None]
        for seed_result, seed_data in results:
            for row_data in seed_data:
                logger.info(
                    f"{row_data['dataset']} (seed {row_data['seed']}, {row_data['approach']}, "
                    f"train={row_data['train_size']}, test={row_data['test_size']}, "
                    f"anom_rate={row_data['actual_anomaly_rate']:.3f}) - "
                    f"FDR: {row_data['fdr']:.3f}, Power: {row_data['power']:.3f}"
                )

        # Merge per-seed CSV files into final dataset CSV
        seed_csvs = [results_dir / f"{ds_name}_seed{seed}.csv" for seed, _ in seed_model_pairs]
        dfs = [pd.read_csv(csv) for csv in seed_csvs if csv.exists()]
        if not dfs:
            logger.warning(f"No valid results for {ds_name}, skipping summary computation")
            continue

        df_results = pd.concat(dfs, ignore_index=True)
        df_results.to_csv(results_csv, index=False)

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

        # Clean up temporary per-seed files
        for csv in seed_csvs:
            if csv.exists():
                csv.unlink()
