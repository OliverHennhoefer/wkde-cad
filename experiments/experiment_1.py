import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from nonconform.utils.data import load
from code.utils.registry import get_dataset_enum, get_model_instance
from code.utils.logger import get_logger

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

####################################################################################################################
# Dataset Setup
####################################################################################################################

for ds_name in tqdm(datasets, desc="Datasets"):
    dataset_enum = get_dataset_enum(ds_name)
    data = load(dataset_enum, setup=False)

    normal = data[data["Class"] == 0]
    anomaly = data[data["Class"] == 1]

    normal_train, normal_test = train_test_split(
        normal,
        train_size=cfg["global"]["train_split"],
        random_state=cfg["global"]["meta_seed"],
    )
    anomaly_train, anomaly_test = train_test_split(
        anomaly,
        train_size=cfg["global"]["train_split"],
        random_state=cfg["global"]["meta_seed"],
    )

    ####################################################################################################################
    # Model Selection
    ####################################################################################################################

    empirical_anomaly_rate = len(anomaly) / len(data)
    models = cfg["experiments"]["models"]
    models = models if isinstance(models, list) else [models]

    output_dir = Path(__file__).parent / "model_selection"
    output_dir.mkdir(exist_ok=True)

    model_results = []
    all_fold_data = []

    for model_name in models:
        if model_name.lower() == "mcd" and ds_name.lower() == "musk":
            continue

        kf = KFold(n_splits=cfg["global"]["selection_folds"], shuffle=True,
                   random_state=cfg["global"]["meta_seed"])
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(normal_train)):
            X_normal_train = normal_train.iloc[train_idx].drop(columns=["Class"])
            X_normal_test = normal_train.iloc[test_idx].drop(columns=["Class"])

            n_anomalies_needed = int(len(X_normal_test) * empirical_anomaly_rate / (1 - empirical_anomaly_rate))
            n_anomalies_needed = max(n_anomalies_needed, 5)

            rng = np.random.default_rng(cfg["global"]["meta_seed"] + fold_idx)
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

        all_fold_data.append({"dataset": ds_name, "model": model_name, "fold": "mean",
                             "prauc": mean_prauc, "rocauc": mean_rocauc, "brier": mean_brier})

    best_model = max(model_results, key=lambda x: (x["mean_prauc"], x["mean_rocauc"], -x["mean_brier"]))

    logger.info(f"{ds_name} - {best_model["model"]}")

    df_all = pd.DataFrame(all_fold_data)
    df_all.to_csv(output_dir / f"{ds_name}.csv", index=False)

    ####################################################################################################################
    # Experimentation
    ####################################################################################################################

    # TODO: Add StandardScaler (fit_transform on train, transform on test)

