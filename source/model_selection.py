"""Model selection for anomaly detection using PRAUC metric."""

import logging
from typing import List, Dict, Any, Type

import numpy as np
import optuna
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from nonconform.utils.data import Dataset
from source.split import StratifiedSplitter

logger = logging.getLogger(__name__)

SKIP_COMBINATIONS = {
    ("MUSK", "MCD"),
}


def _create_failure_result(
    dataset: Dataset, model_class: Type, error: Exception
) -> Dict[str, Any]:
    """Create empty result dict for failed evaluation."""
    error_msg = str(error).replace("\n", " ")[:200]
    return {
        "dataset": dataset.name,
        "model": model_class.__name__,
        "mean_prauc": None,
        "std_prauc": None,
        "mean_auroc": None,
        "std_auroc": None,
        "mean_brier": None,
        "std_brier": None,
        "prauc_scores": [],
        "auroc_scores": [],
        "brier_scores": [],
        "n_runs": 0,
        "failed": True,
        "error": error_msg,
    }


def evaluate_model_on_dataset(
    dataset: Dataset,
    model_class: Type,
    n_runs: int = 10,
) -> Dict[str, Any]:
    """Evaluate a PyOD model on a dataset with multiple seeded runs."""
    prauc_scores = []
    auroc_scores = []
    brier_scores = []

    for seed in range(n_runs):
        splitter = StratifiedSplitter(dataset)
        train_data, val_data, test_data = splitter.split(seed=seed)

        X_train = train_data[:, :-1]
        X_val = val_data[:, :-1]
        y_val = val_data[:, -1]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        try:
            model = model_class(random_state=seed)
        except TypeError:
            model = model_class()

        model.fit(X_train)
        scores = model.decision_function(X_val)

        precision, recall, _ = precision_recall_curve(y_val, scores)
        prauc = auc(recall, precision)
        prauc_scores.append(prauc)

        auroc = roc_auc_score(y_val, scores)
        auroc_scores.append(auroc)

        scores_normalized = (scores - scores.min()) / (
            scores.max() - scores.min() + 1e-10
        )
        brier = brier_score_loss(y_val, scores_normalized)
        brier_scores.append(brier)

        logger.info(
            f"{dataset.name} | {model_class.__name__} | "
            f"Seed {seed} | PRAUC: {prauc:.4f} | AUROC: {auroc:.4f} | Brier: {brier:.4f}"
        )

    return {
        "dataset": dataset.name,
        "model": model_class.__name__,
        "mean_prauc": np.mean(prauc_scores),
        "std_prauc": np.std(prauc_scores),
        "mean_auroc": np.mean(auroc_scores),
        "std_auroc": np.std(auroc_scores),
        "mean_brier": np.mean(brier_scores),
        "std_brier": np.std(brier_scores),
        "prauc_scores": prauc_scores,
        "auroc_scores": auroc_scores,
        "brier_scores": brier_scores,
        "n_runs": n_runs,
    }


def run_model_selection(
    datasets: List[Dataset],
    model_classes: List[Type],
    n_runs: int = 10,
    study_name: str = "model_selection",
) -> Dict[str, Dict[str, Any]]:
    """Run model selection across multiple datasets and models."""
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=None,
    )

    results = {}

    for dataset in datasets:
        results[dataset.name] = {}

        for model_class in tqdm(model_classes, desc=f"{dataset.name}"):
            logger.info(f"\nEvaluating {model_class.__name__} on {dataset.name}...")

            if (dataset.name, model_class.__name__) in SKIP_COMBINATIONS:
                logger.warning(
                    f"Skipping {model_class.__name__} on {dataset.name} (configured skip)"
                )
                result = _create_failure_result(
                    dataset, model_class, Exception("Skipped: too slow")
                )
                results[dataset.name][model_class.__name__] = result
                continue

            try:
                result = evaluate_model_on_dataset(
                    dataset=dataset, model_class=model_class, n_runs=n_runs
                )
            except Exception as e:
                logger.error(f"FAILED: {model_class.__name__} on {dataset.name}: {e}")
                result = _create_failure_result(dataset, model_class, e)

            results[dataset.name][model_class.__name__] = result

            if not result.get("failed", False):
                trial = study.ask()
                study.tell(trial, result["mean_prauc"])

                logger.info(
                    f"Mean PRAUC: {result['mean_prauc']:.4f} ± {result['std_prauc']:.4f} | "
                    f"AUROC: {result['mean_auroc']:.4f} ± {result['std_auroc']:.4f} | "
                    f"Brier: {result['mean_brier']:.4f} ± {result['std_brier']:.4f}"
                )

    return results


def get_best_models(results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Find the best model for each dataset based on mean PRAUC."""
    best_models = {}

    for dataset_name, models in results.items():
        valid_models = {k: v for k, v in models.items() if not v.get("failed", False)}
        if valid_models:
            best_model = max(valid_models.items(), key=lambda x: x[1]["mean_prauc"])
            best_models[dataset_name] = best_model[0]

    return best_models
