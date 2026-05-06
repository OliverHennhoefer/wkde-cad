from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ModelSelectionEvaluationSplit:
    normal_model_selection: pd.DataFrame
    normal_evaluation: pd.DataFrame
    anomaly_model_selection: pd.DataFrame
    anomaly_evaluation: pd.DataFrame


def split_model_selection_evaluation(
    normal: pd.DataFrame,
    anomaly: pd.DataFrame,
    *,
    train_split: float,
    seed: int,
) -> ModelSelectionEvaluationSplit:
    """Split inlier/outlier pools into model-selection and evaluation subsets."""
    normal_model_selection, normal_evaluation = train_test_split(
        normal,
        train_size=train_split,
        random_state=seed,
    )
    anomaly_model_selection, anomaly_evaluation = train_test_split(
        anomaly,
        train_size=train_split,
        random_state=seed,
    )
    return ModelSelectionEvaluationSplit(
        normal_model_selection=normal_model_selection.copy(),
        normal_evaluation=normal_evaluation.copy(),
        anomaly_model_selection=anomaly_model_selection.copy(),
        anomaly_evaluation=anomaly_evaluation.copy(),
    )
