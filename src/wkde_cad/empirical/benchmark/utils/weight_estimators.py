from __future__ import annotations

from nonconform.weighting import (
    BootstrapBaggedWeightEstimator,
    forest_weight_estimator,
)


VALID_WEIGHT_ESTIMATORS = (
    "forest",
    "forest_bagged",
    "logistic",
    "logistic_regression",
)


def build_weight_estimator(weight_choice: str, n_bootstraps: int):
    normalized_choice = str(weight_choice).strip().lower()
    if normalized_choice == "forest":
        return forest_weight_estimator()
    if normalized_choice == "forest_bagged":
        return BootstrapBaggedWeightEstimator(
            base_estimator=forest_weight_estimator(),
            n_bootstraps=n_bootstraps,
        )
    if normalized_choice in {"logistic", "logistic_regression"}:
        from nonconform.weighting import logistic_weight_estimator

        return logistic_weight_estimator()

    valid_options = "', '".join(VALID_WEIGHT_ESTIMATORS)
    raise ValueError(
        f"Invalid weight_estimator '{normalized_choice}'. "
        f"Valid options are: '{valid_options}'."
    )
