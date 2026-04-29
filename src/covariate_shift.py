from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nonconform.weighting import BaseWeightEstimator


EPSILON = 1e-12


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    values_arr = np.asarray(values)
    positive = values_arr >= 0
    negative = ~positive
    result = np.empty_like(values_arr, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values_arr[positive]))
    exp_values = np.exp(values_arr[negative])
    result[negative] = exp_values / (1.0 + exp_values)
    if np.isscalar(values):
        return float(result)
    return result


def _as_2d_float_array(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = (
        features.to_numpy(dtype=float)
        if isinstance(features, pd.DataFrame)
        else np.asarray(features, dtype=float)
    )
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {values.shape}.")
    if values.shape[0] == 0:
        raise ValueError("Cannot fit a propensity model with zero rows.")
    return values


def _first_principal_component(standardized: np.ndarray) -> np.ndarray:
    if standardized.shape[1] == 0:
        raise ValueError("Cannot compute PCA score with zero feature columns.")
    if np.allclose(standardized, 0.0):
        component = np.zeros(standardized.shape[1], dtype=float)
        component[0] = 1.0
        return component

    _, _, vh = np.linalg.svd(standardized, full_matrices=False)
    component = vh[0].astype(float, copy=True)
    largest_loading = int(np.argmax(np.abs(component)))
    if component[largest_loading] < 0:
        component *= -1
    return component


def solve_logistic_intercept(
    score: np.ndarray,
    target_probability: float,
    severity: float,
    propensity_min: float,
    propensity_max: float,
) -> float:
    """Solve intercept so clipped sigmoid probabilities match the target mean."""
    score = np.asarray(score, dtype=float)
    if not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must be in (0, 1).")
    if not 0.0 < propensity_min < propensity_max < 1.0:
        raise ValueError("Require 0 < propensity_min < propensity_max < 1.")
    if not propensity_min <= target_probability <= propensity_max:
        raise ValueError(
            "target_probability must lie within clipping bounds so the mean "
            "propensity is attainable."
        )

    lower = -50.0
    upper = 50.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        probabilities = np.clip(
            sigmoid(midpoint + severity * score),
            propensity_min,
            propensity_max,
        )
        if float(np.mean(probabilities)) < target_probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


@dataclass(frozen=True)
class PropensityModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    component: np.ndarray
    score_mean: float
    score_scale: float
    intercept: float
    severity: float
    propensity_min: float
    propensity_max: float
    target_probability: float

    def score(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = _as_2d_float_array(features)
        if values.shape[1] != self.component.shape[0]:
            raise ValueError(
                f"Expected {self.component.shape[0]} feature columns, got {values.shape[1]}."
            )
        standardized = (values - self.feature_mean) / self.feature_scale
        raw_score = standardized @ self.component
        return (raw_score - self.score_mean) / self.score_scale

    def propensity(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        scores = self.score(features)
        return np.clip(
            sigmoid(self.intercept + self.severity * scores),
            self.propensity_min,
            self.propensity_max,
        )

    def density_ratio(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        probabilities = self.propensity(features)
        return probabilities / np.maximum(1.0 - probabilities, EPSILON)


def fit_propensity_model(
    features: pd.DataFrame | np.ndarray,
    *,
    train_split: float,
    severity: float,
    propensity_min: float,
    propensity_max: float,
) -> PropensityModel:
    """Fit a PCA-1 logistic test-assignment model on reference inliers."""
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be in (0, 1).")
    if severity < 0:
        raise ValueError("severity must be non-negative.")

    values = _as_2d_float_array(features)
    feature_mean = values.mean(axis=0)
    feature_scale = values.std(axis=0)
    feature_scale = np.where(feature_scale < EPSILON, 1.0, feature_scale)
    standardized = (values - feature_mean) / feature_scale

    component = _first_principal_component(standardized)
    raw_score = standardized @ component
    score_mean = float(raw_score.mean())
    score_scale = float(raw_score.std())
    if score_scale < EPSILON:
        score_scale = 1.0
    score = (raw_score - score_mean) / score_scale

    target_probability = 1.0 - train_split
    intercept = solve_logistic_intercept(
        score,
        target_probability,
        severity,
        propensity_min,
        propensity_max,
    )
    return PropensityModel(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        component=component,
        score_mean=score_mean,
        score_scale=score_scale,
        intercept=intercept,
        severity=severity,
        propensity_min=propensity_min,
        propensity_max=propensity_max,
        target_probability=target_probability,
    )


@dataclass(frozen=True)
class RejectionSample:
    accepted: pd.DataFrame
    rejected: pd.DataFrame
    accepted_propensity: pd.Series
    rejected_propensity: pd.Series
    all_propensity: pd.Series

    @property
    def accepted_fraction(self) -> float:
        return len(self.accepted) / len(self.all_propensity)

    @property
    def rejected_fraction(self) -> float:
        return len(self.rejected) / len(self.all_propensity)


def rejection_sample(
    data: pd.DataFrame,
    feature_columns: list[str],
    propensity_model: PropensityModel,
    *,
    seed: int,
    uniforms: pd.Series | np.ndarray | None = None,
) -> RejectionSample:
    """Assign rows to the test-accepted pool by Bernoulli rejection sampling."""
    if len(data) == 0:
        empty = pd.Series(dtype=float, index=data.index)
        return RejectionSample(data.copy(), data.copy(), empty, empty, empty)

    probabilities = pd.Series(
        propensity_model.propensity(data[feature_columns]),
        index=data.index,
        name="test_propensity",
    )

    if uniforms is None:
        uniform_values = np.random.default_rng(seed).random(len(data))
    elif isinstance(uniforms, pd.Series):
        uniform_values = uniforms.loc[data.index].to_numpy(dtype=float)
    else:
        uniform_values = np.asarray(uniforms, dtype=float)

    if uniform_values.shape != (len(data),):
        raise ValueError(
            "uniforms must be one-dimensional and match the number of data rows."
        )
    if np.any((uniform_values < 0.0) | (uniform_values >= 1.0)):
        raise ValueError("uniforms must lie in [0, 1).")

    accepted_mask = pd.Series(
        uniform_values < probabilities.to_numpy(),
        index=data.index,
    )
    accepted = data.loc[accepted_mask].copy()
    rejected = data.loc[~accepted_mask].copy()
    return RejectionSample(
        accepted=accepted,
        rejected=rejected,
        accepted_propensity=probabilities.loc[accepted.index],
        rejected_propensity=probabilities.loc[rejected.index],
        all_propensity=probabilities,
    )


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0:
        return 0.0
    denominator = float(np.sum(weights**2))
    if denominator <= EPSILON:
        return 0.0
    return float(np.sum(weights) ** 2 / denominator)


def weight_summary(prefix: str, weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0:
        return {
            f"{prefix}_weight_mean": np.nan,
            f"{prefix}_weight_std": np.nan,
            f"{prefix}_weight_min": np.nan,
            f"{prefix}_weight_max": np.nan,
            f"{prefix}_weight_ess": 0.0,
        }
    return {
        f"{prefix}_weight_mean": float(np.mean(weights)),
        f"{prefix}_weight_std": float(np.std(weights)),
        f"{prefix}_weight_min": float(np.min(weights)),
        f"{prefix}_weight_max": float(np.max(weights)),
        f"{prefix}_weight_ess": effective_sample_size(weights),
    }


def _array_signature(array: np.ndarray) -> tuple[tuple[int, ...], str, str]:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()
    return contiguous.shape, str(contiguous.dtype), digest


class FixedWeightEstimator(BaseWeightEstimator):
    """Weight estimator that returns precomputed oracle density-ratio weights."""

    def __init__(
        self, calibration_weights: np.ndarray, test_weights: np.ndarray
    ) -> None:
        self.calibration_weights = np.asarray(calibration_weights, dtype=float)
        self.test_weights = np.asarray(test_weights, dtype=float)
        self._calibration_signature: tuple[tuple[int, ...], str, str] | None = None
        self._test_signature: tuple[tuple[int, ...], str, str] | None = None
        if self.calibration_weights.ndim != 1 or self.test_weights.ndim != 1:
            raise ValueError("Fixed weights must be one-dimensional arrays.")
        if np.any(self.calibration_weights <= 0) or np.any(self.test_weights <= 0):
            raise ValueError("Fixed weights must be strictly positive.")
        self._is_fitted = False

    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        if len(calibration_samples) != len(self.calibration_weights):
            raise ValueError(
                "Calibration sample count does not match fixed calibration weights: "
                f"{len(calibration_samples)} != {len(self.calibration_weights)}."
            )
        if len(test_samples) != len(self.test_weights):
            raise ValueError(
                "Test sample count does not match fixed test weights: "
                f"{len(test_samples)} != {len(self.test_weights)}."
            )
        self._calibration_signature = _array_signature(calibration_samples)
        self._test_signature = _array_signature(test_samples)
        self._is_fitted = True

    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        return self.calibration_weights.copy(), self.test_weights.copy()

    def _score_new_data(
        self,
        calibration_samples: np.ndarray,
        test_samples: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(calibration_samples) != len(self.calibration_weights):
            raise ValueError(
                "Calibration sample count does not match fixed calibration weights."
            )
        if len(test_samples) != len(self.test_weights):
            raise ValueError("Test sample count does not match fixed test weights.")
        if (
            _array_signature(calibration_samples) != self._calibration_signature
            or _array_signature(test_samples) != self._test_signature
        ):
            raise ValueError(
                "Fixed oracle weights cannot be rescored for different or reordered "
                "calibration/test arrays."
            )
        return self._get_stored_weights()
