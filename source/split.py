import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np

from nonconform.utils.data import load, Dataset


class StratifiedSplitter:
    """
    Stratified splitter for anomaly detection datasets.

    Splits data into train/validation/test sets while preserving anomaly rates.
    Training set contains only normal samples, while validation and test sets
    maintain empirical anomaly rate π̂ as closely as possible.
    """

    def __init__(
        self,
        dataset: Dataset,
        min_anomalies_floor: int = 5,
        train_normal_frac: float = 0.60,
        val_normal_frac: Optional[float] = 0.20,
    ):
        """
        Initialize the stratified splitter.

        Parameters
        ----------
        dataset : Dataset
            Dataset to load from nonconform package.
        min_anomalies_floor : int, default=5
            Minimum number of anomalies per validation/test split.
        train_normal_frac : float, default=0.70
            Fraction of normal samples for training (clamped to 0.60-0.75).
        val_normal_frac : float or None, default=0.20
            Fraction of normal samples for validation. If ``None`` or ``0.0``,
            the splitter returns only train and test splits (70/30 by default)
            while keeping the interface backward compatible.
        """
        self.min_anomalies_floor = min_anomalies_floor
        self.train_normal_frac = np.clip(train_normal_frac, 0.60, 0.75)
        self._use_validation = val_normal_frac is not None and val_normal_frac > 0.0
        self.val_normal_frac = float(val_normal_frac or 0.0)

        # Load data with fixed seed for reproducibility
        x_train, x_test, y_test = load(dataset, setup=True, seed=42)

        # Combine into full dataset (training data has no anomalies)
        y_train = np.zeros(len(x_train), dtype=y_test.dtype)
        self.X = np.concatenate([x_train, x_test], axis=0)
        self.y = np.concatenate([y_train, y_test], axis=0)
        self.logger = logging.getLogger(__name__)
        self._split_info = {}

    def split(
        self, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Parameters
        ----------
        seed : int, optional
            Seed for reproducibility.

        Returns
        -------
        train_data : np.ndarray
            Training samples with features and label column (all zeros).
        val_data : np.ndarray
            Validation samples with mixed normal/anomalous labels.
        test_data : np.ndarray
            Test samples with mixed normal/anomalous labels.
        """
        rng = np.random.default_rng(seed)

        # Separate normal and anomaly indices
        normal_idx = np.where(self.y == 0)[0]
        anomaly_idx = np.where(self.y == 1)[0]

        # Shuffle indices
        normal_idx = rng.permutation(normal_idx)
        anomaly_idx = rng.permutation(anomaly_idx)

        n_normals = len(normal_idx)
        n_anomalies = len(anomaly_idx)
        n_total = n_normals + n_anomalies

        # Calculate empirical anomaly rate
        empirical_rate = n_anomalies / n_total

        # Split normals
        n_train_normals = int(self.train_normal_frac * n_normals)
        n_val_normals = (
            int(self.val_normal_frac * n_normals) if self._use_validation else 0
        )
        n_test_normals = n_normals - n_train_normals - n_val_normals

        if n_test_normals < 0:
            raise ValueError(
                "Invalid split configuration: test normals count became negative. "
                "Check train_normal_frac and val_normal_frac settings."
            )
        train_normal_idx = normal_idx[:n_train_normals]
        if self._use_validation:
            val_normal_idx = normal_idx[
                n_train_normals : n_train_normals + n_val_normals
            ]
            test_normal_idx = normal_idx[n_train_normals + n_val_normals :]
        else:
            val_normal_idx = np.array([], dtype=int)
            test_normal_idx = normal_idx[n_train_normals:]

        if self._use_validation:
            # Calculate anomaly counts to match empirical rate
            n_val_anomalies = self._compute_anomaly_count(
                n_val_normals, empirical_rate, n_anomalies
            )
            n_test_anomalies = self._compute_anomaly_count(
                n_test_normals, empirical_rate, n_anomalies - n_val_anomalies
            )

            # Ensure minimum anomaly floor and don't exceed available anomalies
            n_val_anomalies = max(
                self.min_anomalies_floor, min(n_val_anomalies, n_anomalies // 2)
            )
            n_test_anomalies = min(
                n_anomalies - n_val_anomalies,
                max(self.min_anomalies_floor, n_test_anomalies),
            )

            # Split anomalies
            val_anomaly_idx = anomaly_idx[:n_val_anomalies]
            test_anomaly_idx = anomaly_idx[
                n_val_anomalies : n_val_anomalies + n_test_anomalies
            ]
        else:
            n_val_anomalies = 0
            n_test_anomalies = n_anomalies
            val_anomaly_idx = np.array([], dtype=int)
            test_anomaly_idx = anomaly_idx

        # Combine indices
        train_idx = train_normal_idx
        val_idx = np.concatenate([val_normal_idx, val_anomaly_idx])
        test_idx = np.concatenate([test_normal_idx, test_anomaly_idx])

        # Shuffle validation and test sets
        val_idx = rng.permutation(val_idx)
        test_idx = rng.permutation(test_idx)

        train_X, train_y = self.X[train_idx], self.y[train_idx]
        val_X, val_y = self.X[val_idx], self.y[val_idx]
        test_X, test_y = self.X[test_idx], self.y[test_idx]

        train_X = self._ensure_2d(train_X)
        val_X = self._ensure_2d(val_X)
        test_X = self._ensure_2d(test_X)

        train_data = np.concatenate([train_X, train_y[:, None]], axis=1)
        val_data = np.concatenate([val_X, val_y[:, None]], axis=1)
        test_data = np.concatenate([test_X, test_y[:, None]], axis=1)

        # Ensure training split contains only normal samples
        if not np.all(train_data[:, -1] == 0):
            raise ValueError(
                "Training split contains anomalous samples; expected only label 0."
            )

        # Calculate observed rates
        train_rate = 0.0
        val_rate = n_val_anomalies / len(val_idx) if len(val_idx) > 0 else 0.0
        test_rate = n_test_anomalies / len(test_idx) if len(test_idx) > 0 else 0.0

        # Log split composition
        self.logger.info(
            f"Train: {len(train_normal_idx)} normals, 0 anomalies ({train_rate:.1%})"
        )
        if self._use_validation:
            self.logger.info(
                f"Val: {n_val_normals} normals, {n_val_anomalies} anomalies ({val_rate:.1%})"
            )
        self.logger.info(
            f"Test: {n_test_normals} normals, {n_test_anomalies} anomalies ({test_rate:.1%})"
        )

        # Store split info
        split_info = {
            "empirical_rate": round(empirical_rate, ndigits=4),
            "train": {
                "normals": len(train_normal_idx),
                "anomalies": 0,
                "total": len(train_idx),
                "rate": train_rate,
            },
            "test": {
                "normals": n_test_normals,
                "anomalies": n_test_anomalies,
                "total": len(test_idx),
                "rate": round(test_rate, ndigits=4),
            },
            "seed": seed,
        }
        if self._use_validation:
            split_info["val"] = {
                "normals": n_val_normals,
                "anomalies": n_val_anomalies,
                "total": len(val_idx),
                "rate": round(val_rate, ndigits=4),
            }
        else:
            split_info["val"] = None
        self._split_info = split_info

        return train_data, val_data, test_data

    def _compute_anomaly_count(
        self, n_normals: int, target_rate: float, max_anomalies: int
    ) -> int:
        """
        Compute number of anomalies needed to achieve target rate.

        target_rate = n_anomalies / (n_normals + n_anomalies)
        n_anomalies = (target_rate * n_normals) / (1 - target_rate)
        """
        if target_rate >= 1.0:
            return max_anomalies

        n_anomalies = int((target_rate * n_normals) / (1 - target_rate))
        return min(n_anomalies, max_anomalies)

    def get_split_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the last split.

        Returns
        -------
        dict
            Dictionary containing split counts and achieved rates.
        """
        return self._split_info.copy()

    @staticmethod
    def _ensure_2d(array: np.ndarray) -> np.ndarray:
        if array.ndim == 1:
            return array.reshape(-1, 1)
        return array
