import unittest

import numpy as np
import pandas as pd

from src.rebuttal.covariate_shift import (
    FixedWeightEstimator,
    fit_propensity_model,
    rejection_sample,
)


def _features() -> pd.DataFrame:
    x = np.linspace(-2.0, 2.0, 80)
    return pd.DataFrame(
        {
            "x1": x,
            "x2": x**2,
            "x3": np.sin(x),
        }
    )


class CovariateShiftTest(unittest.TestCase):
    def test_zero_severity_gives_constant_target_propensity(self):
        features = _features()
        model = fit_propensity_model(
            features,
            train_split=0.5,
            severity=0.0,
            propensity_min=0.02,
            propensity_max=0.98,
        )

        propensities = model.propensity(features)

        self.assertTrue(np.allclose(propensities, 0.5))
        self.assertTrue(np.isclose(np.mean(propensities), 0.5))

    def test_higher_severity_increases_propensity_variation_but_preserves_mean(self):
        features = _features()
        flat = fit_propensity_model(
            features,
            train_split=0.5,
            severity=0.0,
            propensity_min=0.02,
            propensity_max=0.98,
        )
        shifted = fit_propensity_model(
            features,
            train_split=0.5,
            severity=2.0,
            propensity_min=0.02,
            propensity_max=0.98,
        )

        flat_propensities = flat.propensity(features)
        shifted_propensities = shifted.propensity(features)

        self.assertGreater(np.std(shifted_propensities), np.std(flat_propensities))
        self.assertTrue(np.isclose(np.mean(shifted_propensities), 0.5, atol=1e-8))

    def test_rejection_sample_partitions_rows_and_tracks_propensities(self):
        features = _features()
        data = features.assign(Class=0)
        model = fit_propensity_model(
            features,
            train_split=0.5,
            severity=1.0,
            propensity_min=0.02,
            propensity_max=0.98,
        )

        sample = rejection_sample(
            data,
            ["x1", "x2", "x3"],
            model,
            seed=123,
        )

        self.assertEqual(len(sample.accepted) + len(sample.rejected), len(data))
        self.assertTrue(sample.accepted.index.intersection(sample.rejected.index).empty)
        self.assertEqual(len(sample.all_propensity), len(data))

    def test_fixed_weight_estimator_returns_matching_copies_and_rejects_mismatch(self):
        estimator = FixedWeightEstimator(
            calibration_weights=np.array([1.0, 2.0, 3.0]),
            test_weights=np.array([0.5, 1.5]),
        )

        estimator.fit(np.zeros((3, 2)), np.zeros((2, 2)))
        calibration_weights, test_weights = estimator.get_weights()

        self.assertTrue(np.array_equal(calibration_weights, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.array_equal(test_weights, np.array([0.5, 1.5])))

        calibration_weights[0] = 99.0
        copied_calibration_weights, _ = estimator.get_weights()
        self.assertEqual(copied_calibration_weights[0], 1.0)

        with self.assertRaises(ValueError):
            estimator.get_weights(np.zeros((2, 2)), np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
