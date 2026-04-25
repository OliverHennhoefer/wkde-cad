import unittest

import numpy as np
from nonconform.weighting import BootstrapBaggedWeightEstimator, SklearnWeightEstimator
from nonconform.weighting import forest_weight_estimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils.weight_estimators import build_weight_estimator


class WeightEstimatorFactoryTest(unittest.TestCase):
    def test_forest_estimator_matches_previous_factory_path(self):
        estimator = build_weight_estimator("forest", n_bootstraps=3)
        previous = forest_weight_estimator()

        self.assertIsInstance(estimator, SklearnWeightEstimator)
        self.assertEqual(
            estimator.base_estimator.get_params(),
            previous.base_estimator.get_params(),
        )
        self.assertEqual(estimator.clip_quantile, previous.clip_quantile)

    def test_non_logistic_estimator_weights_match_previous_paths(self):
        rng = np.random.default_rng(0)
        calibration = rng.normal(size=(30, 3))
        test = rng.normal(loc=0.3, size=(20, 3))

        previous_forest = forest_weight_estimator()
        current_forest = build_weight_estimator("forest", n_bootstraps=3)
        previous_forest.set_seed(123)
        current_forest.set_seed(123)
        previous_forest.fit(calibration, test)
        current_forest.fit(calibration, test)

        for previous_weights, current_weights in zip(
            previous_forest.get_weights(),
            current_forest.get_weights(),
            strict=True,
        ):
            self.assertTrue(np.allclose(previous_weights, current_weights))

        previous_bagged = BootstrapBaggedWeightEstimator(
            base_estimator=forest_weight_estimator(),
            n_bootstraps=3,
        )
        current_bagged = build_weight_estimator("forest_bagged", n_bootstraps=3)
        previous_bagged.set_seed(123)
        current_bagged.set_seed(123)
        previous_bagged.fit(calibration, test)
        current_bagged.fit(calibration, test)

        for previous_weights, current_weights in zip(
            previous_bagged.get_weights(),
            current_bagged.get_weights(),
            strict=True,
        ):
            self.assertTrue(np.allclose(previous_weights, current_weights))

    def test_builds_logistic_weight_estimator(self):
        estimator = build_weight_estimator("logistic", n_bootstraps=3)

        self.assertIsInstance(estimator, SklearnWeightEstimator)
        self.assertIsInstance(estimator.base_estimator, Pipeline)
        self.assertIsInstance(
            estimator.base_estimator.named_steps["logisticregression"],
            LogisticRegression,
        )

    def test_accepts_logistic_regression_alias(self):
        estimator = build_weight_estimator("logistic_regression", n_bootstraps=3)

        self.assertIsInstance(
            estimator.base_estimator.named_steps["logisticregression"],
            LogisticRegression,
        )

    def test_builds_bagged_forest_weight_estimator(self):
        estimator = build_weight_estimator("forest_bagged", n_bootstraps=3)

        self.assertIsInstance(estimator, BootstrapBaggedWeightEstimator)
        self.assertEqual(estimator.n_bootstraps, 3)

    def test_rejects_unknown_weight_estimator(self):
        with self.assertRaisesRegex(ValueError, "Valid options"):
            build_weight_estimator("unknown", n_bootstraps=3)


if __name__ == "__main__":
    unittest.main()
