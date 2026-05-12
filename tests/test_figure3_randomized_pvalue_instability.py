import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from NEW.Figure3 import figure3_randomized_pvalue_instability as figure3


class Figure3RandomizedPvalueInstabilityTest(unittest.TestCase):
    def test_randomized_pvalue_intervals_match_weighted_formula(self):
        sorted_calib_scores = np.array([0.0, 1.0, 2.0])
        sorted_calib_weights = np.array([1.0, 2.0, 3.0])
        suffix_calib_weights = np.concatenate(
            ([0.0], np.cumsum(sorted_calib_weights[::-1]))
        )[::-1]

        lower, upper = figure3.randomized_pvalue_intervals(
            sorted_calib_scores,
            suffix_calib_weights,
            total_calib_weight=6.0,
            test_scores=np.array([-1.0, 3.0]),
            test_weights=np.array([4.0, 5.0]),
        )

        np.testing.assert_allclose(lower, np.array([0.6, 0.0]))
        np.testing.assert_allclose(upper, np.array([1.0, 5.0 / 11.0]))

    def test_theorem_distribution_matches_two_anomaly_manual_case(self):
        distribution = figure3.theorem_distribution_from_interval_upper(
            np.array([1.0, 1.0]),
            alpha=0.5,
            m_total=2,
        )

        np.testing.assert_allclose(distribution, np.array([0.5, 0.25, 0.25]))

    def test_tiny_run_writes_outputs_and_matches_theorem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path_patches = {
                "OUT_DIR": root,
                "FIGURE_PATH": root / "figure3_randomized_pvalue_instability.png",
                "SUMMARY_PATH": root / "figure3_randomization_summary.csv",
                "DISTRIBUTION_PATH": root / "figure3_discovery_distribution.csv",
                "INTERVAL_TIKZ_PATH": root / "figure3_interval_tikz.csv",
                "DISTRIBUTION_TIKZ_PATH": root / "figure3_distribution_tikz.csv",
                "RATIONALE_PATH": root / "RATIONALE.md",
            }
            value_patches = {
                "N_CAL": 30,
                "M": 30,
                "N_ANOMALY": 3,
                "SHIFTED_RHO": 0.5,
                "N_RANDOMIZATIONS": 5000,
                "SIMULATION_BATCH_SIZE": 1000,
            }

            with mock.patch.multiple(figure3, **path_patches), mock.patch.multiple(
                figure3,
                **value_patches,
            ):
                figure3.main()

            for path in path_patches.values():
                self.assertTrue(Path(path).exists(), path)

            summary = pd.read_csv(path_patches["SUMMARY_PATH"])
            distribution = pd.read_csv(path_patches["DISTRIBUTION_PATH"])
            self.assertEqual(set(summary["scenario"]), {"equal_weights", "shifted_weights"})
            self.assertTrue(summary["inliers_nonrejectable"].all())
            self.assertTrue((summary["min_inlier_interval_lower"] > figure3.ALPHA).all())
            self.assertTrue((summary["miss_probability_error"].abs() < 0.08).all())

            for _, block in distribution.groupby("scenario"):
                self.assertAlmostEqual(float(block["theorem_probability"].sum()), 1.0)
                self.assertAlmostEqual(float(block["observed_probability"].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
