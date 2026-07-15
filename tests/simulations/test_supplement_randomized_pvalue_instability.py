import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from wkde_cad.simulations import (
    supplement_randomized_pvalue_instability as randomization,
)


class RandomizedPvalueInstabilityTest(unittest.TestCase):
    def test_default_output_leaf_matches_generator_stem(self):
        self.assertEqual(randomization.OUT_DIR.name, Path(randomization.__file__).stem)
        self.assertEqual(randomization.OUT_DIR.parent.name, "simulations")
        self.assertEqual(randomization.OUT_DIR.parent.parent.name, "outputs")

    def test_randomized_pvalue_intervals_match_weighted_formula(self):
        sorted_calib_scores = np.array([0.0, 1.0, 2.0])
        sorted_calib_weights = np.array([1.0, 2.0, 3.0])
        suffix_calib_weights = np.concatenate(
            ([0.0], np.cumsum(sorted_calib_weights[::-1]))
        )[::-1]

        lower, upper = randomization.randomized_pvalue_intervals(
            sorted_calib_scores,
            suffix_calib_weights,
            total_calib_weight=6.0,
            test_scores=np.array([-1.0, 3.0]),
            test_weights=np.array([4.0, 5.0]),
        )

        np.testing.assert_allclose(lower, np.array([0.6, 0.0]))
        np.testing.assert_allclose(upper, np.array([1.0, 5.0 / 11.0]))

    def test_theorem_distribution_matches_two_anomaly_manual_case(self):
        distribution = randomization.theorem_distribution_from_intervals(
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            alpha=0.5,
            m_total=2,
        )

        np.testing.assert_allclose(distribution, np.array([0.5, 0.25, 0.25]))

    def test_theorem_distribution_handles_nonzero_lower_interval(self):
        distribution = randomization.theorem_distribution_from_intervals(
            np.array([0.2]),
            np.array([0.6]),
            alpha=0.5,
            m_total=1,
        )

        np.testing.assert_allclose(distribution, np.array([0.25, 0.75]))

    def test_theorem_distribution_is_valid_probability_vector(self):
        distribution = randomization.theorem_distribution_from_intervals(
            np.array([0.0, 0.1, 0.2]),
            np.array([0.4, 0.5, 0.6]),
            alpha=0.3,
            m_total=3,
        )

        self.assertEqual(len(distribution), 4)
        self.assertTrue((distribution >= 0.0).all())
        self.assertAlmostEqual(float(distribution.sum()), 1.0)

    def test_tiny_frontier_run_writes_outputs_and_matches_theorem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path_patches = {
                "OUT_DIR": root,
                "PLOT_PATH": root / "randomized_pvalue_instability.png",
                "SUMMARY_PATH": root / "randomized_pvalue_instability_summary.csv",
                "DISTRIBUTION_PATH": root
                / "randomized_pvalue_instability_distribution.csv",
                "INTERVAL_TIKZ_PATH": root
                / "randomized_pvalue_instability_intervals_tikz.csv",
                "DISTRIBUTION_TIKZ_PATH": root
                / "randomized_pvalue_instability_distribution_tikz.csv",
            }
            value_patches = {
                "RHO_VALUES": np.array([0.0, 1.0]),
                "N_WORLD_SEEDS": 2,
                "N_CAL": 30,
                "M": 30,
                "N_ANOMALY": 3,
                "N_RANDOMIZATIONS": 5000,
                "SIMULATION_BATCH_SIZE": 1000,
                "FRONTIER_BINS": 3,
                "tqdm": lambda iterable, **kwargs: iterable,
            }

            with mock.patch.multiple(
                randomization, **path_patches
            ), mock.patch.multiple(
                randomization,
                **value_patches,
            ):
                randomization.main()
                self.assertTrue(randomization.summaries_are_current())

            self.assertEqual(
                {path.name for path in root.iterdir()},
                {
                    Path(path).name
                    for path in path_patches.values()
                    if Path(path).suffix
                },
            )
            for path in path_patches.values():
                if not Path(path).suffix:
                    continue
                self.assertTrue(Path(path).exists(), path)

            summary = pd.read_csv(path_patches["SUMMARY_PATH"])
            distribution = pd.read_csv(path_patches["DISTRIBUTION_PATH"])
            self.assertEqual(len(summary), 4)
            self.assertEqual(
                {
                    (float(row.rho), int(row.world_seed))
                    for row in summary.itertuples(index=False)
                },
                {(0.0, 0), (0.0, 1), (1.0, 0), (1.0, 1)},
            )
            self.assertTrue(summary["inliers_nonrejectable"].all())
            self.assertTrue(
                (summary["min_inlier_interval_lower"] > randomization.ALPHA).all()
            )
            self.assertTrue((summary["miss_probability_error"].abs() < 0.10).all())

            for _, block in distribution.groupby("world_id"):
                self.assertAlmostEqual(float(block["theorem_probability"].sum()), 1.0)
                self.assertAlmostEqual(float(block["observed_probability"].sum()), 1.0)

            interval_tikz = pd.read_csv(path_patches["INTERVAL_TIKZ_PATH"])
            frontier_tikz = pd.read_csv(path_patches["DISTRIBUTION_TIKZ_PATH"])
            self.assertEqual(
                set(interval_tikz["figure"]), {"randomized_pvalue_instability"}
            )
            self.assertEqual(
                set(frontier_tikz["figure"]), {"randomized_pvalue_instability"}
            )


if __name__ == "__main__":
    unittest.main()
