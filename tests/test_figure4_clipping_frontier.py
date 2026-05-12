import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy.stats import norm

from NEW.Figure4 import figure4_clipping_frontier as figure4


class Figure4ClippingFrontierTest(unittest.TestCase):
    def test_unclipped_analytic_tail_equals_shifted_null_tail(self):
        rho = 1.2
        scores = np.array([-1.0, 0.0, 1.0, 2.5, 4.0])

        observed = figure4.clipped_target_tail(scores, cap=np.inf, rho=rho)

        np.testing.assert_allclose(observed, norm.sf(scores - rho))

    def test_tighter_clipping_increases_oracle_tail_mismatch(self):
        tail_probs = np.geomspace(1e-4, 0.5, 30)

        unclipped = figure4.oracle_tail_metrics(
            cap=np.inf,
            tail_probs=tail_probs,
            rho=1.5,
        )
        clipped = figure4.oracle_tail_metrics(
            cap=1.0,
            tail_probs=tail_probs,
            rho=1.5,
        )

        self.assertLess(
            abs(unclipped["oracle_tail_mismatch_mean_abs_log10"]),
            1e-12,
        )
        self.assertGreater(
            clipped["oracle_tail_mismatch_mean_abs_log10"],
            unclipped["oracle_tail_mismatch_mean_abs_log10"] + 0.05,
        )

    def test_tighter_clipping_lowers_self_atom_and_raises_ess(self):
        calib_z = np.linspace(-2.0, 3.0, 40)
        test_z = np.linspace(-1.0, 4.0, 30)

        tight = figure4.resolution_metrics(calib_z, test_z, cap=1.0, rho=1.5)
        loose = figure4.resolution_metrics(calib_z, test_z, cap=np.inf, rho=1.5)

        self.assertLess(tight["max_test_self_atom"], loose["max_test_self_atom"])
        self.assertGreater(tight["calib_ess"], loose["calib_ess"])
        self.assertGreater(tight["max_test_self_atom_to_bh_threshold"], 0.0)
        self.assertGreater(tight["log10_inverse_max_test_self_atom"], 0.0)

    def test_unclipped_tv_is_zero_and_tight_clipping_has_larger_tv(self):
        self.assertAlmostEqual(
            figure4.clipped_target_tv(cap=np.inf, rho=1.5),
            0.0,
        )

        tight = figure4.clipped_target_tv(cap=1.0, rho=1.5)
        loose = figure4.clipped_target_tv(cap=89.0, rho=1.5)

        self.assertGreater(tight, loose)
        self.assertGreater(loose, 0.0)

    def test_tiny_run_writes_outputs_and_valid_multi_rho_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path_patches = {
                "OUT_DIR": root,
                "FIGURE_PATH": root / "figure4_clipping_frontier.png",
                "SUMMARY_PATH": root / "figure4_clipping_frontier_summary.csv",
                "TIKZ_PATH": root / "figure4_clipping_frontier_tikz.csv",
                "TABLE_PATH": root / "figure4_key_settings_table.csv",
                "RATIONALE_PATH": root / "RATIONALE.md",
            }
            value_patches = {
                "RHO_VALUES": [0.5, 1.5],
                "N_CAL": 40,
                "M": 60,
                "N_SEEDS": 4,
                "CLIP_CAPS": [1.0, 2.0, np.inf],
                "TAIL_PROBS": np.array([0.5, 0.1, 0.01]),
            }

            with mock.patch.multiple(figure4, **path_patches), mock.patch.multiple(
                figure4,
                **value_patches,
            ):
                figure4.main()

            for path in path_patches.values():
                self.assertTrue(Path(path).exists(), path)

            summary = pd.read_csv(path_patches["SUMMARY_PATH"])
            self.assertEqual(len(summary), 2 * 3)
            self.assertEqual(set(summary["summary_version"]), {figure4.SUMMARY_VERSION})
            self.assertEqual(set(summary["rho"]), {0.5, 1.5})
            for rho in [0.5, 1.5]:
                labels = list(
                    summary[summary["rho"].eq(rho)]
                    .sort_values("clip_order")["clip_cap_label"]
                )
                self.assertEqual(labels, ["c=1", "c=2", "unclipped"])
            self.assertFalse(summary.duplicated(["rho", "clip_cap_label"]).any())
            self.assertTrue((summary["max_test_self_atom_mean"] > 0.0).all())
            self.assertTrue((summary["calib_ess_fraction_mean"] <= 1.0).all())
            self.assertTrue((summary["bh_first_threshold"] > 0.0).all())
            self.assertTrue(
                (summary["max_test_self_atom_to_bh_threshold_mean"] > 0.0).all()
            )

            for rho in [0.5, 1.5]:
                block = summary[summary["rho"].eq(rho)]
                clipped = block[block["clip_cap_label"].eq("c=1")].iloc[0]
                unclipped = block[block["clip_cap_label"].eq("unclipped")].iloc[0]
                self.assertLess(
                    clipped["max_test_self_atom_mean"],
                    unclipped["max_test_self_atom_mean"],
                )
                self.assertGreater(
                    clipped["calib_ess_fraction_mean"],
                    unclipped["calib_ess_fraction_mean"],
                )
                self.assertGreater(
                    clipped["oracle_tail_mismatch_mean_abs_log10"],
                    unclipped["oracle_tail_mismatch_mean_abs_log10"],
                )
                self.assertGreater(clipped["clipped_target_tv"], 0.0)
                self.assertAlmostEqual(unclipped["clipped_target_tv"], 0.0)

            table = pd.read_csv(path_patches["TABLE_PATH"])
            self.assertEqual(len(table), 2 * 3)
            self.assertEqual(set(table["setting"]), {
                "tightest_clip",
                "middle_cap",
                "unclipped_or_largest_cap",
            })

            tikz = pd.read_csv(path_patches["TIKZ_PATH"])
            self.assertTrue(
                {
                    "max_test_self_atom",
                    "calib_ess_fraction",
                    "oracle_tail_mismatch",
                    "clipped_target_tv",
                    "frontier_oracle_tail_mismatch",
                }.issubset(set(tikz["metric"]))
            )
            self.assertEqual(set(tikz["panel"]), {"A", "B", "C"})


if __name__ == "__main__":
    unittest.main()
