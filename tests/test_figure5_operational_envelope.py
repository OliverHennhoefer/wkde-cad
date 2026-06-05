import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.scripts import figure5_operational_envelope as figure5


class Figure5OperationalEnvelopeTest(unittest.TestCase):
    def test_default_output_dir_is_under_outputs(self):
        self.assertEqual(figure5.OUT_DIR.name, "figure5")
        self.assertEqual(figure5.OUT_DIR.parent.name, "outputs")

    def test_default_neff_bins_match_figure6_range(self):
        self.assertEqual(len(figure5.N_EFF_BINS), 15)
        self.assertAlmostEqual(float(figure5.N_EFF_BINS[0]), 1.3)
        self.assertAlmostEqual(float(figure5.N_EFF_BINS[-1]), 4.1)

    def test_top_neff_bin_has_uncapped_uniform_weight_candidate(self):
        candidates = figure5.neff_config_candidates(4.0)

        self.assertTrue(
            any(
                rho == 0.0
                and figure5.N_EFF_BINS[-2] <= np.log10(n_cal) < figure5.N_EFF_BINS[-1]
                for n_cal, rho in candidates
            )
        )

    def test_weighted_tail_p_values_match_self_atom_formula(self):
        sorted_calib_scores = np.array([0.0, 1.0, 2.0])
        sorted_calib_weights = np.array([1.0, 2.0, 3.0])
        suffix_calib_weights = np.concatenate(
            ([0.0], np.cumsum(sorted_calib_weights[::-1]))
        )[::-1]

        p_values = figure5.weighted_tail_p_values(
            sorted_calib_scores,
            suffix_calib_weights,
            total_calib_weight=6.0,
            test_scores=np.array([1.0, 3.0]),
            test_weights=np.array([4.0, 5.0]),
        )

        np.testing.assert_allclose(p_values, np.array([0.9, 5.0 / 11.0]))

    def test_tiny_run_writes_operational_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empirical_dir = root / "empirical"
            path_patches = {
                "OUT_DIR": root,
                "POWER_ATLAS_FIGURE_PATH": root
                / "figure5_operational_power_atlas.png",
                "POWER_ATLAS_SUMMARY_PATH": root
                / "figure5_operational_power_atlas_summary.csv",
                "POWER_ATLAS_TIKZ_PATH": root
                / "figure5_operational_power_atlas_tikz.csv",
                "POWER_ATLAS_REFERENCE_TIKZ_PATH": root
                / "figure5_operational_power_atlas_reference_tikz.csv",
                "REQUIRED_ESS_FIGURE_PATH": root
                / "figure5_required_ess_design_map.png",
                "REQUIRED_ESS_SUMMARY_PATH": root / "figure5_required_ess_summary.csv",
                "REQUIRED_ESS_TIKZ_PATH": root / "figure5_required_ess_tikz.csv",
                "EMPIRICAL_PROJECTION_FIGURE_PATH": root
                / "figure5_empirical_projection.png",
                "EMPIRICAL_PROJECTION_SUMMARY_PATH": root
                / "figure5_empirical_projection_summary.csv",
                "EMPIRICAL_PROJECTION_TIKZ_PATH": root
                / "figure5_empirical_projection_tikz.csv",
                "EMPIRICAL_RESULTS_DIR": empirical_dir,
            }
            value_patches = {
                "N_EFF_BINS": np.array([0.0, 1.0, 2.0]),
                "RHO_CANDIDATES": [0.0],
                "N_CAL_MAX": 100,
                "MAX_ACCEPT_ATTEMPTS_PER_CELL": 200,
                "M_VALUES": [8, 10],
                "PI1_VALUES": [0.2, 0.3],
                "ALPHA_VALUES": [0.1, 0.2],
                "REQUIRED_M_VALUES": [8, 10],
                "REQUIRED_PI1_VALUES": [0.2, 0.3],
                "REQUIRED_ALPHA_VALUES": [0.1],
                "BASELINE_M": 10,
                "BASELINE_N_ANOMALY": 2,
                "tqdm": lambda iterable, **kwargs: iterable,
            }

            with mock.patch.multiple(figure5, **path_patches), mock.patch.multiple(
                figure5,
                **value_patches,
            ):
                figure5.main(
                    [
                        "--workers",
                        "1",
                        "--cell-trials",
                        "1",
                        "--wcs-batch-size",
                        "8",
                        "--force",
                    ]
                )

            for path in path_patches.values():
                if Path(path).suffix:
                    self.assertTrue(Path(path).exists(), path)

            atlas = pd.read_csv(path_patches["POWER_ATLAS_SUMMARY_PATH"])
            expected_atlas_rows = (
                len(figure5.SCORE_REGIMES)
                * (2 + 2 + 2 + 2)
                * (len(value_patches["N_EFF_BINS"]) - 1)
            )
            self.assertEqual(len(atlas), expected_atlas_rows)
            self.assertTrue((atlas["count"] > 0).all())
            self.assertEqual(set(atlas["collection"]), {"atlas"})
            self.assertEqual(set(atlas["score_regime"]), set(figure5.SCORE_REGIMES))
            self.assertTrue(
                atlas[atlas["score_regime"].eq("perfect")][
                    "perfect_separation_rate"
                ].eq(1.0).all()
            )
            for column in [
                "discovery_probability",
                "power",
                "fdr",
                "certified_no_rank_rate",
                "mean_auroc",
            ]:
                self.assertTrue(((atlas[column] >= 0.0) & (atlas[column] <= 1.0)).all())
            self.assertTrue(np.isfinite(atlas["median_log10_rank_delta"]).all())

            required = pd.read_csv(path_patches["REQUIRED_ESS_SUMMARY_PATH"])
            expected_required_rows = (
                len(figure5.SCORE_REGIMES)
                * len(value_patches["REQUIRED_ALPHA_VALUES"])
                * len(value_patches["REQUIRED_M_VALUES"])
                * len(value_patches["REQUIRED_PI1_VALUES"])
            )
            self.assertEqual(len(required), expected_required_rows)
            self.assertTrue(required["status"].isin(["attained", "out_of_range"]).all())
            attained = required[required["status"].eq("attained")]
            self.assertTrue(np.isfinite(attained["required_log10_neff"]).all())
            out_of_range = required[required["status"].eq("out_of_range")]
            self.assertTrue(out_of_range["required_log10_neff"].isna().all())

            empirical = pd.read_csv(path_patches["EMPIRICAL_PROJECTION_SUMMARY_PATH"])
            self.assertEqual(empirical.columns.tolist(), figure5.EMPIRICAL_COLUMNS)

            atlas_tikz = pd.read_csv(path_patches["POWER_ATLAS_TIKZ_PATH"])
            self.assertEqual(len(atlas_tikz), len(atlas))
            self.assertEqual(set(atlas_tikz["export_version"]), {"tikz-v1"})
            self.assertTrue(
                {
                    "panel",
                    "group",
                    "x",
                    "y",
                    "x_left",
                    "x_right",
                    "y_bottom",
                    "y_top",
                    "value",
                    "power",
                    "median_log10_rank_delta",
                }.issubset(atlas_tikz.columns)
            )
            self.assertEqual(set(atlas_tikz["group"]), {"heatmap_cell"})

            atlas_reference = pd.read_csv(
                path_patches["POWER_ATLAS_REFERENCE_TIKZ_PATH"]
            )
            self.assertEqual(set(atlas_reference["export_version"]), {"tikz-v1"})
            self.assertEqual(set(atlas_reference["group"]), {"contour_reference"})
            self.assertEqual(
                set(atlas_reference["style_key"]),
                {"median_log10_rank_delta_zero"},
            )

            required_tikz = pd.read_csv(path_patches["REQUIRED_ESS_TIKZ_PATH"])
            self.assertEqual(len(required_tikz), len(required))
            self.assertEqual(set(required_tikz["export_version"]), {"tikz-v1"})
            self.assertTrue(
                {
                    "panel",
                    "group",
                    "x",
                    "y",
                    "x_left",
                    "x_right",
                    "y_bottom",
                    "y_top",
                    "value",
                    "required_log10_neff",
                    "status",
                }.issubset(required_tikz.columns)
            )

            empirical_tikz = pd.read_csv(
                path_patches["EMPIRICAL_PROJECTION_TIKZ_PATH"]
            )
            self.assertTrue(
                {
                    "export_version",
                    "figure",
                    "panel",
                    "group",
                    "x",
                    "y",
                    "value",
                    "style_key",
                }.issubset(empirical_tikz.columns)
            )


if __name__ == "__main__":
    unittest.main()
