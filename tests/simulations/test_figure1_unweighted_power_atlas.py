import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from wkde_cad.simulations import figure1_unweighted_power_atlas as unweighted


class Figure1UnweightedPowerAtlasTest(unittest.TestCase):
    def test_default_output_leaf_matches_generator_stem(self):
        self.assertEqual(unweighted.OUT_DIR.name, Path(unweighted.__file__).stem)
        self.assertEqual(unweighted.OUT_DIR.parent.name, "simulations")
        self.assertEqual(unweighted.OUT_DIR.parent.parent.name, "outputs")

    def test_default_bins_include_ten_thousand_calibration_center(self):
        centers = (unweighted.N_EFF_BINS[:-1] + unweighted.N_EFF_BINS[1:]) / 2.0

        self.assertTrue(np.isclose(centers[-1], 4.0))

    def test_ordinary_conformal_p_values_use_upper_tail_with_self_atom(self):
        p_values = unweighted.ordinary_conformal_p_values(
            calibration_scores=np.array([2.0, 0.0, 1.0]),
            test_scores=np.array([1.0, 3.0, -1.0]),
        )

        np.testing.assert_allclose(p_values, np.array([3.0 / 4.0, 1.0 / 4.0, 1.0]))

    def test_tiny_run_writes_unweighted_power_atlas_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path_patches = {
                "OUT_DIR": root,
                "POWER_ATLAS_FIGURE_PATH": root / "figure1_unweighted_power_atlas.png",
                "POWER_ATLAS_SUMMARY_PATH": root
                / "figure1_unweighted_power_atlas_summary.csv",
                "POWER_ATLAS_TIKZ_PATH": root
                / "figure1_unweighted_power_atlas_tikz.csv",
                "POWER_ATLAS_REFERENCE_TIKZ_PATH": root
                / "figure1_unweighted_power_atlas_reference_tikz.csv",
                "FDR_RATIO_ATLAS_FIGURE_PATH": root
                / "figure1_supp_fdr_ratio_atlas.png",
                "FDR_RATIO_ATLAS_TIKZ_PATH": root
                / "figure1_supp_fdr_ratio_atlas_tikz.csv",
                "FDR_RATIO_ATLAS_REFERENCE_TIKZ_PATH": root
                / "figure1_supp_fdr_ratio_atlas_reference_tikz.csv",
            }
            value_patches = {
                "N_EFF_BINS": np.array([0.0, 1.0, 2.0]),
                "M_VALUES": [8, 10],
                "PI1_VALUES": [0.2, 0.3],
                "ALPHA_VALUES": [0.1, 0.2],
                "BASELINE_M": 10,
                "BASELINE_N_ANOMALY": 2,
                "tqdm": lambda iterable, **kwargs: iterable,
            }

            with mock.patch.multiple(unweighted, **path_patches), mock.patch.multiple(
                unweighted,
                **value_patches,
            ):
                unweighted.main(
                    [
                        "--workers",
                        "1",
                        "--calibration-repeats",
                        "1",
                        "--test-repeats",
                        "1",
                        "--force",
                    ]
                )

            for path in path_patches.values():
                self.assertTrue(Path(path).exists(), path)

            atlas = pd.read_csv(path_patches["POWER_ATLAS_SUMMARY_PATH"])
            expected_atlas_rows = (
                len(unweighted.SCORE_REGIMES)
                * (2 + 2 + 2 + 2)
                * (len(value_patches["N_EFF_BINS"]) - 1)
            )
            self.assertEqual(len(atlas), expected_atlas_rows)
            self.assertTrue((atlas["count"] > 0).all())
            self.assertEqual(set(atlas["calibration_repeats"]), {1})
            self.assertEqual(set(atlas["test_repeats_per_calibration"]), {1})
            self.assertEqual(set(atlas["count"]), {1})
            self.assertEqual(set(atlas["collection"]), {"atlas"})
            self.assertEqual(set(atlas["score_regime"]), set(unweighted.SCORE_REGIMES))
            self.assertTrue(np.allclose(atlas["mean_neff"], atlas["mean_n_cal"]))
            self.assertTrue(np.allclose(atlas["mean_rho"], 0.0))
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

            atlas_tikz = pd.read_csv(path_patches["POWER_ATLAS_TIKZ_PATH"])
            self.assertEqual(len(atlas_tikz), len(atlas))
            self.assertEqual(set(atlas_tikz["export_version"]), {"tikz-v1"})
            self.assertEqual(set(atlas_tikz["figure"]), {"figure1"})
            self.assertEqual(set(atlas_tikz["method"]), {"standard_conformal_bh"})
            self.assertFalse(
                atlas_tikz["method"].str.contains("weighted|wcs", case=False).any()
            )
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
            self.assertEqual(set(atlas_reference["figure"]), {"figure1"})
            self.assertEqual(set(atlas_reference["group"]), {"contour_reference"})
            self.assertEqual(
                set(atlas_reference["style_key"]),
                {"median_log10_rank_delta_zero"},
            )

            fdr_tikz = pd.read_csv(path_patches["FDR_RATIO_ATLAS_TIKZ_PATH"])
            self.assertEqual(len(fdr_tikz), len(atlas))
            self.assertEqual(set(fdr_tikz["export_version"]), {"tikz-v1"})
            self.assertEqual(set(fdr_tikz["figure"]), {"figure1_supp"})
            self.assertEqual(set(fdr_tikz["method"]), {"standard_conformal_bh"})
            self.assertEqual(set(fdr_tikz["group"]), {"heatmap_cell"})
            np.testing.assert_allclose(
                fdr_tikz["fdr_ratio"].to_numpy(),
                fdr_tikz["fdr"].to_numpy() / fdr_tikz["alpha"].to_numpy(),
            )
            np.testing.assert_allclose(
                fdr_tikz["value"].to_numpy(),
                fdr_tikz["fdr_ratio"].to_numpy(),
            )

            fdr_reference = pd.read_csv(
                path_patches["FDR_RATIO_ATLAS_REFERENCE_TIKZ_PATH"]
            )
            self.assertEqual(set(fdr_reference["export_version"]), {"tikz-v1"})
            self.assertEqual(set(fdr_reference["figure"]), {"figure1_supp"})
            self.assertEqual(set(fdr_reference["group"]), {"contour_reference"})
            self.assertEqual(
                set(fdr_reference["style_key"]),
                {"median_log10_rank_delta_zero"},
            )


if __name__ == "__main__":
    unittest.main()
