import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from wkde_cad.simulations import figure2_weighted_power_atlas as weighted


class Figure2WeightedPowerAtlasTest(unittest.TestCase):
    def test_default_output_leaf_matches_generator_stem(self):
        self.assertEqual(weighted.OUT_DIR.name, Path(weighted.__file__).stem)
        self.assertEqual(weighted.OUT_DIR.parent.name, "simulations")
        self.assertEqual(weighted.OUT_DIR.parent.parent.name, "outputs")

    def test_default_neff_bins_match_unweighted_atlas_range(self):
        self.assertEqual(len(weighted.N_EFF_BINS), 15)
        self.assertAlmostEqual(float(weighted.N_EFF_BINS[0]), 1.3)
        self.assertAlmostEqual(float(weighted.N_EFF_BINS[-1]), 4.1)

    def test_top_neff_bin_has_uncapped_uniform_weight_candidate(self):
        candidates = weighted.neff_config_candidates(4.0)

        self.assertTrue(
            any(
                rho == 0.0
                and weighted.N_EFF_BINS[-2]
                <= np.log10(n_cal)
                < weighted.N_EFF_BINS[-1]
                for n_cal, rho in candidates
            )
        )

    def test_weighted_tail_p_values_match_self_atom_formula(self):
        sorted_calib_scores = np.array([0.0, 1.0, 2.0])
        sorted_calib_weights = np.array([1.0, 2.0, 3.0])
        suffix_calib_weights = np.concatenate(
            ([0.0], np.cumsum(sorted_calib_weights[::-1]))
        )[::-1]

        p_values = weighted.weighted_tail_p_values(
            sorted_calib_scores,
            suffix_calib_weights,
            total_calib_weight=6.0,
            test_scores=np.array([1.0, 3.0]),
            test_weights=np.array([4.0, 5.0]),
        )

        np.testing.assert_allclose(p_values, np.array([0.9, 5.0 / 11.0]))

    def test_summary_cache_is_keyed_by_trial_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            pd.DataFrame(
                {
                    "summary_version": [weighted.SUMMARY_VERSION] * 2,
                    "count": [100, 100],
                }
            ).to_csv(path, index=False)

            self.assertTrue(
                weighted.summary_is_current(path, weighted.SUMMARY_VERSION, 100)
            )
            self.assertFalse(
                weighted.summary_is_current(path, weighted.SUMMARY_VERSION, 1)
            )

    def test_tiny_run_writes_only_weighted_power_atlas_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path_patches = {
                "OUT_DIR": root,
                "POWER_ATLAS_FIGURE_PATH": root
                / "figure2_weighted_power_atlas.png",
                "POWER_ATLAS_SUMMARY_PATH": root
                / "figure2_weighted_power_atlas_summary.csv",
                "POWER_ATLAS_TIKZ_PATH": root
                / "figure2_weighted_power_atlas_tikz.csv",
                "POWER_ATLAS_REFERENCE_TIKZ_PATH": root
                / "figure2_weighted_power_atlas_reference_tikz.csv",
            }
            value_patches = {
                "N_EFF_BINS": np.array([0.0, 1.0, 2.0]),
                "RHO_CANDIDATES": [0.0],
                "N_CAL_MAX": 100,
                "MAX_ACCEPT_ATTEMPTS_PER_CELL": 200,
                "M_VALUES": [8, 10],
                "PI1_VALUES": [0.2, 0.3],
                "ALPHA_VALUES": [0.1, 0.2],
                "BASELINE_M": 10,
                "BASELINE_N_ANOMALY": 2,
                "tqdm": lambda iterable, **kwargs: iterable,
            }

            with mock.patch.multiple(weighted, **path_patches), mock.patch.multiple(
                weighted,
                **value_patches,
            ):
                weighted.main(
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

            self.assertEqual(
                {path.name for path in root.iterdir()},
                {
                    Path(path).name
                    for path in path_patches.values()
                    if Path(path).suffix
                },
            )
            for path in path_patches.values():
                self.assertTrue(Path(path).exists(), path)

            atlas = pd.read_csv(path_patches["POWER_ATLAS_SUMMARY_PATH"])
            expected_atlas_rows = (
                len(weighted.SCORE_REGIMES)
                * (2 + 2 + 2 + 2)
                * (len(value_patches["N_EFF_BINS"]) - 1)
            )
            self.assertEqual(len(atlas), expected_atlas_rows)
            self.assertTrue((atlas["count"] > 0).all())
            self.assertEqual(set(atlas["collection"]), {"atlas"})
            self.assertEqual(set(atlas["score_regime"]), set(weighted.SCORE_REGIMES))
            self.assertTrue(
                atlas[atlas["score_regime"].eq("perfect")][
                    "perfect_separation_rate"
                ]
                .eq(1.0)
                .all()
            )
            for column in [
                "discovery_probability",
                "power",
                "fdr",
                "rank_delta_above_one_rate",
                "mean_auroc",
            ]:
                self.assertTrue(
                    ((atlas[column] >= 0.0) & (atlas[column] <= 1.0)).all()
                )
            self.assertTrue(np.isfinite(atlas["median_log10_rank_delta"]).all())

            atlas_tikz = pd.read_csv(path_patches["POWER_ATLAS_TIKZ_PATH"])
            self.assertEqual(len(atlas_tikz), len(atlas))
            self.assertEqual(set(atlas_tikz["export_version"]), {"tikz-v1"})
            self.assertEqual(set(atlas_tikz["figure"]), {"figure2"})
            self.assertEqual(set(atlas_tikz["method"]), {"weighted_wcs_homogeneous"})
            self.assertEqual(set(atlas_tikz["group"]), {"heatmap_cell"})
            self.assertTrue(
                {
                    "panel",
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

            atlas_reference = pd.read_csv(
                path_patches["POWER_ATLAS_REFERENCE_TIKZ_PATH"]
            )
            self.assertEqual(set(atlas_reference["export_version"]), {"tikz-v1"})
            self.assertEqual(set(atlas_reference["figure"]), {"figure2"})
            self.assertEqual(set(atlas_reference["group"]), {"contour_reference"})
            self.assertEqual(
                set(atlas_reference["style_key"]),
                {"median_log10_rank_delta_zero"},
            )


if __name__ == "__main__":
    unittest.main()
