import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from wkde_cad.simulations import figure3_operational_sufficiency as figure3


def _atlas_fixture() -> pd.DataFrame:
    rows = []
    diagnostics = (-1.0, -0.2, 0.2, 1.0)
    powers = (0.95, 0.70, 0.20, 0.0)
    for regime in figure3.SCORE_REGIMES:
        for index, (diagnostic, power) in enumerate(zip(diagnostics, powers)):
            rows.append(
                {
                    "score_regime": regime,
                    "median_log10_rank_delta": diagnostic,
                    "power": max(0.0, power - 0.05 * index),
                }
            )
    return pd.DataFrame(rows)


class Figure3OperationalSufficiencyTest(unittest.TestCase):
    def test_default_output_leaf_matches_generator_stem(self):
        self.assertEqual(figure3.OUT_DIR.name, Path(figure3.__file__).stem)
        self.assertEqual(figure3.OUT_DIR.parent.name, "simulations")
        self.assertEqual(figure3.OUT_DIR.parent.parent.name, "outputs")

    def test_collapse_uses_both_atlases_and_equal_cell_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unweighted_path = root / "unweighted.csv"
            weighted_path = root / "weighted.csv"
            _atlas_fixture().to_csv(unweighted_path, index=False)
            _atlas_fixture().assign(power=lambda frame: frame["power"] * 0.8).to_csv(
                weighted_path,
                index=False,
            )

            summary = figure3.build_collapse_summary(
                unweighted_path,
                weighted_path,
                bins=4,
            )

        self.assertEqual(set(summary["method"]), {"unweighted", "weighted"})
        self.assertEqual(set(summary["score_regime"]), set(figure3.SCORE_REGIMES))
        self.assertEqual(set(summary["aggregation_unit"]), {"equal_atlas_cells"})
        self.assertTrue((summary["cell_count"] > 0).all())
        self.assertTrue(
            ((summary["power_mean"] >= 0.0) & (summary["power_mean"] <= 1.0)).all()
        )

    def test_n_cal_candidates_are_positive_bounded_and_rho_sensitive(self):
        unshifted = figure3.n_cal_candidates(2.0, 0.0, n_cal_max=2_000)
        shifted = figure3.n_cal_candidates(2.0, 1.0, n_cal_max=2_000)

        self.assertTrue(all(2 <= value <= 2_000 for value in unshifted + shifted))
        self.assertGreater(np.median(shifted), np.median(unshifted))

    def test_tiny_run_writes_auditable_outputs_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unweighted_path = root / "figure1_summary.csv"
            weighted_path = root / "figure2_summary.csv"
            _atlas_fixture().to_csv(unweighted_path, index=False)
            _atlas_fixture().to_csv(weighted_path, index=False)

            path_patches = {
                "OUT_DIR": root,
                "FIGURE_PATH": root / "figure3_operational_sufficiency.png",
                "COLLAPSE_SUMMARY_PATH": root
                / "figure3_atlas_collapse_summary.csv",
                "MATCHED_TRIALS_PATH": root / "figure3_matched_neff_trials.csv",
                "MATCHED_SUMMARY_PATH": root / "figure3_matched_neff_summary.csv",
                "UNWEIGHTED_ATLAS_SUMMARY_PATH": unweighted_path,
                "WEIGHTED_ATLAS_SUMMARY_PATH": weighted_path,
            }
            value_patches = {
                "TARGET_LOG10_NEFF": (1.0,),
                "MATCH_RHOS": (0.0,),
                "MATCH_M": 20,
                "MATCH_PI1": 0.2,
                "tqdm": lambda iterable, **kwargs: iterable,
            }
            argv = [
                "--collapse-bins",
                "4",
                "--matched-trials",
                "1",
                "--neff-half-width",
                "0.2",
                "--max-attempts-per-cell",
                "20",
                "--wcs-batch-size",
                "8",
                "--workers",
                "1",
                "--force",
            ]

            with mock.patch.multiple(figure3, **path_patches), mock.patch.multiple(
                figure3, **value_patches
            ):
                figure3.main(argv)
                with mock.patch.object(
                    figure3,
                    "run_matched_tasks",
                    side_effect=AssertionError("compatible cache should be reused"),
                ):
                    figure3.main([value for value in argv if value != "--force"])

            for path in path_patches.values():
                path = Path(path)
                if path.suffix in {".csv", ".png"}:
                    self.assertTrue(path.exists(), path)

            trials = pd.read_csv(path_patches["MATCHED_TRIALS_PATH"])
            matched = pd.read_csv(path_patches["MATCHED_SUMMARY_PATH"])
            collapse = pd.read_csv(path_patches["COLLAPSE_SUMMARY_PATH"])

            self.assertEqual(len(trials), 1)
            self.assertLessEqual(
                abs(float(trials["log10_neff"].iloc[0]) - 1.0),
                0.2 + 1e-12,
            )
            self.assertEqual(set(matched["score_regime"]), {"perfect", "finite_k3"})
            self.assertEqual(len(matched), 2)
            self.assertEqual(set(collapse["method"]), {"unweighted", "weighted"})
            self.assertTrue(
                {
                    "attempts_for_cell",
                    "acceptance_rate",
                    "n_cal_mean",
                    "log10_neff_min",
                    "log10_neff_max",
                    "power_ci95_low",
                    "power_ci95_high",
                }.issubset(matched.columns)
            )


if __name__ == "__main__":
    unittest.main()
