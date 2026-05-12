import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from NEW.Figure2 import figure2_perfect_score_resolution as figure2


class InlineExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def map(self, func, *iterables):
        return map(func, *iterables)


class Figure2PerfectScoreResolutionTest(unittest.TestCase):
    def test_tiny_run_writes_supported_phase_grid_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase_view_limits = figure2.PHASE_VIEW_LIMITS
            phase_x_bins = 2
            phase_y_bins = 2
            path_patches = {
                "OUT_DIR": root,
                "FIGURE_PATH": root / "figure2_perfect_score_resolution.png",
                "SUMMARY_PATH": root / "figure2_perfect_score_summary.csv",
                "TABLE_PATH": root / "figure2_key_settings_table.csv",
                "POWER_CONFIG_FIGURE_PATH": root / "figure2_power_configurations.png",
                "POWER_CONFIG_SUMMARY_PATH": (
                    root / "figure2_power_configurations_summary.csv"
                ),
                "PANEL_A_SCORES_TIKZ_PATH": root / "figure2_panel_a_scores_tikz.csv",
                "PANEL_A_ANNOTATIONS_TIKZ_PATH": (
                    root / "figure2_panel_a_annotations_tikz.csv"
                ),
                "PANEL_B_HEATMAP_TIKZ_PATH": root / "figure2_panel_b_heatmap_tikz.csv",
                "PANEL_B_BOUNDARY_TIKZ_PATH": root / "figure2_panel_b_boundary_tikz.csv",
                "PANEL_C_DETECTABILITY_TIKZ_PATH": (
                    root / "figure2_panel_c_detectability_tikz.csv"
                ),
                "PANEL_C_REFERENCE_TIKZ_PATH": root / "figure2_panel_c_reference_tikz.csv",
                "POWER_CONFIGURATIONS_TIKZ_PATH": (
                    root / "figure2_power_configurations_tikz.csv"
                ),
            }
            value_patches = {
                "N_VALUES": [3, 5],
                "M_VALUES": [4, 5],
                "RHO_VALUES": np.array([0.0]),
                "PANEL_C_M_VALUES": [4, 5],
                "N_SEEDS": 1,
                "WORKERS": 1,
                "PHASE_X_BINS": phase_x_bins,
                "PHASE_Y_BINS": phase_y_bins,
                "PHASE_CELL_TRIALS": 1,
                "PHASE_MAX_ACCEPT_ATTEMPTS_PER_CELL": 100,
                "PHASE_RHO_CANDIDATES": [0.0],
                "LOG_ESS_BINS": np.linspace(0.0, 2.0, 5),
                "DELTA_BINS": np.linspace(-2.0, 2.0, 9),
                "ProcessPoolExecutor": InlineExecutor,
            }

            with mock.patch.multiple(figure2, **path_patches), mock.patch.multiple(
                figure2,
                **value_patches,
            ):
                figure2.main()

            self.assertEqual(phase_view_limits, (2.25, 4.75))
            summary = pd.read_csv(path_patches["SUMMARY_PATH"])
            self.assertEqual(len(summary), phase_x_bins * phase_y_bins)
            self.assertTrue((summary["count"] > 0).all())
            self.assertTrue(summary["perfect_separation_from_calibration"].all())
            np.testing.assert_allclose(
                sorted(summary["x_left"].unique()),
                np.linspace(*phase_view_limits, phase_x_bins + 1)[:-1],
            )
            np.testing.assert_allclose(
                sorted(summary["x_right"].unique()),
                np.linspace(*phase_view_limits, phase_x_bins + 1)[1:],
            )
            np.testing.assert_allclose(
                sorted(summary["y_bottom"].unique()),
                np.linspace(*phase_view_limits, phase_y_bins + 1)[:-1],
            )
            np.testing.assert_allclose(
                sorted(summary["y_top"].unique()),
                np.linspace(*phase_view_limits, phase_y_bins + 1)[1:],
            )

            boundary = pd.read_csv(path_patches["PANEL_B_BOUNDARY_TIKZ_PATH"])
            points = boundary.sort_values("plot_order")[["x", "y"]].to_numpy()
            np.testing.assert_allclose(
                points,
                np.array(
                    [
                        [phase_view_limits[0], phase_view_limits[0]],
                        [phase_view_limits[1], phase_view_limits[1]],
                    ]
                ),
            )
            self.assertEqual(set(boundary["export_version"]), {"tikz-v2"})

            for path in path_patches.values():
                if Path(path).suffix:
                    self.assertTrue(Path(path).exists(), path)


if __name__ == "__main__":
    unittest.main()
