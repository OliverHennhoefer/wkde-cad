import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from NEW.Figure1 import figure1_phase_transition as figure1


class InlineExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def map(self, func, *iterables):
        return map(func, *iterables)


def reference_weighted_cs_decisions(
    p_values,
    test_scores,
    sorted_calib_scores,
    sorted_calib_weights,
    total_calib_weight,
    test_weights,
    alpha,
    pruning,
    seed,
):
    # Upper-tail sign adaptation of conformal-selection's weighted_CS loop.
    m = len(test_scores)
    cumulative_calib_weights = np.concatenate(([0.0], np.cumsum(sorted_calib_weights)))
    strict_tail_start = np.searchsorted(sorted_calib_scores, test_scores, side="right")
    calib_mass_strictly_above = (
        total_calib_weight - cumulative_calib_weights[strict_tail_start]
    )
    rejection_sizes = np.empty(m, dtype=int)
    for candidate in range(m):
        auxiliary_p_values = (
            calib_mass_strictly_above
            + test_weights * (test_scores < test_scores[candidate])
        ) / (total_calib_weight + test_weights[candidate])
        auxiliary_p_values[candidate] = 0.0
        rejection_sizes[candidate] = figure1._bh_rejection_counts_by_row(
            auxiliary_p_values[None, :],
            alpha,
        )[0]

    first_step_idx = np.flatnonzero(p_values <= alpha * rejection_sizes / m)
    if len(first_step_idx) == 0:
        return np.zeros(m, dtype=bool)

    rng = np.random.default_rng(seed)
    if pruning == "heterogeneous":
        metrics = rng.uniform(size=m)[first_step_idx] * rejection_sizes[first_step_idx]
    elif pruning == "homogeneous":
        metrics = rng.uniform() * rejection_sizes[first_step_idx]
    else:
        metrics = rejection_sizes[first_step_idx].astype(float)

    final_idx = figure1._select_by_pruning_metrics(first_step_idx, metrics)
    decisions = np.zeros(m, dtype=bool)
    decisions[final_idx] = True
    return decisions


class Figure1PhaseTransitionTest(unittest.TestCase):
    def test_standard_tail_p_values_and_bh_decisions(self):
        sorted_calib_scores = np.array([0.0, 1.0, 2.0])

        p_values = figure1.standard_tail_p_values(
            sorted_calib_scores,
            np.array([-1.0, 0.5, 2.0, 3.0]),
        )

        np.testing.assert_allclose(p_values, np.array([1.0, 0.75, 0.5, 0.25]))
        decisions = figure1.bh_decisions(np.array([0.01, 0.02, 0.5]), alpha=0.06)
        np.testing.assert_array_equal(decisions, np.array([True, True, False]))

    def test_weighted_tail_p_values_match_current_formula(self):
        sorted_calib_scores = np.array([0.0, 1.0, 2.0])
        sorted_calib_weights = np.array([1.0, 2.0, 3.0])
        suffix_calib_weights = np.concatenate(
            ([0.0], np.cumsum(sorted_calib_weights[::-1]))
        )[::-1]

        p_values = figure1.weighted_tail_p_values(
            sorted_calib_scores,
            suffix_calib_weights,
            total_calib_weight=6.0,
            test_scores=np.array([1.0, 3.0]),
            test_weights=np.array([4.0, 5.0]),
        )

        np.testing.assert_allclose(p_values, np.array([0.9, 5.0 / 11.0]))

    def test_accelerated_wcs_matches_conformal_selection_reference(self):
        rng = np.random.default_rng(20260511)
        base_calib_scores = rng.normal(size=30)
        base_test_scores = rng.normal(size=40)
        base_test_scores[-5:] += 4.0
        calib_weights = np.exp(rng.normal(scale=0.5, size=len(base_calib_scores)))
        test_weights = np.exp(rng.normal(scale=0.5, size=len(base_test_scores)))

        for score_case, calib_scores, test_scores in [
            ("continuous", base_calib_scores, base_test_scores),
            ("tied", np.round(base_calib_scores, 1), np.round(base_test_scores, 1)),
        ]:
            order = np.argsort(calib_scores, kind="mergesort")
            sorted_calib_scores = calib_scores[order]
            sorted_calib_weights = calib_weights[order]
            suffix_calib_weights = np.concatenate(
                ([0.0], np.cumsum(sorted_calib_weights[::-1]))
            )[::-1]
            p_values = figure1.weighted_tail_p_values(
                sorted_calib_scores,
                suffix_calib_weights,
                float(np.sum(calib_weights)),
                test_scores,
                test_weights,
            )
            for pruning_name in figure1.WCS_PRUNING_METHODS:
                with self.subTest(score_case=score_case, pruning=pruning_name):
                    expected = reference_weighted_cs_decisions(
                        p_values,
                        test_scores,
                        sorted_calib_scores,
                        sorted_calib_weights,
                        float(np.sum(calib_weights)),
                        test_weights,
                        alpha=0.1,
                        pruning=pruning_name,
                        seed=7,
                    )
                    observed = figure1.accelerated_wcs_decisions(
                        p_values,
                        test_scores,
                        sorted_calib_scores,
                        sorted_calib_weights,
                        float(np.sum(calib_weights)),
                        test_weights,
                        alpha=0.1,
                        pruning=pruning_name,
                        seed=7,
                    )
                    np.testing.assert_array_equal(observed, expected)

    def test_accelerated_wcs_skips_when_no_candidate_can_pass(self):
        decisions = figure1.accelerated_wcs_decisions(
            p_values=np.array([0.2, 0.3, 0.4]),
            test_scores=np.array([1.0, 2.0, 3.0]),
            sorted_calib_scores=np.array([0.0, 0.5]),
            sorted_calib_weights=np.array([1.0, 1.0]),
            total_calib_weight=2.0,
            test_weights=np.ones(3),
            alpha=0.1,
            pruning="homogeneous",
            seed=1,
        )

        np.testing.assert_array_equal(decisions, np.zeros(3, dtype=bool))

    def test_heatmap_matrix_handles_sparse_original_bin_ids(self):
        summary = pd.DataFrame(
            [
                {
                    "scenario": "baseline",
                    "kappa": "inf",
                    "x_bin": 7,
                    "y_bin": 27,
                    "x_left": 1.0,
                    "x_right": 2.0,
                    "y_bottom": 4.0,
                    "y_top": 5.0,
                    "probability": 0.75,
                },
                {
                    "scenario": "baseline",
                    "kappa": "inf",
                    "x_bin": 9,
                    "y_bin": 31,
                    "x_left": 2.0,
                    "x_right": 3.0,
                    "y_bottom": 5.0,
                    "y_top": 6.0,
                    "probability": 0.25,
                },
            ]
        )

        x_edges, y_edges, matrix = figure1.heatmap_matrix(summary, "baseline", "inf")

        np.testing.assert_allclose(x_edges, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(y_edges, np.array([4.0, 5.0, 6.0]))
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(float(matrix[0, 0]), 0.75)
        self.assertEqual(float(matrix[1, 1]), 0.25)

    def test_fixed_heatmap_edges_cover_viewport(self):
        with mock.patch.multiple(
            figure1,
            HEATMAP_VIEW_LIMITS=(1.0, 2.0),
            HEATMAP_X_BINS=2,
            HEATMAP_Y_BINS=4,
        ):
            np.testing.assert_allclose(
                figure1.heatmap_x_edges(),
                np.array([1.0, 1.5, 2.0]),
            )
            np.testing.assert_allclose(
                figure1.heatmap_y_edges(),
                np.array([1.0, 1.25, 1.5, 1.75, 2.0]),
            )

    def test_default_tasks_skip_partial_supplement_grid(self):
        patches = {
            "N_VALUES": [10],
            "M_VALUES": [20, 30],
            "RHO_VALUES": np.array([0.0]),
            "SUPPLEMENT_N_VALUES": [1000],
            "SUPPLEMENT_M_VALUES": [20],
            "SUPPLEMENT_RHO_VALUES": np.array([0.0]),
            "INCLUDE_SUPPLEMENT_GRID": False,
        }

        with mock.patch.multiple(figure1, **patches):
            tasks = figure1.tasks_for_scenario("unweighted", "baseline")

        self.assertEqual(tasks, [("unweighted", "base", "baseline", 10, 0.0, (20, 30))])

    def test_tiny_default_run_writes_unweighted_and_weighted_schema_compatible_outputs(self):
        expected_heatmap_columns = [
            "summary_version",
            "scenario",
            "alpha",
            "pi1",
            "kappa",
            "x_bin",
            "y_bin",
            "x_left",
            "x_right",
            "y_bottom",
            "y_top",
            "successes",
            "count",
            "probability",
        ]
        expected_collapse_columns = [
            "summary_version",
            "scenario",
            "diagnostic",
            "kappa",
            "bin",
            "x_left",
            "x_right",
            "x",
            "successes",
            "count",
            "probability",
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            heatmap_view_limits = (0.8, 1.2)
            heatmap_x_bins = 2
            heatmap_y_bins = 2
            heatmap_expected_rows = (
                len(figure1.SCENARIOS)
                * len(figure1.HEATMAP_KAPPAS)
                * heatmap_x_bins
                * heatmap_y_bins
            )
            patches = {
                "OUT_ROOT": root,
                "N_VALUES": [2, 3],
                "M_VALUES": [4, 5],
                "RHO_VALUES": np.array([0.0]),
                "SUPPLEMENT_N_VALUES": [],
                "SUPPLEMENT_M_VALUES": [4],
                "SUPPLEMENT_RHO_VALUES": np.array([0.0]),
                "N_SEEDS": 1,
                "WORKERS": 1,
                "HEATMAP_VIEW_LIMITS": heatmap_view_limits,
                "HEATMAP_X_BINS": heatmap_x_bins,
                "HEATMAP_Y_BINS": heatmap_y_bins,
                "HEATMAP_CELL_TRIALS": 1,
                "HEATMAP_MAX_ACCEPT_ATTEMPTS_PER_CELL": 200,
                "WEIGHTED_HEATMAP_RHO_CANDIDATES": [0.5, 1.0],
                "COLLAPSE_BINS": np.linspace(-2.0, 3.0, 6),
                "ProcessPoolExecutor": InlineExecutor,
            }

            with mock.patch.multiple(figure1, **patches):
                figure1.main([])

            for mode in figure1.MODES:
                out_dir = root / mode
                self.assertTrue(out_dir.exists())
                heatmap = pd.read_csv(out_dir / "figure1_heatmap_summary.csv")
                collapse = pd.read_csv(out_dir / "figure1_collapse_summary.csv")
                self.assertEqual(heatmap.columns.tolist(), expected_heatmap_columns)
                self.assertEqual(collapse.columns.tolist(), expected_collapse_columns)
                self.assertEqual(
                    set(heatmap["scenario"]),
                    {scenario["name"] for scenario in figure1.SCENARIOS},
                )
                self.assertEqual(len(heatmap), heatmap_expected_rows)
                self.assertTrue((heatmap["count"] > 0).all())
                np.testing.assert_allclose(
                    sorted(heatmap["x_left"].unique()),
                    np.linspace(*heatmap_view_limits, heatmap_x_bins + 1)[:-1],
                )
                np.testing.assert_allclose(
                    sorted(heatmap["x_right"].unique()),
                    np.linspace(*heatmap_view_limits, heatmap_x_bins + 1)[1:],
                )
                np.testing.assert_allclose(
                    sorted(heatmap["y_bottom"].unique()),
                    np.linspace(*heatmap_view_limits, heatmap_y_bins + 1)[:-1],
                )
                np.testing.assert_allclose(
                    sorted(heatmap["y_top"].unique()),
                    np.linspace(*heatmap_view_limits, heatmap_y_bins + 1)[1:],
                )
                for _, block in heatmap.groupby(["scenario", "kappa"]):
                    n_x = block[["x_left", "x_right"]].drop_duplicates().shape[0]
                    n_y = block[["y_bottom", "y_top"]].drop_duplicates().shape[0]
                    self.assertEqual(len(block), n_x * n_y)
                self.assertEqual(set(collapse["scenario"]), {figure1.BASELINE_SCENARIO})
                for filename in [
                    "figure1_panel_a_schematic.png",
                    "figure1_heatmaps_alpha_pi_sensitivity.png",
                    "figure1_collapse_diagnostics.png",
                    "figure1_schematic_points_tikz.csv",
                    "figure1_schematic_annotations_tikz.csv",
                    "figure1_heatmap_tikz.csv",
                    "figure1_heatmap_boundary_tikz.csv",
                    "figure1_collapse_tikz.csv",
                    "figure1_collapse_reference_tikz.csv",
                ]:
                    self.assertTrue((out_dir / filename).exists(), filename)

                heatmap_tikz = pd.read_csv(out_dir / "figure1_heatmap_tikz.csv")
                panel_order = (
                    heatmap_tikz.groupby("panel", as_index=False)
                    .agg(
                        scenario=("scenario", "first"),
                        kappa=("kappa", "first"),
                        plot_order=("plot_order", "first"),
                    )
                    .sort_values("plot_order")
                )
                panel_order["kappa"] = panel_order["kappa"].astype(str)
                self.assertEqual(
                    panel_order[["panel", "scenario", "kappa", "plot_order"]]
                    .to_records(index=False)
                    .tolist(),
                    [
                        ("heatmap_r1_c1", "baseline", "inf", 1),
                        ("heatmap_r1_c2", "alpha_005", "inf", 2),
                        ("heatmap_r1_c3", "pi1_001", "inf", 3),
                        ("heatmap_r2_c1", "baseline", "3.0", 4),
                        ("heatmap_r2_c2", "alpha_005", "3.0", 5),
                        ("heatmap_r2_c3", "pi1_001", "3.0", 6),
                    ],
                )
                self.assertEqual(set(heatmap_tikz["export_version"]), {"tikz-v3"})

                boundary = pd.read_csv(out_dir / "figure1_heatmap_boundary_tikz.csv")
                for _, block in boundary.groupby("panel"):
                    points = block.sort_values("plot_order")[["x", "y"]].to_numpy()
                    np.testing.assert_allclose(
                        points,
                        np.array(
                            [
                                [heatmap_view_limits[0], heatmap_view_limits[0]],
                                [heatmap_view_limits[1], heatmap_view_limits[1]],
                            ]
                        ),
                    )
                self.assertEqual(set(boundary["export_version"]), {"tikz-v3"})


if __name__ == "__main__":
    unittest.main()
