import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.scripts import fdr_table


def _result_frame(fdr_values, power_values=None) -> pd.DataFrame:
    if power_values is None:
        power_values = [0.5] * len(fdr_values)

    return pd.DataFrame(
        [
            {
                "seed": seed,
                "dataset": "demo",
                "approach": "probabilistic_weighted",
                "severity": 0.0,
                "weight_mode": "estimated",
                "fdr": fdr,
                "power": power,
            }
            for seed, (fdr, power) in enumerate(zip(fdr_values, power_values), start=1)
        ]
    )


def _approach_frame() -> pd.DataFrame:
    rows = []
    for approach in [
        "empirical_weighted",
        "empirical_randomized_weighted",
        "probabilistic_weighted",
    ]:
        for seed in [1, 2, 3]:
            rows.append(
                {
                    "seed": seed,
                    "dataset": "demo",
                    "approach": approach,
                    "severity": 0.0,
                    "weight_mode": "estimated",
                    "fdr": 0.02,
                    "power": 0.5,
                }
            )
    return pd.DataFrame(rows)


class FdrTableTest(unittest.TestCase):
    def test_classifies_valid_when_upper_ci_is_below_alpha(self):
        summary = fdr_table.compute_summary(
            _result_frame([0.02, 0.02, 0.02]),
            alpha=0.1,
            delta=0.05,
        )

        self.assertEqual(summary["fdr_control"].iloc[0], "valid")

    def test_classifies_invalid_when_lower_ci_is_above_alpha(self):
        summary = fdr_table.compute_summary(
            _result_frame([0.2, 0.2, 0.2]),
            alpha=0.1,
            delta=0.05,
        )

        self.assertEqual(summary["fdr_control"].iloc[0], "invalid")

    def test_classifies_inconclusive_when_ci_overlaps_alpha(self):
        summary = fdr_table.compute_summary(
            _result_frame([0.0, 0.2]),
            alpha=0.1,
            delta=0.05,
        )

        self.assertEqual(summary["fdr_control"].iloc[0], "inconclusive")

    def test_uses_one_sided_critical_value_for_validity_bounds(self):
        fdr_values = [0.03] * 12 + [0.08] + [0.13] * 12

        summary = fdr_table.compute_summary(
            _result_frame(fdr_values),
            alpha=0.1,
            delta=0.05,
        )

        self.assertEqual(summary["fdr_control"].iloc[0], "valid")

    def test_inconclusive_rows_report_mean_side_of_alpha(self):
        below = fdr_table.compute_summary(
            _result_frame([0.0, 0.18]),
            alpha=0.1,
            delta=0.05,
        )
        above = fdr_table.compute_summary(
            _result_frame([0.02, 0.22]),
            alpha=0.1,
            delta=0.05,
        )

        self.assertEqual(below["fdr_control_note"].iloc[0], "FDRhat <= alpha")
        self.assertEqual(above["fdr_control_note"].iloc[0], "FDRhat > alpha")

    def test_summary_rows_are_ignored(self):
        df = _result_frame([0.02])
        summary_row = df.iloc[0].copy()
        summary_row["seed"] = "mean"
        summary_row["fdr"] = 0.8
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.csv"
            df.to_csv(path, index=False)

            loaded = fdr_table.load_and_validate_csv(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(float(loaded["fdr"].iloc[0]), 0.02)

    def test_compute_summary_includes_all_approaches_by_default(self):
        summary = fdr_table.compute_summary(
            _approach_frame(),
            alpha=0.1,
            delta=0.05,
        )

        self.assertEqual(
            summary["approach"].tolist(),
            [
                "empirical_weighted",
                "empirical_randomized_weighted",
                "probabilistic_weighted",
            ],
        )

    def test_markdown_and_latex_include_control_status_column(self):
        summary = fdr_table.compute_summary(
            _result_frame([0.02, 0.02, 0.02]),
            alpha=0.1,
            delta=0.05,
        )

        markdown = fdr_table.render_markdown(summary, delta=0.05, precision=3)
        latex = fdr_table.render_latex(summary, delta=0.05, precision=3)

        self.assertIn("FDR control", markdown)
        self.assertIn("valid", markdown)
        self.assertIn("FDR control", latex)
        self.assertIn("valid", latex)

    def test_renderers_include_inconclusive_mean_side_note(self):
        summary = fdr_table.compute_summary(
            _result_frame([0.02, 0.22]),
            alpha=0.1,
            delta=0.05,
        )

        markdown = fdr_table.render_markdown(summary, delta=0.05, precision=3)
        latex = fdr_table.render_latex(summary, delta=0.05, precision=3)

        self.assertIn("inconclusive (FDRhat > alpha)", markdown)
        self.assertIn(r"inconclusive ($\widehat{\mathrm{FDR}}>\alpha$)", latex)

    def test_renderers_split_by_severity_and_omit_weight_mode_and_n_columns(self):
        summary = fdr_table.compute_summary(
            _approach_frame(),
            alpha=0.1,
            delta=0.05,
        )

        markdown = fdr_table.render_markdown(summary, delta=0.05, precision=3)
        latex = fdr_table.render_latex(summary, delta=0.05, precision=3)

        self.assertIn("### Severity 0", markdown)
        self.assertIn(r"\paragraph{Severity 0}", latex)
        self.assertNotIn("Weight mode", markdown)
        self.assertNotIn("Weight mode", latex)
        self.assertNotIn("| N |", markdown)
        self.assertNotIn(r"\textbf{N}", latex)
        self.assertNotIn("95% CI", markdown)
        self.assertNotIn(r"95\% CI", latex)
        self.assertIn("| FDR |", markdown)
        self.assertIn(r"\textbf{FDR}", latex)

    def test_missing_required_columns_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.csv"
            pd.DataFrame([{"seed": 1, "dataset": "demo"}]).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "Missing required columns"):
                fdr_table.load_and_validate_csv(path)


if __name__ == "__main__":
    unittest.main()
