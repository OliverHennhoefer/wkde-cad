import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

from src.scripts import covariate_shift_summary


def _rows_for(approach: str, fdr_values: list[float]) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "dataset": "demo",
            "approach": approach,
            "severity": 0.0,
            "weight_mode": "estimated",
            "fdr": fdr,
            "power": 0.5,
            "n_train": 10,
            "n_test": 20,
            "propensity_std": 0.1,
            "propensity_min_observed": 0.3,
            "propensity_max_observed": 0.7,
            "normal_test_assignment_rate": 0.5,
            "oracle_calib_weight_max": 1.0,
            "oracle_calib_weight_ess": 10.0,
            "oracle_test_weight_max": 1.2,
            "oracle_test_weight_ess": 20.0,
        }
        for seed, fdr in enumerate(fdr_values, start=1)
    ]


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            *_rows_for("empirical", [0.02, 0.02, 0.02]),
            *_rows_for("empirical_randomized", [0.0, 0.2]),
            *_rows_for("probabilistic", [0.2, 0.2, 0.2]),
        ]
    )


class CovariateShiftSummaryTest(unittest.TestCase):
    def test_compute_summary_adds_fdr_control_symbols(self):
        summary = covariate_shift_summary.compute_summary(
            _summary_frame(),
            alpha=0.1,
            delta=0.05,
        )

        controls = dict(zip(summary["approach"], summary["fdr_control"], strict=True))
        symbols = dict(
            zip(summary["approach"], summary["fdr_control_symbol"], strict=True)
        )

        self.assertEqual(controls["empirical"], "valid")
        self.assertEqual(symbols["empirical"], "+")
        self.assertEqual(controls["empirical_randomized"], "inconclusive")
        self.assertEqual(symbols["empirical_randomized"], "=")
        self.assertEqual(controls["probabilistic"], "violated")
        self.assertEqual(symbols["probabilistic"], "-")

    def test_table_output_includes_fdr_control_symbol_column(self):
        summary = covariate_shift_summary.compute_summary(
            pd.DataFrame(_rows_for("empirical", [0.02, 0.02, 0.02])),
            alpha=0.1,
            delta=0.05,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            covariate_shift_summary.print_table(summary)

        rendered = output.getvalue()
        self.assertIn("Ctrl", rendered)
        self.assertRegex(rendered, r"empirical\s+0\.020 \+/- 0\.000\s+\+")

    def test_csv_output_includes_fdr_control_columns(self):
        summary = covariate_shift_summary.compute_summary(
            _summary_frame(),
            alpha=0.1,
            delta=0.05,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            covariate_shift_summary.print_csv(summary)

        rendered = pd.read_csv(io.StringIO(output.getvalue()))
        self.assertIn("fdr_control", rendered.columns)
        self.assertIn("fdr_control_symbol", rendered.columns)
        self.assertEqual(
            rendered.loc[
                rendered["approach"] == "probabilistic",
                "fdr_control_symbol",
            ].iloc[0],
            "-",
        )


if __name__ == "__main__":
    unittest.main()
