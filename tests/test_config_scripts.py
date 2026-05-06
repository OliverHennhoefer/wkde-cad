import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.scripts import plot_covariate_shift


class ConfigScriptTest(unittest.TestCase):
    def test_plot_covariate_shift_reads_restructured_config_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.toml"
            config_path.write_text(
                """
[experiment]
meta_seeds = 2
datasets = ["demo"]
severities = [0.0, 1.0]

[splits]
train_split = 0.75
test_use_proportion = 0.4
test_anomaly_rate = 0.1

[covariate_shift]
propensity_min = 0.2
propensity_max = 0.8

[plots]
output_dir = "plots"
bins = 7
""",
                encoding="utf-8",
            )

            normal = pd.DataFrame(
                {
                    "x1": [0.0, 1.0],
                    "x2": [1.0, 2.0],
                    "Class": [0, 0],
                }
            )
            anomaly = pd.DataFrame(
                {
                    "x1": [3.0],
                    "x2": [4.0],
                    "Class": [1],
                }
            )
            captured = {}

            def fake_plot_dataset(**kwargs):
                captured.update(kwargs)
                return tmp_path / "plots" / "demo.png"

            with (
                mock.patch.object(plot_covariate_shift, "DEFAULT_CONFIG", config_path),
                mock.patch.object(plot_covariate_shift, "REPO_ROOT", tmp_path),
                mock.patch.object(
                    plot_covariate_shift,
                    "_load_dataset_data",
                    return_value=(normal, anomaly, ["x1", "x2"]),
                ),
                mock.patch.object(
                    plot_covariate_shift,
                    "_plot_dataset",
                    side_effect=fake_plot_dataset,
                ),
            ):
                plot_covariate_shift.main()

            self.assertEqual(captured["dataset"], "demo")
            self.assertEqual(captured["train_split"], 0.75)
            self.assertEqual(captured["test_use_proportion"], 0.4)
            self.assertEqual(captured["test_anomaly_rate"], 0.1)
            self.assertEqual(captured["severities"], [0.0, 1.0])
            self.assertEqual(captured["propensity_min"], 0.2)
            self.assertEqual(captured["propensity_max"], 0.8)
            self.assertEqual(captured["seeds"], [1, 2])
            self.assertEqual(captured["output_dir"], tmp_path / "plots")
            self.assertEqual(captured["bins"], 7)

    def test_plot_covariate_shift_uses_seed_specific_evaluation_pool(self):
        normal = pd.DataFrame(
            {
                "x1": [float(i) for i in range(16)],
                "x2": [float(i * i) for i in range(16)],
                "Class": [0] * 16,
            }
        )
        anomaly = pd.DataFrame(
            {
                "x1": [20.0 + i for i in range(8)],
                "x2": [30.0 + i for i in range(8)],
                "Class": [1] * 8,
            },
            index=range(16, 24),
        )

        diagnostics, _ = plot_covariate_shift._split_diagnostics(
            normal=normal,
            anomaly=anomaly,
            feature_columns=["x1", "x2"],
            train_split=0.5,
            test_use_proportion=0.5,
            test_anomaly_rate=0.2,
            severity=0.0,
            propensity_min=0.2,
            propensity_max=0.8,
            seeds=[3],
        )
        split = plot_covariate_shift.split_model_selection_evaluation(
            normal,
            anomaly,
            train_split=0.5,
            seed=3,
        )

        self.assertEqual(
            set(diagnostics["source_index"]),
            set(split.normal_evaluation.index),
        )
        self.assertTrue(
            set(diagnostics["source_index"]).isdisjoint(
                split.normal_model_selection.index
            )
        )


if __name__ == "__main__":
    unittest.main()
