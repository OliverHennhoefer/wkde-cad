import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import src.model_selection as model_selection
import src.rebuttal.covariate_shift_experiment as covariate_shift_experiment


def _cfg(meta_seeds=2):
    return {
        "global": {
            "meta_seeds": meta_seeds,
            "train_split": 0.75,
            "selection_folds": 3,
            "fdr_rate": 0.1,
            "n_bootstraps": 2,
            "n_trials": 2,
            "test_use_proportion": 0.5,
            "test_anomaly_rate": 0.2,
            "approaches": ["empirical"],
            "pruning": "homogeneous",
        },
        "experiments": {
            "datasets": ["demo"],
            "models": ["linear"],
        },
        "rebuttal_covariate_shift": {
            "severities": [0.0],
            "propensity_min": 0.3,
            "propensity_max": 0.7,
            "weight_mode": "oracle",
        },
    }


def _dataframe():
    normal = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, 24),
            "x2": np.linspace(1.0, 2.0, 24),
            "Class": np.zeros(24, dtype=int),
        }
    )
    anomaly = pd.DataFrame(
        {
            "x1": np.linspace(1.2, 2.0, 10),
            "x2": np.linspace(2.2, 3.0, 10),
            "Class": np.ones(10, dtype=int),
        },
        index=range(24, 34),
    )
    return pd.concat([normal, anomaly])


class DeterministicDetector:
    def fit(self, x_train):
        return self

    def decision_function(self, x_test):
        return x_test[:, 0] + 0.1 * x_test[:, 1]


class SharedModelSelectionTest(unittest.TestCase):
    def test_existing_selection_csv_skips_without_loading_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "demo.csv").write_text("seed,dataset\n1,demo\n")
            logger = mock.Mock()

            with mock.patch(
                "src.model_selection.load",
                side_effect=AssertionError("load should not be called"),
            ):
                model_selection.run_model_selection(
                    cfg=_cfg(),
                    datasets=["demo"],
                    seeds=[1],
                    output_dir=output_dir,
                    jobs=1,
                    logger=logger,
                )

            logger.info.assert_called_with(
                "Skipping model selection for demo (results exist)"
            )

    def test_missing_selection_csv_merges_seed_files_in_seed_order_and_cleans_up(self):
        def write_seed_csv(
            seed,
            ds_name,
            normal,
            anomaly,
            empirical_anomaly_rate,
            models,
            cfg,
            output_dir,
        ):
            pd.DataFrame(
                [
                    {
                        "seed": seed,
                        "dataset": ds_name,
                        "model": models[0],
                        "fold": "mean",
                        "prauc": float(seed),
                        "rocauc": float(seed),
                        "brier": 0.0,
                        "is_best": True,
                    }
                ]
            ).to_csv(output_dir / f"{ds_name}_seed{seed}.csv", index=False)
            return ds_name, seed, models[0]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            with (
                mock.patch("src.model_selection.get_dataset_enum", return_value="demo"),
                mock.patch("src.model_selection.load", return_value=_dataframe()),
                mock.patch(
                    "src.model_selection.process_seed_phase1",
                    side_effect=write_seed_csv,
                ),
            ):
                model_selection.run_model_selection(
                    cfg=_cfg(),
                    datasets=["demo"],
                    seeds=[2, 1],
                    output_dir=output_dir,
                    jobs=1,
                    logger=mock.Mock(),
                )

            merged = pd.read_csv(output_dir / "demo.csv")
            self.assertEqual(merged["seed"].tolist(), [2, 1])
            self.assertFalse((output_dir / "demo_seed2.csv").exists())
            self.assertFalse((output_dir / "demo_seed1.csv").exists())

    def test_selection_worker_is_deterministic_for_same_seed_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            kwargs = {
                "seed": 7,
                "ds_name": "demo",
                "normal": _dataframe().query("Class == 0"),
                "anomaly": _dataframe().query("Class == 1"),
                "empirical_anomaly_rate": 10 / 34,
                "models": ["linear"],
                "cfg": _cfg(),
            }

            with mock.patch(
                "src.model_selection.get_model_instance",
                return_value=DeterministicDetector(),
            ):
                result1 = model_selection.process_seed_phase1(
                    output_dir=Path(tmp1),
                    **kwargs,
                )
                result2 = model_selection.process_seed_phase1(
                    output_dir=Path(tmp2),
                    **kwargs,
                )

            self.assertEqual(result1, result2)
            pd.testing.assert_frame_equal(
                pd.read_csv(Path(tmp1) / "demo_seed7.csv"),
                pd.read_csv(Path(tmp2) / "demo_seed7.csv"),
            )

    def test_covariate_shift_runs_model_selection_for_all_config_seeds(self):
        calls = []

        def write_selection_csv(cfg, datasets, seeds, output_dir, jobs, logger):
            calls.append(
                {
                    "datasets": list(datasets),
                    "seeds": list(seeds),
                    "jobs": jobs,
                }
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "seed": seed,
                        "dataset": datasets[0],
                        "model": "linear",
                        "fold": "mean",
                        "prauc": 1.0,
                        "rocauc": 1.0,
                        "brier": 0.0,
                        "is_best": True,
                    }
                    for seed in seeds
                ]
            ).to_csv(output_dir / f"{datasets[0]}.csv", index=False)

        def fake_process_shift_seed(
            seed,
            severity,
            model_name,
            ds_name,
            normal,
            anomaly,
            cfg,
            approaches_to_run,
            pruning_method,
        ):
            return [
                {
                    "seed": seed,
                    "dataset": ds_name,
                    "model": model_name,
                    "approach": approaches_to_run[0],
                    "severity": severity,
                    "train_size": 1,
                    "test_size": 1,
                    "fdr": 0.0,
                    "power": 1.0,
                }
            ]

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            output_dir = repo_root / "outputs" / "rebuttal"

            with (
                mock.patch.object(covariate_shift_experiment, "REPO_ROOT", repo_root),
                mock.patch.object(
                    covariate_shift_experiment,
                    "run_model_selection",
                    side_effect=write_selection_csv,
                ),
                mock.patch.object(
                    covariate_shift_experiment,
                    "get_dataset_enum",
                    return_value="demo",
                ),
                mock.patch.object(
                    covariate_shift_experiment,
                    "load",
                    return_value=_dataframe(),
                ),
                mock.patch.object(
                    covariate_shift_experiment,
                    "process_shift_seed",
                    side_effect=fake_process_shift_seed,
                ),
            ):
                covariate_shift_experiment.run_experiment(
                    cfg=_cfg(meta_seeds=3),
                    datasets=["demo"],
                    seeds=[1],
                    severities=[0.0],
                    approaches_to_run=["empirical"],
                    output_dir=output_dir,
                    force=False,
                    jobs=1,
                )

            self.assertEqual(calls, [{"datasets": ["demo"], "seeds": [1, 2, 3], "jobs": 1}])
            results = pd.read_csv(output_dir / "demo.csv")
            self.assertEqual(results["seed"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
