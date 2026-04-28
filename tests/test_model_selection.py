import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import src.experiment as experiment
import src.model_selection as model_selection


def _cfg(meta_seeds=2):
    return {
        "experiment": {
            "meta_seeds": meta_seeds,
            "datasets": ["demo"],
            "severities": [0.0],
            "output_dir": "outputs/experiment_results",
        },
        "model_selection": {
            "models": ["linear"],
            "folds": 3,
            "output_dir": "outputs/model_selection",
        },
        "splits": {
            "train_split": 0.75,
            "test_use_proportion": 0.5,
            "test_anomaly_rate": 0.2,
        },
        "conformal": {
            "fdr_rate": 0.1,
            "n_bootstraps": 2,
            "n_trials": 2,
            "pruning": "homogeneous",
        },
        "weighting": {
            "mode": "oracle",
            "estimator": "forest",
        },
        "covariate_shift": {
            "propensity_min": 0.3,
            "propensity_max": 0.7,
        },
        "methods": {
            "approaches": ["empirical"],
        },
        "plots": {
            "output_dir": "outputs/experiment_plots",
            "bins": 10,
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
            output_dir = repo_root / "outputs" / "experiment_results"

            with (
                mock.patch.object(experiment, "REPO_ROOT", repo_root),
                mock.patch.object(
                    experiment,
                    "run_model_selection",
                    side_effect=write_selection_csv,
                ),
                mock.patch.object(
                    experiment,
                    "get_dataset_enum",
                    return_value="demo",
                ),
                mock.patch.object(
                    experiment,
                    "load",
                    return_value=_dataframe(),
                ),
                mock.patch.object(
                    experiment,
                    "process_shift_seed",
                    side_effect=fake_process_shift_seed,
                ),
            ):
                experiment.run_experiment(
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

    def test_covariate_shift_filters_unweighted_methods_to_zero_severity(self):
        captured = []

        def write_selection_csv(cfg, datasets, seeds, output_dir, jobs, logger):
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "seed": 1,
                        "dataset": datasets[0],
                        "model": "linear",
                        "fold": "mean",
                        "prauc": 1.0,
                        "rocauc": 1.0,
                        "brier": 0.0,
                        "is_best": True,
                    }
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
            captured.append((severity, list(approaches_to_run)))
            return [
                {
                    "seed": seed,
                    "dataset": ds_name,
                    "model": model_name,
                    "approach": approach,
                    "severity": severity,
                    "train_size": 1,
                    "test_size": 1,
                    "fdr": 0.0,
                    "power": 1.0,
                }
                for approach in approaches_to_run
            ]

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            output_dir = repo_root / "outputs" / "experiment_results"

            with (
                mock.patch.object(experiment, "REPO_ROOT", repo_root),
                mock.patch.object(
                    experiment,
                    "run_model_selection",
                    side_effect=write_selection_csv,
                ),
                mock.patch.object(experiment, "get_dataset_enum", return_value="demo"),
                mock.patch.object(experiment, "load", return_value=_dataframe()),
                mock.patch.object(
                    experiment,
                    "process_shift_seed",
                    side_effect=fake_process_shift_seed,
                ),
            ):
                experiment.run_experiment(
                    cfg=_cfg(meta_seeds=1),
                    datasets=["demo"],
                    seeds=[1],
                    severities=[0.0, 1.0],
                    approaches_to_run=[
                        "empirical",
                        "empirical_randomized",
                        "probabilistic",
                        "empirical_weighted",
                    ],
                    output_dir=output_dir,
                    force=False,
                    jobs=1,
                )

        self.assertEqual(
            captured,
            [
                (
                    0.0,
                    [
                        "empirical",
                        "empirical_randomized",
                        "probabilistic",
                        "empirical_weighted",
                    ],
                ),
                (1.0, ["empirical_weighted"]),
            ],
        )

    def test_covariate_shift_skips_nonzero_severity_when_only_unweighted_methods_apply(self):
        process_shift_seed = mock.Mock()

        def write_selection_csv(cfg, datasets, seeds, output_dir, jobs, logger):
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "seed": 1,
                        "dataset": datasets[0],
                        "model": "linear",
                        "fold": "mean",
                        "prauc": 1.0,
                        "rocauc": 1.0,
                        "brier": 0.0,
                        "is_best": True,
                    }
                ]
            ).to_csv(output_dir / f"{datasets[0]}.csv", index=False)

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            output_dir = repo_root / "outputs" / "experiment_results"

            with (
                mock.patch.object(experiment, "REPO_ROOT", repo_root),
                mock.patch.object(
                    experiment,
                    "run_model_selection",
                    side_effect=write_selection_csv,
                ),
                mock.patch.object(experiment, "get_dataset_enum", return_value="demo"),
                mock.patch.object(experiment, "load", return_value=_dataframe()),
                mock.patch.object(
                    experiment,
                    "process_shift_seed",
                    side_effect=process_shift_seed,
                ),
            ):
                experiment.run_experiment(
                    cfg=_cfg(meta_seeds=1),
                    datasets=["demo"],
                    seeds=[1],
                    severities=[1.0],
                    approaches_to_run=["empirical"],
                    output_dir=output_dir,
                    force=False,
                    jobs=1,
                )

            process_shift_seed.assert_not_called()
            self.assertFalse((output_dir / "demo.csv").exists())

    def test_covariate_shift_config_snapshot_preserves_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.toml"
            output_dir = tmp_path / "outputs"
            config_path.write_text("version = 1\n", encoding="utf-8")

            experiment.run_experiment(
                cfg=_cfg(),
                datasets=[],
                seeds=[1],
                severities=[0.0],
                approaches_to_run=["empirical_weighted"],
                output_dir=output_dir,
                force=False,
                jobs=1,
                config_path=config_path,
            )
            self.assertEqual((output_dir / "config.toml").read_text(), "version = 1\n")

            config_path.write_text("version = 2\n", encoding="utf-8")
            experiment.run_experiment(
                cfg=_cfg(),
                datasets=[],
                seeds=[1],
                severities=[0.0],
                approaches_to_run=["empirical_weighted"],
                output_dir=output_dir,
                force=False,
                jobs=1,
                config_path=config_path,
            )
            self.assertEqual((output_dir / "config.toml").read_text(), "version = 1\n")

            experiment.run_experiment(
                cfg=_cfg(),
                datasets=[],
                seeds=[1],
                severities=[0.0],
                approaches_to_run=["empirical_weighted"],
                output_dir=output_dir,
                force=True,
                jobs=1,
                config_path=config_path,
            )
            self.assertEqual((output_dir / "config.toml").read_text(), "version = 2\n")


if __name__ == "__main__":
    unittest.main()
