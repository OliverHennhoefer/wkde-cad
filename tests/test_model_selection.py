import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import src.experiment as experiment
import src.model_selection as model_selection
from src.utils.splits import (
    ModelSelectionEvaluationSplit,
    split_model_selection_evaluation,
)


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
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, x_train):
        return self

    def decision_function(self, x_test):
        return x_test[:, 0] + 0.1 * x_test[:, 1]


class FailingDetector(DeterministicDetector):
    def fit(self, x_train):
        raise ValueError("fit failed")


class SharedModelSelectionTest(unittest.TestCase):
    def test_model_selection_evaluation_split_is_disjoint_and_deterministic(self):
        data = _dataframe()
        normal = data.query("Class == 0")
        anomaly = data.query("Class == 1")

        split1 = split_model_selection_evaluation(
            normal,
            anomaly,
            train_split=0.5,
            seed=11,
        )
        split2 = split_model_selection_evaluation(
            normal,
            anomaly,
            train_split=0.5,
            seed=11,
        )

        self.assertTrue(
            split1.normal_model_selection.index.intersection(
                split1.normal_evaluation.index
            ).empty
        )
        self.assertTrue(
            split1.anomaly_model_selection.index.intersection(
                split1.anomaly_evaluation.index
            ).empty
        )
        self.assertEqual(
            set(split1.normal_model_selection.index)
            | set(split1.normal_evaluation.index),
            set(normal.index),
        )
        self.assertEqual(
            set(split1.anomaly_model_selection.index)
            | set(split1.anomaly_evaluation.index),
            set(anomaly.index),
        )
        pd.testing.assert_frame_equal(
            split1.normal_model_selection,
            split2.normal_model_selection,
        )
        pd.testing.assert_frame_equal(
            split1.anomaly_evaluation,
            split2.anomaly_evaluation,
        )

    def test_existing_selection_csv_skips_without_loading_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "seed": 1,
                        "dataset": "demo",
                        "model": "linear",
                        "fold": "mean",
                        "prauc": 1.0,
                        "rocauc": 1.0,
                        "brier": 0.0,
                        "is_best": True,
                    }
                ]
            ).to_csv(output_dir / "demo.csv", index=False)
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

    def test_incomplete_selection_csv_reruns_when_configured_models_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "seed": 1,
                        "dataset": "demo",
                        "model": "linear",
                        "fold": "mean",
                        "prauc": 1.0,
                        "rocauc": 1.0,
                        "brier": 0.0,
                        "is_best": True,
                    }
                ]
            ).to_csv(output_dir / "demo.csv", index=False)

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
                            "model": model,
                            "fold": "mean",
                            "prauc": 1.0,
                            "rocauc": 1.0,
                            "brier": 0.0,
                            "is_best": model == "linear",
                        }
                        for model in models
                    ]
                ).to_csv(output_dir / f"{ds_name}_seed{seed}.csv", index=False)
                return ds_name, seed, "linear"

            cfg = _cfg()
            cfg["model_selection"]["models"] = ["linear", "hbos"]
            with (
                mock.patch("src.model_selection.get_dataset_enum", return_value="demo"),
                mock.patch("src.model_selection.load", return_value=_dataframe()),
                mock.patch(
                    "src.model_selection.process_seed_phase1",
                    side_effect=write_seed_csv,
                ) as process_seed_phase1,
            ):
                model_selection.run_model_selection(
                    cfg=cfg,
                    datasets=["demo"],
                    seeds=[1],
                    output_dir=output_dir,
                    jobs=1,
                    logger=mock.Mock(),
                )

            process_seed_phase1.assert_called_once()
            merged = pd.read_csv(output_dir / "demo.csv")
            self.assertEqual(set(merged["model"]), {"linear", "hbos"})

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

    def test_model_selection_uses_only_model_selection_pools(self):
        data = _dataframe()
        normal = data.query("Class == 0")
        anomaly = data.query("Class == 1")
        captured = {}

        def fake_split(normal_arg, anomaly_arg, *, train_split, seed):
            captured["train_split"] = train_split
            captured["seed"] = seed
            return ModelSelectionEvaluationSplit(
                normal_model_selection=normal.iloc[:12].copy(),
                normal_evaluation=pd.DataFrame(
                    {"x1": ["bad"], "x2": ["bad"], "Class": [0]},
                    index=[999],
                ),
                anomaly_model_selection=anomaly.iloc[:6].copy(),
                anomaly_evaluation=pd.DataFrame(
                    {"x1": ["bad"], "x2": ["bad"], "Class": [1]},
                    index=[1000],
                ),
            )

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(
                    model_selection,
                    "split_model_selection_evaluation",
                    side_effect=fake_split,
                ),
                mock.patch(
                    "src.model_selection.get_model_instance",
                    return_value=DeterministicDetector(),
                ),
            ):
                result = model_selection.process_seed_phase1(
                    seed=5,
                    ds_name="demo",
                    normal=normal,
                    anomaly=anomaly,
                    empirical_anomaly_rate=10 / 34,
                    models=["linear"],
                    cfg=_cfg(),
                    output_dir=Path(tmp),
                )

        self.assertEqual(result, ("demo", 5, "linear"))
        self.assertEqual(captured, {"train_split": 0.75, "seed": 5})

    def test_model_selection_skips_failed_candidate_and_selects_valid_model(self):
        def model_factory(model_name, random_state=None):
            if model_name == "bad":
                return FailingDetector()
            return DeterministicDetector()

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "src.model_selection.get_model_instance",
                side_effect=model_factory,
            ):
                result = model_selection.process_seed_phase1(
                    seed=5,
                    ds_name="demo",
                    normal=_dataframe().query("Class == 0"),
                    anomaly=_dataframe().query("Class == 1"),
                    empirical_anomaly_rate=10 / 34,
                    models=["bad", "linear"],
                    cfg=_cfg(),
                    output_dir=Path(tmp),
                )

            self.assertEqual(result, ("demo", 5, "linear"))
            selection = pd.read_csv(Path(tmp) / "demo_seed5.csv")

        failed = selection[selection["model"] == "bad"].iloc[0]
        selected = selection[selection["model"] == "linear"].iloc[0]
        self.assertFalse(bool(failed["is_best"]))
        self.assertIn("fit failed", failed["fit_error"])
        self.assertTrue(bool(selected["is_best"]))
        self.assertTrue(pd.isna(selected["fit_error"]))

    def test_selection_worker_is_deterministic_for_same_seed_and_inputs(self):
        with (
            tempfile.TemporaryDirectory() as tmp1,
            tempfile.TemporaryDirectory() as tmp2,
        ):
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
            split_audit,
        ):
            calls[-1]["normal_indices"] = list(normal.index)
            calls[-1]["anomaly_indices"] = list(anomaly.index)
            calls[-1]["split_audit"] = dict(split_audit)
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

            self.assertEqual(
                {
                    "datasets": calls[0]["datasets"],
                    "seeds": calls[0]["seeds"],
                    "jobs": calls[0]["jobs"],
                },
                {"datasets": ["demo"], "seeds": [1, 2, 3], "jobs": 1},
            )
            results = pd.read_csv(output_dir / "demo.csv")
            self.assertEqual(results["seed"].tolist(), [1])
            expected_split = split_model_selection_evaluation(
                _dataframe().query("Class == 0"),
                _dataframe().query("Class == 1"),
                train_split=0.75,
                seed=1,
            )
            self.assertEqual(
                calls[0]["normal_indices"],
                list(expected_split.normal_evaluation.index),
            )
            self.assertEqual(
                calls[0]["anomaly_indices"],
                list(expected_split.anomaly_evaluation.index),
            )
            self.assertTrue(
                set(calls[0]["normal_indices"]).isdisjoint(
                    expected_split.normal_model_selection.index
                )
            )
            self.assertTrue(
                set(calls[0]["anomaly_indices"]).isdisjoint(
                    expected_split.anomaly_model_selection.index
                )
            )
            self.assertEqual(
                calls[0]["split_audit"],
                {
                    "n_normal_model_selection_excluded": len(
                        expected_split.normal_model_selection
                    ),
                    "n_anomaly_model_selection_excluded": len(
                        expected_split.anomaly_model_selection
                    ),
                    "n_normal_evaluation_pool": len(
                        expected_split.normal_evaluation
                    ),
                    "n_anomaly_evaluation_pool": len(
                        expected_split.anomaly_evaluation
                    ),
                },
            )

    def test_process_shift_seed_records_strict_split_audit_columns(self):
        cfg = _cfg()
        cfg["splits"]["train_split"] = 0.5
        cfg["covariate_shift"]["propensity_min"] = 0.2
        cfg["covariate_shift"]["propensity_max"] = 0.8
        audit = {
            "n_normal_model_selection_excluded": 12,
            "n_anomaly_model_selection_excluded": 5,
            "n_normal_evaluation_pool": 12,
            "n_anomaly_evaluation_pool": 5,
        }

        with mock.patch.object(
            experiment,
            "get_model_instance",
            return_value=DeterministicDetector(),
        ):
            rows = experiment.process_shift_seed(
                1,
                0.0,
                "linear",
                "demo",
                _dataframe().query("Class == 0"),
                _dataframe().query("Class == 1"),
                cfg,
                ["empirical"],
                experiment._pruning_method("homogeneous"),
                audit,
            )

        self.assertIsNotNone(rows)
        row = rows[0]
        for key, value in audit.items():
            self.assertEqual(row[key], value)

    def test_process_shift_seed_skips_failed_detector_without_crashing(self):
        cfg = _cfg()
        cfg["splits"]["train_split"] = 0.5
        cfg["covariate_shift"]["propensity_min"] = 0.2
        cfg["covariate_shift"]["propensity_max"] = 0.8

        with mock.patch.object(
            experiment,
            "get_model_instance",
            return_value=FailingDetector(),
        ):
            rows = experiment.process_shift_seed(
                1,
                0.0,
                "bad",
                "demo",
                _dataframe().query("Class == 0"),
                _dataframe().query("Class == 1"),
                cfg,
                ["empirical"],
                experiment._pruning_method("homogeneous"),
            )

        self.assertEqual(rows, [])

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
            split_audit,
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

    def test_covariate_shift_skips_nonzero_severity_when_only_unweighted_methods_apply(
        self,
    ):
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
