from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import shutil
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nonconform import ConformalDetector, Empirical, JackknifeBootstrap, Probabilistic
from nonconform.fdr import Pruning, weighted_false_discovery_control
from nonconform.metrics import false_discovery_rate, statistical_power
from scipy.stats import false_discovery_control
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.covariate_shift import (
    FixedWeightEstimator,
    fit_propensity_model,
    rejection_sample,
    weight_summary,
)
from src.model_selection import run_model_selection
from src.utils.data_loader import load
from src.utils.logger import get_logger
from src.utils.registry import get_dataset_enum, get_model_instance
from src.utils.splits import split_model_selection_evaluation
from src.utils.weight_estimators import build_weight_estimator


logging.getLogger("nonconform").setLevel(logging.CRITICAL)

N_JOBS = min(mp.cpu_count(), 2) if mp.cpu_count() <= 4 else max(1, mp.cpu_count() - 2)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.toml"
UNWEIGHTED_APPROACHES = {
    "empirical",
    "empirical_randomized",
    "probabilistic",
}
WEIGHTED_APPROACHES = {
    "empirical_weighted",
    "empirical_randomized_weighted",
    "probabilistic_weighted",
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _seed_list(seed_count: int) -> list[int]:
    if (
        not isinstance(seed_count, int)
        or isinstance(seed_count, bool)
        or seed_count < 1
    ):
        raise ValueError("experiment.meta_seeds must be a positive integer count.")
    return list(range(1, seed_count + 1))


def _pruning_method(name: str) -> Pruning:
    pruning_choice = str(name).strip().lower()
    return {
        "deterministic": Pruning.DETERMINISTIC,
        "homogeneous": Pruning.HOMOGENEOUS,
        "heterogeneous": Pruning.HETEROGENEOUS,
    }[pruning_choice]


def _build_estimated_weight_estimator(weight_choice: str, n_bootstraps: int):
    return build_weight_estimator(weight_choice, n_bootstraps)


def _make_weight_estimator(
    weight_mode: str,
    train_weights: np.ndarray,
    test_weights: np.ndarray,
    cfg: dict[str, Any],
):
    if weight_mode == "oracle":
        return FixedWeightEstimator(train_weights, test_weights)
    if weight_mode == "estimated":
        return _build_estimated_weight_estimator(
            cfg["weighting"].get("estimator", "forest"),
            cfg["conformal"]["n_bootstraps"],
        )
    raise ValueError("weighting.mode must be 'oracle' or 'estimated'.")


def _experiment_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    experiment_cfg = cfg.get("experiment")
    if experiment_cfg is None:
        raise ValueError("Missing [experiment] section in config.")
    return experiment_cfg


def _build_approaches(
    *,
    cfg: dict[str, Any],
    weight_mode: str,
    train_weights: np.ndarray,
    test_weights: np.ndarray,
) -> dict[str, dict[str, Any]]:
    n_bootstraps = cfg["conformal"]["n_bootstraps"]
    n_trials = cfg["conformal"]["n_trials"]
    return {
        "empirical": {
            "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
            "estimation": Empirical(tie_break="classical"),
            "weight_estimator": None,
            "weighted": False,
        },
        "empirical_randomized": {
            "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
            "estimation": Empirical(tie_break="randomized"),
            "weight_estimator": None,
            "weighted": False,
        },
        "probabilistic": {
            "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
            "estimation": Probabilistic(n_trials=n_trials),
            "weight_estimator": None,
            "weighted": False,
        },
        "empirical_weighted": {
            "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
            "estimation": Empirical(tie_break="classical"),
            "weight_estimator": _make_weight_estimator(
                weight_mode, train_weights, test_weights, cfg
            ),
            "weighted": True,
        },
        "empirical_randomized_weighted": {
            "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
            "estimation": Empirical(tie_break="randomized"),
            "weight_estimator": _make_weight_estimator(
                weight_mode, train_weights, test_weights, cfg
            ),
            "weighted": True,
        },
        "probabilistic_weighted": {
            "strategy": JackknifeBootstrap(n_bootstraps=n_bootstraps),
            "estimation": Probabilistic(n_trials=n_trials),
            "weight_estimator": _make_weight_estimator(
                weight_mode, train_weights, test_weights, cfg
            ),
            "weighted": True,
        },
    }


def _seeded_uniforms(index: pd.Index, seed: int) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).random(len(index)), index=index)


def _sample_by_priority(
    data: pd.DataFrame,
    n: int,
    priority: pd.Series,
) -> pd.DataFrame:
    if n > len(data):
        raise ValueError("Cannot sample more rows than are available.")
    if n == len(data):
        return data.copy()

    selected_index = (
        priority.loc[data.index].sort_values(kind="mergesort").head(n).index
    )
    return data.loc[selected_index].copy()


def _split_anomaly_candidates(
    anomaly: pd.DataFrame,
    feature_columns: list[str],
    propensity_model,
    train_split: float,
    seed: int,
    assignment_uniforms: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(anomaly) < 2:
        return anomaly.copy(), pd.Series(
            propensity_model.propensity(anomaly[feature_columns]),
            index=anomaly.index,
        )

    sampled = rejection_sample(
        anomaly,
        feature_columns,
        propensity_model,
        seed=seed + 10_000,
        uniforms=assignment_uniforms,
    )
    if len(sampled.accepted) > 0:
        return sampled.accepted, sampled.accepted_propensity

    _, fallback = train_test_split(
        anomaly,
        train_size=train_split,
        random_state=seed,
    )
    fallback_propensity = pd.Series(
        propensity_model.propensity(fallback[feature_columns]),
        index=fallback.index,
        name="test_propensity",
    )
    return fallback.copy(), fallback_propensity


def process_shift_seed(
    seed: int,
    severity: float,
    model_name: str,
    ds_name: str,
    normal: pd.DataFrame,
    anomaly: pd.DataFrame,
    cfg: dict[str, Any],
    approaches_to_run: list[str],
    pruning_method: Pruning,
    split_audit: dict[str, int] | None = None,
) -> list[dict[str, Any]] | None:
    feature_columns = [col for col in normal.columns if col != "Class"]
    shift_cfg = cfg["covariate_shift"]
    train_split = cfg["splits"]["train_split"]
    test_use_proportion = cfg["splits"]["test_use_proportion"]
    test_anomaly_rate = cfg["splits"]["test_anomaly_rate"]
    weight_mode = str(cfg["weighting"].get("mode", "oracle")).strip().lower()

    propensity_model = fit_propensity_model(
        normal[feature_columns],
        train_split=train_split,
        severity=severity,
        propensity_min=float(shift_cfg["propensity_min"]),
        propensity_max=float(shift_cfg["propensity_max"]),
    )
    normal_assignment_uniforms = _seeded_uniforms(normal.index, seed)
    anomaly_assignment_uniforms = _seeded_uniforms(anomaly.index, seed + 10_000)
    normal_test_priority = _seeded_uniforms(normal.index, seed + 20_000)
    anomaly_test_priority = _seeded_uniforms(anomaly.index, seed + 30_000)

    normal_sample = rejection_sample(
        normal,
        feature_columns,
        propensity_model,
        seed=seed,
        uniforms=normal_assignment_uniforms,
    )
    normal_train = normal_sample.rejected
    normal_test = normal_sample.accepted

    anomaly_test, anomaly_test_propensity = _split_anomaly_candidates(
        anomaly,
        feature_columns,
        propensity_model,
        train_split,
        seed,
        assignment_uniforms=anomaly_assignment_uniforms,
    )

    if len(normal_train) < 2 or len(normal_test) < 1 or len(anomaly_test) < 1:
        return None

    total_test_available = len(normal_test) + len(anomaly_test)
    target_test_size = round(test_use_proportion * total_test_available)
    if target_test_size < 2:
        return None

    n_anomalies_test = round(target_test_size * test_anomaly_rate)
    n_normal_test = target_test_size - n_anomalies_test
    if n_anomalies_test < 1:
        n_anomalies_test = 1
        n_normal_test = target_test_size - 1
    if n_anomalies_test > len(anomaly_test):
        n_anomalies_test = len(anomaly_test)
        n_normal_test = target_test_size - n_anomalies_test
    if n_normal_test > len(normal_test):
        n_normal_test = len(normal_test)
    if n_normal_test < 1 or n_anomalies_test < 1:
        return None

    normal_test_sampled = _sample_by_priority(
        normal_test,
        n_normal_test,
        normal_test_priority,
    )
    anomaly_test_sampled = _sample_by_priority(
        anomaly_test,
        n_anomalies_test,
        anomaly_test_priority,
    )

    train_propensity = normal_sample.rejected_propensity.loc[normal_train.index]
    normal_test_propensity = normal_sample.accepted_propensity.loc[
        normal_test_sampled.index
    ]
    sampled_anomaly_propensity = anomaly_test_propensity.loc[anomaly_test_sampled.index]

    train_weights = train_propensity.to_numpy() / (1.0 - train_propensity.to_numpy())
    test_propensity = pd.concat(
        [normal_test_propensity, sampled_anomaly_propensity],
        axis=0,
    )
    test_weights = test_propensity.to_numpy() / (1.0 - test_propensity.to_numpy())

    x_train = normal_train.drop(columns=["Class"]).values
    x_test = (
        pd.concat([normal_test_sampled, anomaly_test_sampled])
        .drop(columns=["Class"])
        .values
    )
    y_test = np.concatenate(
        [
            np.zeros(len(normal_test_sampled)),
            np.ones(len(anomaly_test_sampled)),
        ]
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    final_test_size = len(y_test)
    actual_anomaly_rate = n_anomalies_test / final_test_size

    all_approaches = _build_approaches(
        cfg=cfg,
        weight_mode=weight_mode,
        train_weights=train_weights,
        test_weights=test_weights,
    )
    approaches = {k: v for k, v in all_approaches.items() if k in approaches_to_run}

    propensities = normal_sample.all_propensity.to_numpy()
    base_diagnostics = {
        "severity": severity,
        "weight_mode": weight_mode,
        "n_normal_model_selection_excluded": int(
            (split_audit or {}).get("n_normal_model_selection_excluded", 0)
        ),
        "n_anomaly_model_selection_excluded": int(
            (split_audit or {}).get("n_anomaly_model_selection_excluded", 0)
        ),
        "n_normal_evaluation_pool": int(
            (split_audit or {}).get("n_normal_evaluation_pool", len(normal))
        ),
        "n_anomaly_evaluation_pool": int(
            (split_audit or {}).get("n_anomaly_evaluation_pool", len(anomaly))
        ),
        "target_test_probability": propensity_model.target_probability,
        "propensity_mean": float(np.mean(propensities)),
        "propensity_std": float(np.std(propensities)),
        "propensity_min_observed": float(np.min(propensities)),
        "propensity_max_observed": float(np.max(propensities)),
        "normal_test_assignment_rate": normal_sample.accepted_fraction,
        "target_test_size": target_test_size,
        "n_normal_rejection_pool": len(normal_train),
        "n_normal_accepted_pool": len(normal_test),
        "n_anomaly_accepted_pool": len(anomaly_test),
    }
    base_diagnostics.update(weight_summary("oracle_calib", train_weights))
    base_diagnostics.update(weight_summary("oracle_test", test_weights))

    seed_results = []
    for approach_name, approach_config in approaches.items():
        detector_kwargs = {
            "detector": get_model_instance(model_name, random_state=seed),
            "strategy": approach_config["strategy"],
            "seed": seed,
        }
        if approach_config["estimation"] is not None:
            detector_kwargs["estimation"] = approach_config["estimation"]
        if approach_config["weight_estimator"] is not None:
            detector_kwargs["weight_estimator"] = approach_config["weight_estimator"]

        detector = ConformalDetector(**detector_kwargs)
        detector.fit(x_train_scaled)
        p_values = detector.compute_p_values(x_test_scaled)

        if approach_config["weighted"]:
            decisions = weighted_false_discovery_control(
                detector.last_result,
                alpha=cfg["conformal"]["fdr_rate"],
                pruning=pruning_method,
                seed=seed,
            )
        else:
            decisions = (
                false_discovery_control(p_values, method="bh")
                <= cfg["conformal"]["fdr_rate"]
            )

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        row = {
            "seed": seed,
            "dataset": ds_name,
            "model": model_name,
            "approach": approach_name,
            "train_size": len(x_train_scaled),
            "test_size": final_test_size,
            "n_train": len(x_train_scaled),
            "n_test": final_test_size,
            "n_test_normal": n_normal_test,
            "n_test_anomaly": n_anomalies_test,
            "actual_anomaly_rate": round(actual_anomaly_rate, 3),
            "fdr": round(fdr, 3),
            "power": round(power, 3),
        }
        row.update(base_diagnostics)
        if (
            approach_config["weighted"]
            and detector.last_result.calib_weights is not None
        ):
            row.update(weight_summary("used_calib", detector.last_result.calib_weights))
            row.update(weight_summary("used_test", detector.last_result.test_weights))
        else:
            row.update(weight_summary("used_calib", np.array([])))
            row.update(weight_summary("used_test", np.array([])))
        seed_results.append(row)

    return seed_results


def _valid_approaches() -> set[str]:
    return UNWEIGHTED_APPROACHES | WEIGHTED_APPROACHES


def _approaches_for_severity(
    approaches_to_run: list[str],
    severity: float,
) -> list[str]:
    approaches = list(approaches_to_run)
    if np.isclose(severity, 0.0):
        return approaches
    return [approach for approach in approaches if approach in WEIGHTED_APPROACHES]


def _copy_config_snapshot(
    config_path: Path,
    output_dir: Path,
    force: bool,
    logger,
) -> None:
    snapshot_path = output_dir / "config.toml"
    if snapshot_path.exists() and not force:
        logger.info(f"Preserving existing config snapshot at {snapshot_path}")
        return
    shutil.copy2(config_path, snapshot_path)
    logger.info(f"Saved config snapshot to {snapshot_path}")


def _load_selected_models(model_selection_dir: Path, ds_name: str) -> pd.DataFrame:
    selection_csv = model_selection_dir / f"{ds_name}.csv"
    if not selection_csv.exists():
        raise FileNotFoundError(selection_csv)
    selection_df = pd.read_csv(selection_csv)
    best_mask = selection_df["is_best"].astype(str).str.lower().eq("true")
    return selection_df[best_mask].copy()


def run_experiment(
    *,
    cfg: dict[str, Any],
    datasets: list[str],
    seeds: list[int],
    severities: list[float],
    approaches_to_run: list[str],
    output_dir: Path,
    force: bool,
    jobs: int,
    config_path: Path | None = None,
) -> None:
    logger = get_logger("experiment")
    model_selection_dir = REPO_ROOT / cfg["model_selection"]["output_dir"]
    pruning_method = _pruning_method(cfg["conformal"].get("pruning", "heterogeneous"))

    invalid_approaches = set(approaches_to_run) - _valid_approaches()
    if invalid_approaches:
        raise ValueError(
            f"Invalid approaches in config: {invalid_approaches}. "
            f"Valid options are: {sorted(_valid_approaches())}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        _copy_config_snapshot(config_path, output_dir, force, logger)

    for ds_name in datasets:
        results_csv = output_dir / f"{ds_name}.csv"
        if results_csv.exists() and not force:
            logger.info(f"Skipping {ds_name} (results exist)")
            continue

        selection_seeds = _seed_list(cfg["experiment"]["meta_seeds"])
        run_model_selection(
            cfg=cfg,
            datasets=[ds_name],
            seeds=selection_seeds,
            output_dir=model_selection_dir,
            jobs=jobs,
            logger=logger,
        )

        try:
            best_models = _load_selected_models(model_selection_dir, ds_name)
        except FileNotFoundError:
            logger.warning(f"No model selection results for {ds_name}, skipping")
            continue

        best_models = best_models[best_models["seed"].astype(int).isin(seeds)]
        if best_models.empty:
            logger.warning(
                f"No selected models for requested seeds in {ds_name}, skipping"
            )
            continue

        dataset_enum = get_dataset_enum(ds_name)
        data = load(dataset_enum, setup=False)
        normal = data[data["Class"] == 0]
        anomaly = data[data["Class"] == 1]

        tasks = []
        for _, row in best_models.iterrows():
            seed_value = int(row["seed"])
            split = split_model_selection_evaluation(
                normal,
                anomaly,
                train_split=cfg["splits"]["train_split"],
                seed=seed_value,
            )
            split_audit = {
                "n_normal_model_selection_excluded": len(
                    split.normal_model_selection
                ),
                "n_anomaly_model_selection_excluded": len(
                    split.anomaly_model_selection
                ),
                "n_normal_evaluation_pool": len(split.normal_evaluation),
                "n_anomaly_evaluation_pool": len(split.anomaly_evaluation),
            }
            for severity in severities:
                severity_approaches = _approaches_for_severity(
                    approaches_to_run,
                    float(severity),
                )
                if not severity_approaches:
                    logger.warning(
                        f"Skipping {ds_name} seed {int(row['seed'])} "
                        f"severity={float(severity):g}: no configured methods apply"
                    )
                    continue
                tasks.append(
                    (
                        seed_value,
                        float(severity),
                        row["model"],
                        ds_name,
                        split.normal_evaluation,
                        split.anomaly_evaluation,
                        cfg,
                        severity_approaches,
                        pruning_method,
                        split_audit,
                    )
                )

        if jobs == 1:
            raw_results = [process_shift_seed(*task) for task in tasks]
        else:
            with mp.Pool(jobs) as pool:
                raw_results = pool.starmap(process_shift_seed, tasks)

        rows = [row for result in raw_results if result is not None for row in result]
        if not rows:
            logger.warning(f"No valid shifted results for {ds_name}")
            continue

        df_results = pd.DataFrame(rows)
        df_results = df_results.sort_values(
            ["severity", "seed", "approach"],
            kind="mergesort",
        )
        df_results.to_csv(results_csv, index=False)

        for row in rows:
            logger.info(
                f"{row['dataset']} (seed {row['seed']}, severity={row['severity']}, "
                f"{row['approach']}, train={row['train_size']}, test={row['test_size']}) - "
                f"FDR: {row['fdr']:.3f}, Power: {row['power']:.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run covariate-shift conformal anomaly detection experiments."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--severities", nargs="+", type=float)
    parser.add_argument("--approaches", nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=N_JOBS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    experiment_cfg = _experiment_cfg(cfg)

    datasets = args.datasets or _as_list(experiment_cfg["datasets"])
    seeds = args.seeds or _seed_list(experiment_cfg["meta_seeds"])
    severities = args.severities or [
        float(v) for v in _as_list(experiment_cfg["severities"])
    ]
    approaches_to_run = args.approaches or _as_list(cfg["methods"]["approaches"])
    output_dir = args.output_dir or (REPO_ROOT / experiment_cfg["output_dir"])
    jobs = max(1, int(args.jobs))

    run_experiment(
        cfg=cfg,
        datasets=[str(dataset) for dataset in datasets],
        seeds=[int(seed) for seed in seeds],
        severities=[float(severity) for severity in severities],
        approaches_to_run=[str(approach) for approach in approaches_to_run],
        output_dir=output_dir,
        force=args.force,
        jobs=jobs,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
