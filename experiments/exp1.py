"""Conformal FDR experiment: comparing unweighted vs weighted approaches."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

import numpy as np
from pyod.models.base import BaseDetector
from pyod.models.copod import COPOD
from sklearn.preprocessing import StandardScaler
from scipy.stats import false_discovery_control

from nonconform.detection import ConformalDetector
from nonconform.detection.weight import LogisticWeightEstimator
from nonconform.strategy import JackknifeBootstrap
from nonconform.strategy import Empirical, Probabilistic
from nonconform.utils.data import Dataset
from nonconform.utils.func.enums import Aggregation, Pruning
from nonconform.utils.stat import (
    false_discovery_rate,
    statistical_power,
)
from nonconform.utils.stat.weighted_fdr import (
    weighted_bh,
    weighted_false_discovery_control,
)

from source.split import StratifiedSplitter


@dataclass
class ExperimentConfig:
    """Configuration for conformal FDR experiment."""

    dataset: Dataset = Dataset.LYMPHOGRAPHY
    detector_cls: Type[BaseDetector] = COPOD
    detector_params: Dict[str, Any] = field(default_factory=dict)
    num_seeds: int = 20
    alpha: float = 0.1
    n_bootstraps: int = 100
    aggregation: Aggregation = Aggregation.MEDIAN
    train_normal_frac: float = 0.70
    min_anomalies_floor: int = 5
    kde_n_trials: int = 100
    kde_cv_folds: int = -1
    weight_estimator_params: Dict[str, Any] = field(default_factory=dict)
    output_path: str | None = None


def _prepare_data(config: ExperimentConfig, seed: int):
    """Prepare train/test data with stratified splitting and scaling."""
    splitter = StratifiedSplitter(
        dataset=config.dataset,
        min_anomalies_floor=config.min_anomalies_floor,
        train_normal_frac=config.train_normal_frac,
        val_normal_frac=None,
    )
    train_data, _, test_data = splitter.split(seed=seed)

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].astype(int)
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].astype(int)

    if not np.all(y_train == 0):
        raise ValueError("Training data must contain only normal samples.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_test, splitter.get_split_info()


def _calculate_metrics(decisions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Calculate FDR, power, and discovery metrics."""
    decisions = decisions.astype(int)
    n_discoveries = int(np.sum(decisions))
    n_total = len(decisions)

    fdr = false_discovery_rate(labels, decisions)
    power = statistical_power(labels, decisions)
    discovery_rate = n_discoveries / n_total if n_total > 0 else 0.0

    return {
        "fdr": float(fdr),
        "power": float(power),
        "n_discoveries": n_discoveries,
        "discovery_rate": discovery_rate,
    }


def _run_single_seed(config: ExperimentConfig, seed: int) -> Dict[str, Any]:
    """Run experiment for a single seed."""
    seed_start = time.perf_counter()

    X_train, X_test, y_test, split_info = _prepare_data(config, seed)

    base_detector = config.detector_cls(**config.detector_params)
    strategy = JackknifeBootstrap(
        n_bootstraps=config.n_bootstraps,
        aggregation_method=config.aggregation,
    )
    weight_params = dict(config.weight_estimator_params)
    weight_params.setdefault("seed", seed)

    variants = {}

    # 1. Empirical + Unweighted (1 variant)
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        estimation=Empirical(),
        weight_estimator=None,
        seed=seed,
    )
    detector.fit(X_train)
    n_calibration = len(detector._calibration_set)
    p_values = detector.predict(X_test)
    decisions = false_discovery_control(p_values, method='bh') <= config.alpha
    variants['empirical'] = _calculate_metrics(decisions, y_test)

    # 2. Empirical + Weighted (4 variants from 1 run)
    detector = ConformalDetector(
        detector=config.detector_cls(**config.detector_params),
        strategy=JackknifeBootstrap(
            n_bootstraps=config.n_bootstraps,
            aggregation_method=config.aggregation,
        ),
        estimation=Empirical(),
        weight_estimator=LogisticWeightEstimator(**weight_params),
        seed=seed,
    )
    detector.fit(X_train)
    p_values = detector.predict(X_test)
    result = detector.last_result

    decisions = weighted_bh(result, alpha=config.alpha)
    variants['empirical_weighted_bh'] = _calculate_metrics(decisions, y_test)

    decisions = weighted_false_discovery_control(
        result, alpha=config.alpha, pruning=Pruning.DETERMINISTIC, seed=seed
    )
    variants['empirical_wcs_dtm'] = _calculate_metrics(decisions, y_test)

    decisions = weighted_false_discovery_control(
        result, alpha=config.alpha, pruning=Pruning.HOMOGENEOUS, seed=seed
    )
    variants['empirical_wcs_homo'] = _calculate_metrics(decisions, y_test)

    decisions = weighted_false_discovery_control(
        result, alpha=config.alpha, pruning=Pruning.HETEROGENEOUS, seed=seed
    )
    variants['empirical_wcs_hete'] = _calculate_metrics(decisions, y_test)

    # 3. KDE + Unweighted (1 variant)
    detector = ConformalDetector(
        detector=config.detector_cls(**config.detector_params),
        strategy=JackknifeBootstrap(
            n_bootstraps=config.n_bootstraps,
            aggregation_method=config.aggregation,
        ),
        estimation=Probabilistic(
            n_trials=config.kde_n_trials,
            cv_folds=config.kde_cv_folds,
        ),
        weight_estimator=None,
        seed=seed,
    )
    detector.fit(X_train)
    p_values = detector.predict(X_test)
    decisions = false_discovery_control(p_values, method='bh') <= config.alpha
    variants['kde'] = _calculate_metrics(decisions, y_test)

    # 4. KDE + Weighted (4 variants from 1 run)
    detector = ConformalDetector(
        detector=config.detector_cls(**config.detector_params),
        strategy=JackknifeBootstrap(
            n_bootstraps=config.n_bootstraps,
            aggregation_method=config.aggregation,
        ),
        estimation=Probabilistic(
            n_trials=config.kde_n_trials,
            cv_folds=config.kde_cv_folds,
        ),
        weight_estimator=LogisticWeightEstimator(**weight_params),
        seed=seed,
    )
    detector.fit(X_train)
    p_values = detector.predict(X_test)
    result = detector.last_result

    decisions = weighted_bh(result, alpha=config.alpha)
    variants['kde_weighted_bh'] = _calculate_metrics(decisions, y_test)

    decisions = weighted_false_discovery_control(
        result, alpha=config.alpha, pruning=Pruning.DETERMINISTIC, seed=seed
    )
    variants['kde_wcs_dtm'] = _calculate_metrics(decisions, y_test)

    decisions = weighted_false_discovery_control(
        result, alpha=config.alpha, pruning=Pruning.HOMOGENEOUS, seed=seed
    )
    variants['kde_wcs_homo'] = _calculate_metrics(decisions, y_test)

    decisions = weighted_false_discovery_control(
        result, alpha=config.alpha, pruning=Pruning.HETEROGENEOUS, seed=seed
    )
    variants['kde_wcs_hete'] = _calculate_metrics(decisions, y_test)

    seed_runtime = time.perf_counter() - seed_start

    print(f"[Seed {seed:>3}] {config.dataset.name} | {config.detector_cls.__name__} (alpha={config.alpha:.3f})")
    for variant_name, metrics in variants.items():
        print(f"    {variant_name:>20}: FDR={metrics['fdr']:.4f} | Power={metrics['power']:.4f}")

    return {
        "seed": seed,
        "alpha": config.alpha,
        "n_calibration": n_calibration,
        "n_test_samples": len(y_test),
        "n_test_anomalies": int(np.sum(y_test)),
        "variants": variants,
        "split_info": split_info,
        "runtime_seconds": seed_runtime,
    }


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run experiment across multiple seeds."""
    experiment_start = time.perf_counter()

    seeds = list(range(1, config.num_seeds + 1))
    per_seed_results = []

    for seed in seeds:
        per_seed_results.append(_run_single_seed(config, seed))

    total_runtime = time.perf_counter() - experiment_start

    # Aggregate results across seeds
    aggregate_variants = {}
    if per_seed_results:
        variant_names = per_seed_results[0]["variants"].keys()
        for variant in variant_names:
            variant_metrics = [seed["variants"][variant] for seed in per_seed_results]
            fdr_values = [m["fdr"] for m in variant_metrics]
            power_values = [m["power"] for m in variant_metrics]
            discoveries = [m["n_discoveries"] for m in variant_metrics]
            discovery_rates = [m["discovery_rate"] for m in variant_metrics]

            aggregate_variants[variant] = {
                "fdr_mean": float(np.mean(fdr_values)),
                "fdr_std": float(np.std(fdr_values)),
                "power_mean": float(np.mean(power_values)),
                "power_std": float(np.std(power_values)),
                "discoveries_mean": float(np.mean(discoveries)),
                "discoveries_std": float(np.std(discoveries)),
                "discovery_rate_mean": float(np.mean(discovery_rates)),
                "discovery_rate_std": float(np.std(discovery_rates)),
            }

    # Save to CSV
    output_path = Path(config.output_path) if config.output_path else (
        Path(__file__).parent.parent / "results" / f"exp1_{config.dataset.name.lower()}_{config.detector_cls.__name__.lower()}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "model", "seed", "alpha", "n_calibration",
                "n_test_samples", "n_test_anomalies",
                "variant", "fdr", "fdr_std", "power", "power_std",
                "n_discoveries", "discoveries_std",
                "discovery_rate", "discovery_rate_std",
            ],
        )
        writer.writeheader()

        dataset_name = config.dataset.name
        model_name = config.detector_cls.__name__

        # Write per-seed rows
        for seed_result in per_seed_results:
            for variant_name, metrics in seed_result["variants"].items():
                writer.writerow({
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed_result["seed"],
                    "alpha": f"{seed_result['alpha']:.6f}",
                    "n_calibration": seed_result["n_calibration"],
                    "n_test_samples": seed_result["n_test_samples"],
                    "n_test_anomalies": seed_result["n_test_anomalies"],
                    "variant": variant_name,
                    "fdr": f"{metrics['fdr']:.6f}",
                    "fdr_std": "",
                    "power": f"{metrics['power']:.6f}",
                    "power_std": "",
                    "n_discoveries": metrics["n_discoveries"],
                    "discoveries_std": "",
                    "discovery_rate": f"{metrics['discovery_rate']:.6f}",
                    "discovery_rate_std": "",
                })

        # Write aggregate rows
        for variant_name, agg_metrics in aggregate_variants.items():
            writer.writerow({
                "dataset": dataset_name,
                "model": model_name,
                "seed": "MEAN",
                "alpha": f"{config.alpha:.6f}",
                "n_calibration": "",
                "n_test_samples": "",
                "n_test_anomalies": "",
                "variant": variant_name,
                "fdr": f"{agg_metrics['fdr_mean']:.6f}",
                "fdr_std": f"{agg_metrics['fdr_std']:.6f}",
                "power": f"{agg_metrics['power_mean']:.6f}",
                "power_std": f"{agg_metrics['power_std']:.6f}",
                "n_discoveries": f"{agg_metrics['discoveries_mean']:.6f}",
                "discoveries_std": f"{agg_metrics['discoveries_std']:.6f}",
                "discovery_rate": f"{agg_metrics['discovery_rate_mean']:.6f}",
                "discovery_rate_std": f"{agg_metrics['discovery_rate_std']:.6f}",
            })

    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name} | Model: {model_name} | Seeds: {len(per_seed_results)}")
    print(f"Total runtime: {total_runtime:.2f}s | Avg per seed: {total_runtime/len(per_seed_results):.2f}s")
    print(f"\nResults saved: {output_path}")
    print(f"{'='*80}\n")

    for variant_name, agg in aggregate_variants.items():
        print(f"{variant_name:>20}: FDR={agg['fdr_mean']:.4f}�{agg['fdr_std']:.4f} | "
              f"Power={agg['power_mean']:.4f}�{agg['power_std']:.4f} | "
              f"Discoveries={agg['discoveries_mean']:.2f}�{agg['discoveries_std']:.2f}")
    print(f"{'='*80}")

    return {
        "config": config,
        "aggregate": {"variants": aggregate_variants, "n_runs": len(per_seed_results)},
        "per_seed": per_seed_results,
        "runtime": {"total_seconds": total_runtime},
    }


def main():
    config = ExperimentConfig()
    run_experiment(config)


if __name__ == "__main__":
    main()
