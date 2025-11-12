"""Model selection for PyOD anomaly detection models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import logging

from pyod.models.cblof import CBLOF
from pyod.models.mcd import MCD
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.inne import INNE
from pyod.models.copod import COPOD
from pyod.models.gmm import GMM
from pyod.models.loda import LODA
from pyod.models.cd import CD

from nonconform.utils.data import Dataset
from source.model_selection import run_model_selection, get_best_models

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    datasets = [
        Dataset.LYMPHOGRAPHY,
        Dataset.IONOSPHERE,
        Dataset.WBC,
        Dataset.CARDIO,
        Dataset.MUSK,
        Dataset.THYROID,
        Dataset.MAMMOGRAPHY,
        Dataset.SHUTTLE,
    ]

    model_classes = [IForest, ECOD, HBOS, COPOD, LODA, INNE, MCD]

    print("=" * 80)
    print(f"Datasets: {len(datasets)} | Models: {len(model_classes)} | Runs per pair: 25")
    print("=" * 80)

    results = run_model_selection(
        datasets=datasets,
        model_classes=model_classes,
        n_runs=25,
        study_name="exp0_model_selection",
    )

    best_models = get_best_models(results)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for dataset_name, models in results.items():
        print(f"\n{dataset_name}:")
        sorted_models = sorted(
            models.items(),
            key=lambda x: x[1]["mean_prauc"] if x[1]["mean_prauc"] is not None else -1,
            reverse=True,
        )

        for rank, (model_name, r) in enumerate(sorted_models, 1):
            marker = "★" if model_name == best_models.get(dataset_name) else " "
            if r.get("failed") and r["n_runs"] == 0:
                print(f"{marker} {rank}. {model_name:10s} | FAILED: {r['error']}")
            else:
                print(
                    f"{marker} {rank}. {model_name:10s} | "
                    f"PRAUC: {r['mean_prauc']:.4f}±{r['std_prauc']:.4f} | "
                    f"AUROC: {r['mean_auroc']:.4f}±{r['std_auroc']:.4f} | "
                    f"Brier: {r['mean_brier']:.4f}±{r['std_brier']:.4f}"
                )

    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "exp0.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "mean_prauc",
                "std_prauc",
                "mean_auroc",
                "std_auroc",
                "mean_brier",
                "std_brier",
                "n_runs",
                "failed",
                "error",
                "is_best",
            ],
        )
        writer.writeheader()

        for dataset_name, models in results.items():
            for model_name, r in models.items():
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "mean_prauc": "" if r["mean_prauc"] is None else f"{r['mean_prauc']:.6f}",
                        "std_prauc": "" if r["std_prauc"] is None else f"{r['std_prauc']:.6f}",
                        "mean_auroc": "" if r["mean_auroc"] is None else f"{r['mean_auroc']:.6f}",
                        "std_auroc": "" if r["std_auroc"] is None else f"{r['std_auroc']:.6f}",
                        "mean_brier": "" if r["mean_brier"] is None else f"{r['mean_brier']:.6f}",
                        "std_brier": "" if r["std_brier"] is None else f"{r['std_brier']:.6f}",
                        "n_runs": r["n_runs"],
                        "failed": "TRUE" if r.get("failed", False) else "FALSE",
                        "error": r.get("error", ""),
                        "is_best": "TRUE" if model_name == best_models.get(dataset_name) else "FALSE",
                    }
                )

    print(f"\nResults saved: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
