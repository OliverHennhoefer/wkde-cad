import argparse
import glob
import math
import sys
from pathlib import Path

import pandas as pd

DATASET_ORDER = [
    ("wbc", "WBC"),
    ("ionosphere", "Ionosphere"),
    ("wdbc", "WDBC"),
    ("breastw", "BreastCa"),
    ("vowels", "Vowels"),
    ("cardio", "Cardio"),
    ("satellite", "Satellite"),
    ("mammography", "Mammogr."),
]

PLOT_SPECS = [
    (
        "W. EDF (det.)",
        "edfdet",
        "W.\\ EDF (deterministic)",
        "deterministic",
        "empirical_randomized_weighted",
    ),
    (
        "W. EDF (homog.)",
        "edfhomog",
        "W.\\ EDF (homogeneous)",
        "homogeneous",
        "empirical_randomized_weighted",
    ),
    (
        "W. EDF (het.)",
        "edfhet",
        "W.\\ EDF (heterogeneous)",
        "heterogeneous",
        "empirical_randomized_weighted",
    ),
    (
        "W. KDE (Ours)",
        "kdestd",
        "W.\\ KDE (Ours)",
        "deterministic",
        "probabilistic_weighted",
    ),
]


def load_and_validate_csv(file_path: Path) -> pd.DataFrame:
    expected_cols = ["seed", "dataset", "approach", "power"]
    df = pd.read_csv(file_path)

    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}")

    # Ignore appended summary rows.
    df = df[df["seed"].astype(str) != "mean"].copy()
    df["power"] = pd.to_numeric(df["power"], errors="coerce")

    if df["power"].isna().any():
        raise ValueError(f"Found NaN/non-numeric power values in {file_path}")

    return df


def expand_csv_files(pattern: str) -> list[Path]:
    matches = glob.glob(pattern)
    return [Path(match) for match in matches]


def sem_from_series(series: pd.Series) -> float:
    sample_std = float(series.std(ddof=1))
    return sample_std / math.sqrt(len(series))


def compute_dataset_metric(
    results_root: Path,
    condition: str,
    dataset_key: str,
    approach: str,
    expected_trials: int,
) -> tuple[float, float]:
    file_path = results_root / condition / f"{dataset_key}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {file_path}")

    df = load_and_validate_csv(file_path)
    subset = df[df["approach"] == approach].copy()
    if subset.empty:
        raise ValueError(
            f"No rows found for approach '{approach}' in {file_path}"
        )

    n_rows = len(subset)
    if n_rows != expected_trials:
        raise ValueError(
            f"Expected {expected_trials} trials for {dataset_key}/{condition}/{approach}, got {n_rows}"
        )

    power_mean = float(subset["power"].mean())
    power_sem = sem_from_series(subset["power"])
    return power_mean, power_sem


def print_plot_block(
    comment_label: str,
    style_key: str,
    legend_label: str,
    condition: str,
    approach: str,
    results_root: Path,
    trials: int,
    mean_precision: int,
    sem_precision: int,
) -> None:
    print(f"% {comment_label}")
    print(r"\addplot+[")
    print(f"  {style_key},")
    print(r"  error bars/.cd,")
    print(r"    y dir=both,")
    print(r"    y explicit,")
    print(r"    error bar style={line width=0.8pt, black},")
    print(r"    error mark options={rotate=90, mark size=2.2pt, line width=0.8pt, black},")
    print(r"] coordinates {")

    for dataset_key, dataset_label in DATASET_ORDER:
        mean, sem = compute_dataset_metric(
            results_root=results_root,
            condition=condition,
            dataset_key=dataset_key,
            approach=approach,
            expected_trials=trials,
        )
        mean_s = f"{mean:.{mean_precision}f}"
        sem_s = f"{sem:.{sem_precision}f}"
        print(f"    ({dataset_label},{mean_s}) +- (0,{sem_s})")

    print(r"};")
    print(rf"\addlegendentry{{{legend_label}}}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pruning summary PGFPlots blocks (power mean +- SEM).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing deterministic/homogeneous/heterogeneous CSV folders.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Expected number of trials per dataset/approach (default: 20).",
    )
    parser.add_argument(
        "--mean-precision",
        type=int,
        default=3,
        help="Decimal precision for mean power values (default: 3).",
    )
    parser.add_argument(
        "--sem-precision",
        type=int,
        default=4,
        help="Decimal precision for SEM values (default: 4).",
    )
    args = parser.parse_args()

    if args.trials < 2:
        parser.error("--trials must be >= 2 for sample standard deviation (ddof=1).")

    root_pattern = str(args.results_root / "*" / "*.csv")
    if not expand_csv_files(root_pattern):
        parser.error(f"No CSV files found under {args.results_root}")

    try:
        for idx, (comment, style_key, legend, condition, approach) in enumerate(PLOT_SPECS):
            if idx > 0:
                print()
            print_plot_block(
                comment_label=comment,
                style_key=style_key,
                legend_label=legend,
                condition=condition,
                approach=approach,
                results_root=args.results_root,
                trials=args.trials,
                mean_precision=args.mean_precision,
                sem_precision=args.sem_precision,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
