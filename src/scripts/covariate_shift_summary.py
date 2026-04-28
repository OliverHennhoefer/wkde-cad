import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


APPROACH_ORDER = [
    "empirical",
    "empirical_randomized",
    "empirical_weighted",
    "empirical_randomized_weighted",
    "probabilistic",
    "probabilistic_weighted",
]
APPROACH_ORDER_MAP = {name: idx for idx, name in enumerate(APPROACH_ORDER)}


def load_and_validate_csv(file_path: Path) -> pd.DataFrame:
    """Load a covariate-shift experiment result CSV."""
    expected_cols = [
        "seed",
        "dataset",
        "approach",
        "severity",
        "weight_mode",
        "fdr",
        "power",
        "n_train",
        "n_test",
        "propensity_std",
        "propensity_min_observed",
        "propensity_max_observed",
        "normal_test_assignment_rate",
        "oracle_calib_weight_max",
        "oracle_calib_weight_ess",
        "oracle_test_weight_max",
        "oracle_test_weight_ess",
    ]

    df = pd.read_csv(file_path)
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {sorted(missing)}")

    df = df[df["seed"] != "mean"].copy()
    numeric_cols = [
        "seed",
        "severity",
        "fdr",
        "power",
        "n_train",
        "n_test",
        "propensity_std",
        "propensity_min_observed",
        "propensity_max_observed",
        "normal_test_assignment_rate",
        "oracle_calib_weight_max",
        "oracle_calib_weight_ess",
        "oracle_test_weight_max",
        "oracle_test_weight_ess",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["seed", "severity", "fdr", "power"])


def _normalize_std(value: float) -> float:
    return 0.0 if pd.isna(value) else float(value)


def format_metric(mean: float, std: float, precision: int = 3) -> str:
    return f"{mean:.{precision}f} +/- {_normalize_std(std):.{precision}f}"


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance and shift diagnostics by severity."""
    summary = (
        df.groupby(["dataset", "severity", "weight_mode", "approach"])
        .agg(
            fdr_mean=("fdr", "mean"),
            fdr_std=("fdr", "std"),
            power_mean=("power", "mean"),
            power_std=("power", "std"),
            n_runs=("fdr", "count"),
            n_train_mean=("n_train", "mean"),
            n_test_mean=("n_test", "mean"),
            propensity_std_mean=("propensity_std", "mean"),
            propensity_min_mean=("propensity_min_observed", "mean"),
            propensity_max_mean=("propensity_max_observed", "mean"),
            assignment_rate_mean=("normal_test_assignment_rate", "mean"),
            oracle_calib_weight_max_mean=("oracle_calib_weight_max", "mean"),
            oracle_test_weight_max_mean=("oracle_test_weight_max", "mean"),
            oracle_calib_weight_ess_mean=("oracle_calib_weight_ess", "mean"),
            oracle_test_weight_ess_mean=("oracle_test_weight_ess", "mean"),
        )
        .reset_index()
    )
    summary["_approach_order"] = (
        summary["approach"].map(APPROACH_ORDER_MAP).fillna(len(APPROACH_ORDER))
    )
    return summary.sort_values(
        ["dataset", "weight_mode", "severity", "_approach_order"],
        kind="mergesort",
    ).drop(columns=["_approach_order"])


def print_table(summary: pd.DataFrame) -> None:
    """Print dataset/severity blocks with approach rows."""
    for (dataset, weight_mode, severity), block in summary.groupby(
        ["dataset", "weight_mode", "severity"],
        sort=False,
    ):
        print(f"\nDataset: {dataset} | severity: {severity:g} | weights: {weight_mode}")
        print("=" * 118)
        headers = [
            "Approach",
            "FDR",
            "Power",
            "N",
            "Prop SD",
            "Prop range",
            "Assign",
            "Max w",
            "ESS calib/test",
        ]
        widths = [31, 17, 17, 5, 8, 15, 8, 8, 16]
        print(" ".join(f"{header:<{width}}" for header, width in zip(headers, widths)))
        print("-" * 118)

        for _, row in block.iterrows():
            fdr = format_metric(row["fdr_mean"], row["fdr_std"])
            power = format_metric(row["power_mean"], row["power_std"])
            prop_range = f"{row['propensity_min_mean']:.3f}-{row['propensity_max_mean']:.3f}"
            max_weight = max(
                row["oracle_calib_weight_max_mean"],
                row["oracle_test_weight_max_mean"],
            )
            ess = (
                f"{row['oracle_calib_weight_ess_mean']:.1f}/"
                f"{row['oracle_test_weight_ess_mean']:.1f}"
            )
            values = [
                row["approach"],
                fdr,
                power,
                str(int(row["n_runs"])),
                f"{row['propensity_std_mean']:.3f}",
                prop_range,
                f"{row['assignment_rate_mean']:.3f}",
                f"{max_weight:.1f}",
                ess,
            ]
            print(" ".join(f"{value:<{width}}" for value, width in zip(values, widths)))
        print()


def print_csv(summary: pd.DataFrame) -> None:
    print(summary.to_csv(index=False), end="")


def expand_files(files: list[Path]) -> list[Path]:
    expanded = []
    for file_path in files:
        path_str = str(file_path)
        if "*" in path_str or "?" in path_str:
            matches = glob.glob(path_str)
            if matches:
                expanded.extend(Path(match) for match in matches)
            else:
                print(f"Error: No files match pattern: {path_str}", file=sys.stderr)
        else:
            expanded.append(file_path)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize covariate-shift experiment results by severity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.scripts.covariate_shift_summary outputs/experiment_results/wbc.csv
  python -m src.scripts.covariate_shift_summary outputs/experiment_results/*.csv
  python -m src.scripts.covariate_shift_summary outputs/experiment_results/*.csv --format csv
        """,
    )
    parser.add_argument("files", nargs="+", type=Path, help="CSV file(s) to summarize")
    parser.add_argument(
        "--format",
        choices=["table", "csv"],
        default="table",
        dest="output_format",
        help="Output format (default: table)",
    )

    args = parser.parse_args()
    frames = []
    for file_path in expand_files(args.files):
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            continue
        try:
            frames.append(load_and_validate_csv(file_path))
        except Exception as exc:
            print(f"Error processing {file_path}: {exc}", file=sys.stderr)

    if not frames:
        raise SystemExit(1)

    summary = compute_summary(pd.concat(frames, ignore_index=True))
    if args.output_format == "csv":
        print_csv(summary)
    else:
        print_table(summary)


if __name__ == "__main__":
    main()
