import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


def load_and_validate_csv(file_path: Path) -> pd.DataFrame:
    """Load CSV and validate expected columns."""
    expected_cols = [
        "seed",
        "dataset",
        "model",
        "approach",
        "train_size",
        "test_size",
        "fdr",
        "power",
    ]

    df = pd.read_csv(file_path)

    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}")

    # Filter out summary rows (where seed is "mean")
    df = df[df["seed"] != "mean"].copy()

    # Convert seed to int for proper handling
    df["seed"] = df["seed"].astype(int)

    # Ensure fdr and power are numeric (in case they were read as strings)
    df["fdr"] = pd.to_numeric(df["fdr"], errors="coerce")
    df["power"] = pd.to_numeric(df["power"], errors="coerce")

    return df


def compute_experiment_summary(
    df: pd.DataFrame, group_by: list[str]
) -> pd.DataFrame:
    """Compute summary statistics grouped by specified columns.

    Args:
        df: DataFrame with experiment results
        group_by: List of columns to group by (e.g., ["approach"], ["approach", "train_size"])

    Returns:
        DataFrame with aggregated statistics
    """
    # Ensure required grouping columns exist
    for col in group_by:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data")

    # Group and aggregate
    summary = (
        df.groupby(group_by)
        .agg(
            fdr_mean=("fdr", "mean"),
            fdr_std=("fdr", "std"),
            power_mean=("power", "mean"),
            power_std=("power", "std"),
            n_runs=("fdr", "count"),
        )
        .reset_index()
    )

    return summary


def format_metric(mean: float, std: float, precision: int = 3) -> str:
    """Format metric as mean +/- std."""
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def print_summary_table(
    summary: pd.DataFrame, dataset_name: str, group_by: list[str]
) -> None:
    """Print formatted summary table."""
    print(f"\nDataset: {dataset_name}")
    print(f"Grouped by: {', '.join(group_by)}")
    print("=" * 100)

    # Build header dynamically based on grouping
    header_parts = [f"{col.capitalize():<20}" for col in group_by]
    header_parts.extend(
        ["FDR (mean +/- std)      ", "Power (mean +/- std)    ", "N"]
    )
    print("".join(header_parts))
    print("-" * 100)

    # Rows
    for _, row in summary.iterrows():
        row_parts = [f"{str(row[col]):<20}" for col in group_by]
        fdr = format_metric(row["fdr_mean"], row["fdr_std"])
        power = format_metric(row["power_mean"], row["power_std"])
        n_runs = f"{int(row['n_runs'])}"

        row_parts.extend([f"{fdr:<24}", f"{power:<24}", n_runs])
        print("".join(row_parts))

    print()


def print_csv_output(summary: pd.DataFrame, group_by: list[str]) -> None:
    """Print summary in CSV format."""
    # Build column list
    cols = group_by + ["fdr_mean", "fdr_std", "power_mean", "power_std", "n_runs"]
    print(",".join(cols))

    for _, row in summary.iterrows():
        values = []
        for col in group_by:
            values.append(str(row[col]))
        values.extend(
            [
                f"{row['fdr_mean']:.3f}",
                f"{row['fdr_std']:.3f}",
                f"{row['power_mean']:.3f}",
                f"{row['power_std']:.3f}",
                f"{int(row['n_runs'])}",
            ]
        )
        print(",".join(values))


def process_file(
    file_path: Path, group_by: list[str], output_format: str
) -> None:
    """Process a single CSV file."""
    df = load_and_validate_csv(file_path)
    dataset_name = df["dataset"].iloc[0] if "dataset" in df.columns else file_path.stem

    summary = compute_experiment_summary(df, group_by)

    if output_format == "csv":
        print_csv_output(summary, group_by)
    else:
        print_summary_table(summary, dataset_name, group_by)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize experiment results from CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m code.utils.experiment_summary experiments/results/experiment1/ionosphere.csv
  python -m code.utils.experiment_summary experiments/results/experiment1/*.csv --group-by approach train_size
  python -m code.utils.experiment_summary experiments/results/experiment1/ionosphere.csv --format csv
        """,
    )

    parser.add_argument("files", nargs="+", type=Path, help="CSV file(s) to analyze")

    parser.add_argument(
        "--group-by",
        nargs="+",
        choices=["approach", "train_size", "test_size"],
        default=["approach"],
        help="Columns to group by (default: approach)",
    )

    parser.add_argument(
        "--format",
        choices=["table", "csv"],
        default="table",
        dest="output_format",
        help="Output format (default: table)",
    )

    args = parser.parse_args()

    # Expand glob patterns (needed for PowerShell which doesn't expand them)
    expanded_files = []
    for file_path in args.files:
        path_str = str(file_path)
        if "*" in path_str or "?" in path_str:
            matches = glob.glob(path_str)
            if matches:
                expanded_files.extend(Path(m) for m in matches)
            else:
                print(f"Error: No files match pattern: {path_str}", file=sys.stderr)
        else:
            expanded_files.append(file_path)

    for file_path in expanded_files:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            continue

        try:
            process_file(file_path, args.group_by, args.output_format)
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
