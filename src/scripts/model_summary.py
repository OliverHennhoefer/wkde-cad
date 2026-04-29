import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = ["prauc", "rocauc", "brier"]


def load_and_validate_csv(file_path: Path) -> pd.DataFrame:
    """Load CSV and validate expected columns."""
    expected_cols = [
        "seed",
        "dataset",
        "model",
        "fold",
        "prauc",
        "rocauc",
        "brier",
        "is_best",
    ]

    df = pd.read_csv(file_path)

    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}")

    for col in METRIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(
                f"Found non-numeric values in column '{col}' in {file_path}"
            )

    if df["is_best"].dtype != bool:
        df["is_best"] = df["is_best"].astype(str).str.lower().eq("true")

    return df


def compute_model_summary(df: pd.DataFrame, metric: str = "rocauc") -> pd.DataFrame:
    """Compute summary statistics per model."""
    summary = (
        df.groupby("model")
        .agg(
            rocauc_mean=("rocauc", "mean"),
            rocauc_std=("rocauc", "std"),
            prauc_mean=("prauc", "mean"),
            prauc_std=("prauc", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            wins=("is_best", "sum"),
            total=("is_best", "count"),
        )
        .reset_index()
    )

    ascending = metric == "brier"
    sort_col = f"{metric}_mean"
    summary = summary.sort_values(
        [sort_col, "model"],
        ascending=[ascending, True],
        kind="mergesort",
    )

    return summary


def _normalize_std(std: float) -> float:
    return 0.0 if pd.isna(std) else float(std)


def format_metric(mean: float, std: float, precision: int = 4) -> str:
    """Format metric as mean +/- std."""
    return f"{mean:.{precision}f} +/- {_normalize_std(std):.{precision}f}"


def print_summary_table(summary: pd.DataFrame, dataset_name: str) -> None:
    """Print formatted summary table."""
    print(f"\nDataset: {dataset_name}")
    print("=" * 80)

    # Header
    print(
        f"{'Model':<10} {'PR-AUC (mean +/- std)':<24} {'ROC-AUC (mean +/- std)':<24} {'Brier':<18} {'Wins'}"
    )
    print("-" * 80)

    # Rows
    for _, row in summary.iterrows():
        prauc = format_metric(row["prauc_mean"], row["prauc_std"])
        rocauc = format_metric(row["rocauc_mean"], row["rocauc_std"])
        brier = format_metric(row["brier_mean"], row["brier_std"])
        wins = f"{int(row['wins'])}/{int(row['total'])}"

        print(f"{row['model']:<10} {prauc:<24} {rocauc:<24} {brier:<18} {wins}")

    print()


def print_csv_output(summary: pd.DataFrame, dataset_name: str) -> None:
    """Print summary in CSV format."""
    print(
        "model,rocauc_mean,rocauc_std,prauc_mean,prauc_std,brier_mean,brier_std,wins,total"
    )
    for _, row in summary.iterrows():
        rocauc_std = _normalize_std(row["rocauc_std"])
        prauc_std = _normalize_std(row["prauc_std"])
        brier_std = _normalize_std(row["brier_std"])
        print(
            f"{row['model']},{row['rocauc_mean']:.4f},{rocauc_std:.4f},"
            f"{row['prauc_mean']:.4f},{prauc_std:.4f},"
            f"{row['brier_mean']:.4f},{brier_std:.4f},"
            f"{int(row['wins'])},{int(row['total'])}"
        )


def process_file(file_path: Path, metric: str, output_format: str) -> None:
    """Process a single CSV file."""
    df = load_and_validate_csv(file_path)
    dataset_name = df["dataset"].iloc[0] if "dataset" in df.columns else file_path.stem

    summary = compute_model_summary(df, metric)

    if output_format == "csv":
        print_csv_output(summary, dataset_name)
    else:
        print_summary_table(summary, dataset_name)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize model selection results from CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.scripts.model_summary outputs/model_selection/breastw.csv
  python -m src.scripts.model_summary outputs/model_selection/*.csv --metric prauc
  python -m src.scripts.model_summary outputs/model_selection/breastw.csv --format csv
        """,
    )

    parser.add_argument("files", nargs="+", type=Path, help="CSV file(s) to analyze")

    parser.add_argument(
        "--metric",
        choices=["prauc", "rocauc", "brier"],
        default="prauc",
        help="Metric to sort by (default: prauc)",
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
            process_file(file_path, args.metric, args.output_format)
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
