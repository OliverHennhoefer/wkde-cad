import argparse
from dataclasses import dataclass
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

LATEX_METHOD_ROWS = [
    ("EDF", "empirical", "empirical_randomized", False),
    ("Weighted EDF", "empirical_weighted", "empirical_randomized_weighted", False),
    ("KDE", "probabilistic", None, True),
    ("Weighted KDE", "probabilistic_weighted", None, True),
]

DATASET_LABEL_MAP = {
    "wbc": "WBC",
    "ionosphere": "Ionosphere",
    "wdbc": "WDBC",
    "breastw": r"\shortstack{Breast Cancer\\(Wisconsin)}",
    "vowels": "Vowels",
    "cardio": "Cardio",
    "musk": "Musk",
    "satellite": "Satellite",
    "mammography": "Mammography",
}


@dataclass
class LatexDatasetResult:
    dataset_key: str
    dataset_label: str
    n_train: int
    n_test: int
    approach_stats: pd.DataFrame


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


def compute_experiment_summary(df: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
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


def sort_summary_by_group(summary: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    """Apply stable sorting, including canonical ordering for approach names."""
    if "approach" not in group_by:
        return summary

    sorted_summary = summary.copy()
    sorted_summary["_approach_order"] = (
        sorted_summary["approach"].map(APPROACH_ORDER_MAP).fillna(len(APPROACH_ORDER))
    )

    sort_cols = []
    for col in group_by:
        if col == "approach":
            sort_cols.append("_approach_order")
        else:
            sort_cols.append(col)

    sorted_summary = sorted_summary.sort_values(sort_cols, kind="mergesort")
    return sorted_summary.drop(columns=["_approach_order"])


def compute_approach_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-approach summary used for LaTeX body output."""
    summary = compute_experiment_summary(df, ["approach"])
    summary = sort_summary_by_group(summary, ["approach"])
    return summary.set_index("approach")


def _normalize_std(std: float) -> float:
    return 0.0 if pd.isna(std) else float(std)


def format_metric(mean: float, std: float, precision: int = 3) -> str:
    """Format metric as mean +/- std."""
    return f"{mean:.{precision}f} +/- {_normalize_std(std):.{precision}f}"


def format_latex_metric(mean: float, std: float, precision: int = 3) -> str:
    """Format metric as LaTeX math value mean\\pmstd."""
    return f"${mean:.{precision}f}\\pm{_normalize_std(std):.{precision}f}$"


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in user-provided text."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _latex_metric_cells(
    approach_stats: pd.DataFrame, approach_key: str | None
) -> tuple[str, str]:
    """Return FDR/Power cells for a given approach key."""
    if not approach_key or approach_key not in approach_stats.index:
        return r"\textemdash", r"\textemdash"

    row = approach_stats.loc[approach_key]
    return (
        format_latex_metric(row["fdr_mean"], row["fdr_std"]),
        format_latex_metric(row["power_mean"], row["power_std"]),
    )


def extract_sample_sizes(df: pd.DataFrame) -> tuple[int, int]:
    """Extract n_train and n_test from first available non-summary row."""
    n_train = None
    n_test = None

    for col in ("n_train", "train_size"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                n_train = int(values.iloc[0])
                break

    for col in ("n_test", "test_size"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                n_test = int(values.iloc[0])
                break

    if n_train is None or n_test is None:
        raise ValueError("Could not extract n_train/n_test values from input CSV")

    return n_train, n_test


def format_int_with_commas(value: int) -> str:
    """Format integer with thousands separators for LaTeX table cells."""
    return f"{value:,}"


def dataset_display_label(dataset_key: str) -> str:
    """Return pretty dataset label for LaTeX output."""
    key = dataset_key.strip().lower()
    if key in DATASET_LABEL_MAP:
        return DATASET_LABEL_MAP[key]

    title = " ".join(part.capitalize() for part in key.replace("_", " ").split())
    return escape_latex(title)


def build_latex_dataset_result(df: pd.DataFrame, dataset_name: str) -> LatexDatasetResult:
    """Build LaTeX-ready summary object for one dataset CSV."""
    n_train, n_test = extract_sample_sizes(df)
    dataset_key = str(dataset_name).strip().lower()
    return LatexDatasetResult(
        dataset_key=dataset_key,
        dataset_label=dataset_display_label(dataset_key),
        n_train=n_train,
        n_test=n_test,
        approach_stats=compute_approach_stats(df),
    )


def print_summary_table(
    summary: pd.DataFrame, dataset_name: str, group_by: list[str]
) -> None:
    """Print formatted summary table."""
    summary = sort_summary_by_group(summary, group_by)

    print(f"\nDataset: {dataset_name}")
    print(f"Grouped by: {', '.join(group_by)}")

    headers = [col.capitalize() for col in group_by] + [
        "FDR (mean +/- std)",
        "Power (mean +/- std)",
        "N",
    ]
    rows = []
    for _, row in summary.iterrows():
        row_parts = [str(row[col]) for col in group_by]
        fdr = format_metric(row["fdr_mean"], row["fdr_std"])
        power = format_metric(row["power_mean"], row["power_std"])
        n_runs = f"{int(row['n_runs'])}"
        row_parts.extend([fdr, power, n_runs])
        rows.append(row_parts)

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    header_line = " | ".join(
        f"{header:<{widths[idx]}}" for idx, header in enumerate(headers)
    )
    separator_line = "-+-".join("-" * width for width in widths)

    print(header_line)
    print(separator_line)
    for row in rows:
        print(" | ".join(f"{value:<{widths[idx]}}" for idx, value in enumerate(row)))

    print()


def print_csv_output(summary: pd.DataFrame, group_by: list[str]) -> None:
    """Print summary in CSV format."""
    summary = sort_summary_by_group(summary, group_by)

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


def build_latex_header(condition_label: str) -> list[str]:
    """Build full LaTeX table header."""
    escaped_condition = escape_latex(condition_label.strip())
    return [
        r"\begin{tabularx}{\textwidth}{l l c c c c p{0.5em} C C}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Method} & \multicolumn{2}{c}{\textbf{Standard}} & \multicolumn{2}{c}{\textbf{Randomized}} & & $n_{\text{train}}$ & $n_{\text{test}}$\\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        rf"& \textit{{{escaped_condition}}} & \textbf{{FDR}} & \textbf{{Power}} & \textbf{{FDR}} & \textbf{{Power}} & & & \\",
        r"\midrule",
    ]


def build_latex_footer() -> list[str]:
    """Build full LaTeX table footer."""
    return [r"\bottomrule", r"\end{tabularx}"]


def build_latex_dataset_block(
    dataset_result: LatexDatasetResult, include_separator: bool
) -> list[str]:
    """Build LaTeX rows for one dataset block."""
    lines: list[str] = []

    for idx, (method_name, standard_key, randomized_key, bold_method) in enumerate(
        LATEX_METHOD_ROWS
    ):
        dataset_cell = (
            rf"\multirow{{4}}{{*}}{{{dataset_result.dataset_label}}}" if idx == 0 else ""
        )
        n_train_cell = (
            rf"\multirow{{4}}{{*}}{{{format_int_with_commas(dataset_result.n_train)}}}"
            if idx == 0
            else ""
        )
        n_test_cell = (
            rf"\multirow{{4}}{{*}}{{{format_int_with_commas(dataset_result.n_test)}}}"
            if idx == 0
            else ""
        )
        method_cell = (
            rf"\textbf{{{escape_latex(method_name)}}}"
            if bold_method
            else escape_latex(method_name)
        )

        std_fdr, std_power = _latex_metric_cells(dataset_result.approach_stats, standard_key)
        if randomized_key is None:
            row_parts = [
                dataset_cell,
                method_cell,
                std_fdr,
                std_power,
                r"\multicolumn{2}{c}{\textemdash}",
                "",
                n_train_cell,
                n_test_cell,
            ]
        else:
            rand_fdr, rand_power = _latex_metric_cells(
                dataset_result.approach_stats, randomized_key
            )
            row_parts = [
                dataset_cell,
                method_cell,
                std_fdr,
                std_power,
                rand_fdr,
                rand_power,
                "",
                n_train_cell,
                n_test_cell,
            ]

        lines.append(" & ".join(row_parts) + r" \\")

    if include_separator:
        lines.append(r"\cmidrule(lr){1-9}")

    return lines


def print_latex_full_table(
    dataset_results: list[LatexDatasetResult], condition_label: str
) -> None:
    """Print full copy-paste LaTeX table."""
    for line in build_latex_header(condition_label):
        print(line)

    for idx, dataset_result in enumerate(dataset_results):
        include_separator = idx < len(dataset_results) - 1
        for line in build_latex_dataset_block(dataset_result, include_separator):
            print(line)

    for line in build_latex_footer():
        print(line)


def print_latex_body_table(dataset_results: list[LatexDatasetResult]) -> None:
    """Print LaTeX body-only dataset blocks."""
    for idx, dataset_result in enumerate(dataset_results):
        include_separator = idx < len(dataset_results) - 1
        for line in build_latex_dataset_block(dataset_result, include_separator):
            print(line)


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
  python -m src.scripts.experiment_summary outputs/experiment_results/ionosphere.csv
  python -m src.scripts.experiment_summary outputs/experiment_results/*.csv --group-by approach train_size
  python -m src.scripts.experiment_summary outputs/experiment_results/ionosphere.csv --format csv
  python -m src.scripts.experiment_summary outputs/experiment_results/*.csv --format latex --condition-label Heterogeneous
  python -m src.scripts.experiment_summary outputs/experiment_results/*.csv --format latex-body --condition-label Heterogeneous
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
        choices=["table", "csv", "latex", "latex-body"],
        default="table",
        dest="output_format",
        help="Output format (default: table). For latex/latex-body, grouping is fixed by approach.",
    )

    parser.add_argument(
        "--condition-label",
        default="",
        help="Condition label used in latex/latex-body output (e.g., Heterogeneous).",
    )

    args = parser.parse_args()

    if args.output_format in {"latex", "latex-body"} and not args.condition_label.strip():
        parser.error("--condition-label is required for --format latex and --format latex-body")

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

    if args.output_format in {"latex", "latex-body"}:
        dataset_results: list[LatexDatasetResult] = []
        for file_path in expanded_files:
            if not file_path.exists():
                print(f"Error: File not found: {file_path}", file=sys.stderr)
                continue

            try:
                df = load_and_validate_csv(file_path)
                dataset_name = (
                    df["dataset"].iloc[0] if "dataset" in df.columns else file_path.stem
                )
                dataset_results.append(build_latex_dataset_result(df, dataset_name))
            except Exception as e:
                print(f"Error processing {file_path}: {e}", file=sys.stderr)

        dataset_results.sort(key=lambda r: (r.n_train, r.dataset_key))
        if not dataset_results:
            return

        if args.output_format == "latex":
            print_latex_full_table(dataset_results, args.condition_label)
        else:
            print_latex_body_table(dataset_results)
        return

    for file_path in expanded_files:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            continue

        try:
            process_file(
                file_path=file_path,
                group_by=args.group_by,
                output_format=args.output_format,
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
