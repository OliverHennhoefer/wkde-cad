from __future__ import annotations

import argparse
import math
import sys
import tomllib
from pathlib import Path

import pandas as pd
from scipy.stats import t


REQUIRED_COLUMNS = {"seed", "dataset", "approach", "fdr", "power"}
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.toml"
DEFAULT_DELTA = 0.05
APPROACH_ORDER = [
    "empirical",
    "empirical_randomized",
    "empirical_weighted",
    "empirical_randomized_weighted",
    "probabilistic",
    "probabilistic_weighted",
]
APPROACH_ORDER_MAP = {name: idx for idx, name in enumerate(APPROACH_ORDER)}
METHOD_LABELS = {
    "empirical": "EDF",
    "empirical_randomized": "EDF (randomized)",
    "empirical_weighted": "Weighted EDF",
    "empirical_randomized_weighted": "Weighted EDF (randomized)",
    "probabilistic": "KDE",
    "probabilistic_weighted": "Weighted KDE",
}


def read_default_alpha(config_path: Path = DEFAULT_CONFIG) -> float:
    """Read the nominal FDR level from the project config."""
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    try:
        return float(cfg["global"]["fdr_rate"])
    except KeyError as exc:
        raise ValueError(f"Missing global.fdr_rate in {config_path}") from exc


def load_and_validate_csv(file_path: Path) -> pd.DataFrame:
    """Load one experiment-style result CSV and keep trial-level rows."""
    df = pd.read_csv(file_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {file_path}: {missing}")

    df = df[df["seed"].astype(str) != "mean"].copy()
    for col in ("fdr", "power"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Found non-numeric values in column '{col}' in {file_path}")

    if "severity" in df.columns:
        numeric_severity = pd.to_numeric(df["severity"], errors="coerce")
        if numeric_severity.notna().all():
            df["severity"] = numeric_severity

    return df


def load_output_folder(output_folder: Path) -> pd.DataFrame:
    """Load every CSV in an output folder."""
    if not output_folder.is_dir():
        raise ValueError(f"Output folder does not exist or is not a directory: {output_folder}")

    csv_files = sorted(output_folder.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {output_folder}")

    frames = [load_and_validate_csv(file_path) for file_path in csv_files]
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError(f"No trial-level rows found in {output_folder}")
    return combined


def grouping_columns(df: pd.DataFrame) -> list[str]:
    """Return the report grouping columns available in the input."""
    cols = ["dataset"]
    if "severity" in df.columns:
        cols.append("severity")
    cols.append("approach")
    return cols


def classify_control(
    *,
    ci_lower: float | None,
    ci_upper: float | None,
    alpha: float,
) -> str:
    """Classify FDR control from the requested confidence interval rule."""
    if ci_lower is None or ci_upper is None:
        return "inconclusive"
    if ci_upper <= alpha:
        return "valid"
    if ci_lower > alpha:
        return "invalid"
    return "inconclusive"


def control_note(*, fdr_mean: float, alpha: float, status: str) -> str:
    """Report the mean-FDR side of alpha for inconclusive classifications."""
    if status != "inconclusive":
        return ""
    if fdr_mean > alpha:
        return "FDRhat > alpha"
    return "FDRhat <= alpha"


def compute_summary(
    df: pd.DataFrame,
    *,
    approaches: list[str] | None = None,
    alpha: float,
    delta: float = DEFAULT_DELTA,
) -> pd.DataFrame:
    """Summarize FDR, power, and FDR-control status for result approaches."""
    if approaches is None:
        subset = df.copy()
    else:
        subset = df[df["approach"].isin(approaches)].copy()

    if subset.empty:
        if approaches is None:
            raise ValueError("No rows found")
        raise ValueError(f"No rows found for approaches: {approaches}")

    rows = []
    group_cols = grouping_columns(subset)
    grouped = subset.groupby(group_cols, dropna=False, sort=True)

    for group_key, block in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_values = dict(zip(group_cols, group_key, strict=True))
        n_trials = int(block["fdr"].count())
        fdr_mean = float(block["fdr"].mean())
        fdr_std = float(block["fdr"].std(ddof=1)) if n_trials >= 2 else math.nan
        power_mean = float(block["power"].mean())
        power_std = float(block["power"].std(ddof=1)) if n_trials >= 2 else math.nan

        if n_trials >= 2:
            half_width = (
                float(t.ppf(1.0 - delta, n_trials - 1))
                * fdr_std
                / math.sqrt(n_trials)
            )
            ci_lower = fdr_mean - half_width
            ci_upper = fdr_mean + half_width
        else:
            ci_lower = None
            ci_upper = None

        row = {
            **group_values,
            "n_trials": n_trials,
            "fdr_mean": fdr_mean,
            "fdr_std": fdr_std,
            "power_mean": power_mean,
            "power_std": power_std,
        }
        row["fdr_control"] = classify_control(
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            alpha=alpha,
        )
        row["fdr_control_note"] = control_note(
            fdr_mean=fdr_mean,
            alpha=alpha,
            status=row["fdr_control"],
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["_approach_order"] = (
        summary["approach"].map(APPROACH_ORDER_MAP).fillna(len(APPROACH_ORDER))
    )
    sort_cols = [col for col in group_cols if col != "approach"]
    sort_cols.extend(["_approach_order", "approach"])
    return (
        summary.sort_values(sort_cols, kind="mergesort")
        .drop(columns=["_approach_order"])
        .reset_index(drop=True)
    )


def _format_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_mean_std(mean: float, std: float, precision: int) -> str:
    std_value = 0.0 if pd.isna(std) else float(std)
    return f"{mean:.{precision}f} +/- {std_value:.{precision}f}"


def _method_label(approach: object) -> str:
    return METHOD_LABELS.get(str(approach), str(approach))


def _format_control(row: pd.Series) -> str:
    status = str(row["fdr_control"])
    note = str(row.get("fdr_control_note", ""))
    if not note:
        return status
    return f"{status} ({note})"


def _table_columns(
    summary: pd.DataFrame,
    *,
    include_severity: bool,
) -> list[tuple[str, str]]:
    columns = [("dataset", "Dataset")]
    if include_severity and "severity" in summary.columns:
        columns.append(("severity", "Severity"))
    columns.extend(
        [
            ("approach", "Method"),
            ("fdr", "FDR"),
            ("power", "Power"),
            ("fdr_control", "FDR control"),
        ]
    )
    return columns


def build_display_rows(
    summary: pd.DataFrame,
    *,
    delta: float,
    precision: int,
    include_severity: bool,
) -> tuple[list[str], list[list[str]]]:
    """Build string rows shared by Markdown and LaTeX renderers."""
    columns = _table_columns(summary, include_severity=include_severity)
    headers = [header for _, header in columns]
    rows = []

    for _, row in summary.iterrows():
        values = []
        for col, _ in columns:
            if col == "fdr":
                values.append(_format_mean_std(row["fdr_mean"], row["fdr_std"], precision))
            elif col == "power":
                values.append(_format_mean_std(row["power_mean"], row["power_std"], precision))
            elif col == "approach":
                values.append(_method_label(row[col]))
            elif col == "fdr_control":
                values.append(_format_control(row))
            else:
                values.append(_format_value(row[col]))
        rows.append(values)

    return headers, rows


def _severity_blocks(summary: pd.DataFrame) -> list[tuple[object | None, pd.DataFrame]]:
    if "severity" not in summary.columns:
        return [(None, summary)]
    return [
        (severity, block.reset_index(drop=True))
        for severity, block in summary.groupby("severity", sort=True, dropna=False)
    ]


def _render_markdown_block(
    summary: pd.DataFrame,
    *,
    delta: float,
    precision: int,
    include_severity: bool,
) -> str:
    headers, rows = build_display_rows(
        summary,
        delta=delta,
        precision=precision,
        include_severity=include_severity,
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(summary: pd.DataFrame, *, delta: float, precision: int) -> str:
    """Render GitHub-style Markdown tables, split by severity when available."""
    rendered_blocks = []
    for severity, block in _severity_blocks(summary):
        if severity is None:
            rendered_blocks.append(
                _render_markdown_block(
                    block,
                    delta=delta,
                    precision=precision,
                    include_severity=False,
                )
            )
            continue

        rendered_blocks.append(
            "\n".join(
                [
                    f"### Severity {_format_value(severity)}",
                    "",
                    _render_markdown_block(
                        block,
                        delta=delta,
                        precision=precision,
                        include_severity=False,
                    ),
                ]
            )
        )
    return "\n\n".join(rendered_blocks)


def escape_latex(text: object) -> str:
    """Escape LaTeX special characters in text cells."""
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


def _format_latex_mean_std(mean: float, std: float, precision: int) -> str:
    std_value = 0.0 if pd.isna(std) else float(std)
    return rf"${mean:.{precision}f}\pm{std_value:.{precision}f}$"


def _format_latex_control(row: pd.Series) -> str:
    status = escape_latex(row["fdr_control"])
    note = str(row.get("fdr_control_note", ""))
    if note == "FDRhat > alpha":
        return rf"{status} ($\widehat{{\mathrm{{FDR}}}}>\alpha$)"
    if note == "FDRhat <= alpha":
        return rf"{status} ($\widehat{{\mathrm{{FDR}}}}\le\alpha$)"
    return status


def render_latex(summary: pd.DataFrame, *, delta: float, precision: int) -> str:
    """Render booktabs LaTeX tables, split by severity when available."""
    rendered_blocks = []
    for severity, block in _severity_blocks(summary):
        lines = []
        if severity is not None:
            lines.append(rf"\paragraph{{Severity {_format_value(severity)}}}")
        lines.append(
            _render_latex_block(
                block,
                delta=delta,
                precision=precision,
                include_severity=False,
            )
        )
        rendered_blocks.append("\n".join(lines))
    return "\n\n".join(rendered_blocks)


def _render_latex_block(
    summary: pd.DataFrame,
    *,
    delta: float,
    precision: int,
    include_severity: bool,
) -> str:
    columns = _table_columns(summary, include_severity=include_severity)
    headers = [header for _, header in columns]
    column_spec = " ".join("l" for _ in headers)
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{escape_latex(header)}}}" for header in headers) + r" \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        values = []
        for col, _ in columns:
            if col == "fdr":
                values.append(
                    _format_latex_mean_std(row["fdr_mean"], row["fdr_std"], precision)
                )
            elif col == "power":
                values.append(
                    _format_latex_mean_std(row["power_mean"], row["power_std"], precision)
                )
            elif col == "approach":
                values.append(escape_latex(_method_label(row[col])))
            elif col == "fdr_control":
                values.append(_format_latex_control(row))
            else:
                values.append(escape_latex(_format_value(row[col])))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def render_table(
    summary: pd.DataFrame,
    *,
    output_format: str,
    delta: float,
    precision: int,
) -> str:
    if output_format == "markdown":
        return render_markdown(summary, delta=delta, precision=precision)
    if output_format == "latex":
        return render_latex(summary, delta=delta, precision=precision)
    raise ValueError(f"Unsupported format: {output_format}")


def validate_args(alpha: float, delta: float, precision: int) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if not 0.0 < delta < 1.0:
        raise ValueError("--delta must be between 0 and 1")
    if precision < 0:
        raise ValueError("--precision must be >= 0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Markdown or LaTeX FDR-control tables from result CSV folders.",
    )
    parser.add_argument("output_folder", type=Path, help="Folder containing result CSVs.")
    parser.add_argument(
        "--format",
        choices=["markdown", "latex"],
        default="markdown",
        dest="output_format",
        help="Output table format (default: markdown).",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        help="Optional approaches to include. Defaults to all approaches in the CSVs.",
    )
    parser.add_argument(
        "--validity-approach",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help=f"Nominal FDR level. Defaults to global.fdr_rate in {DEFAULT_CONFIG}.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA,
        help="One-sided confidence-bound tail probability (default: 0.05 for 95%% bounds).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal precision for numeric cells (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output file. Writes to stdout when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        alpha = args.alpha if args.alpha is not None else read_default_alpha()
        validate_args(alpha, args.delta, args.precision)
        df = load_output_folder(args.output_folder)
        approaches = args.approaches
        if args.validity_approach is not None:
            approaches = [args.validity_approach]
        summary = compute_summary(
            df,
            approaches=approaches,
            alpha=alpha,
            delta=args.delta,
        )
        output = render_table(
            summary,
            output_format=args.output_format,
            delta=args.delta,
            precision=args.precision,
        )

        if args.output is None:
            print(output)
        else:
            args.output.write_text(output + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
