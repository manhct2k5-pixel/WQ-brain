#!/usr/bin/env python3
"""Summarize WorldQuant-style simulation CSV exports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

KNOWN_COLUMNS = [
    "alpha_id",
    "regular_code",
    "turnover",
    "returns",
    "drawdown",
    "margin",
    "fitness",
    "sharpe",
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "LOW_SUB_UNIVERSE_SHARPE",
    "SELF_CORRELATION",
    "MATCHES_COMPETITION",
    "date",
]

CHECK_COLUMNS = [
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "LOW_SUB_UNIVERSE_SHARPE",
    "SELF_CORRELATION",
    "MATCHES_COMPETITION",
]


def normalize_expression(expr: str) -> str:
    return re.sub(r"\s+", "", expr or "")


def skeletonize(expr: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "N", normalize_expression(expr))


def coerce_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def discover_csv(path: str | None) -> str:
    candidates = []
    if path:
        if os.path.isdir(path):
            for name in ("simulation_results.csv", "simulations.csv"):
                candidate = os.path.join(path, name)
                if os.path.exists(candidate):
                    return candidate
            raise FileNotFoundError(f"No known CSV found inside {path}")
        return path

    for name in ("simulation_results.csv", "simulations.csv"):
        if os.path.exists(name):
            return name
        candidates.append(name)
    raise FileNotFoundError(
        "No CSV path provided and none of these files exist: "
        + ", ".join(candidates)
    )


def read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return []

    first = [cell.strip() for cell in rows[0]]
    has_header = "regular_code" in first or "alpha_id" in first

    if has_header:
        headers = first
        data_rows = rows[1:]
    else:
        headers = KNOWN_COLUMNS[: len(first)]
        data_rows = rows

    normalized_headers = []
    for header in headers:
        header = header.strip()
        if header == "datetime":
            header = "date"
        normalized_headers.append(header)

    parsed = []
    for raw_row in data_rows:
        if not any(cell.strip() for cell in raw_row):
            continue
        row = {}
        for index, header in enumerate(normalized_headers):
            row[header] = raw_row[index].strip() if index < len(raw_row) else ""
        parsed.append(row)
    return parsed


def compute_research_score(row: dict) -> float:
    sharpe = coerce_float(row.get("sharpe")) or 0.0
    fitness = coerce_float(row.get("fitness")) or 0.0
    turnover = coerce_float(row.get("turnover")) or 0.0
    passed_all = 1.0 if passes_all_checks(row) else 0.0
    turnover_penalty = abs(turnover - 0.25) * 0.5
    return round(sharpe * 0.45 + fitness * 0.35 + passed_all * 0.4 - turnover_penalty, 4)


def failing_checks(row: dict) -> list[str]:
    return [name for name in CHECK_COLUMNS if row.get(name, "").upper() == "FAIL"]


def passes_all_checks(row: dict) -> bool:
    available = [name for name in CHECK_COLUMNS if name in row]
    if not available:
        return False
    return all(row.get(name, "").upper() == "PASS" for name in available)


def build_summary(rows: list[dict], top_n: int) -> dict:
    enriched = []
    failure_counter = Counter()
    skeleton_groups = defaultdict(list)

    for row in rows:
        row = dict(row)
        row["research_score"] = compute_research_score(row)
        row["failures"] = failing_checks(row)
        row["passed_all_checks"] = passes_all_checks(row)
        row["normalized_expression"] = normalize_expression(row.get("regular_code", ""))
        row["skeleton"] = skeletonize(row.get("regular_code", ""))
        skeleton_groups[row["skeleton"]].append(row.get("alpha_id") or row.get("regular_code") or "?")
        failure_counter.update(row["failures"])
        enriched.append(row)

    enriched.sort(
        key=lambda row: (
            row["research_score"],
            coerce_float(row.get("sharpe")) or float("-inf"),
            coerce_float(row.get("fitness")) or float("-inf"),
        ),
        reverse=True,
    )

    top_rows = []
    for row in enriched[:top_n]:
        top_rows.append(
            {
                "alpha_id": row.get("alpha_id", ""),
                "research_score": row["research_score"],
                "sharpe": coerce_float(row.get("sharpe")),
                "fitness": coerce_float(row.get("fitness")),
                "turnover": coerce_float(row.get("turnover")),
                "passed_all_checks": row["passed_all_checks"],
                "failures": row["failures"],
                "regular_code": row.get("regular_code", ""),
            }
        )

    suggestions = []
    if failure_counter["LOW_SHARPE"] >= max(2, len(rows) // 3):
        suggestions.append("Low Sharpe dominates. Change thesis or add stronger normalization instead of tuning windows only.")
    if failure_counter["HIGH_TURNOVER"] >= max(2, len(rows) // 3):
        suggestions.append("High turnover is common. Test slower horizons, smoothing, or liquidity-weighted variants.")
    if failure_counter["LOW_TURNOVER"] >= max(2, len(rows) // 3):
        suggestions.append("Low turnover is common. Add a more reactive trigger or shorten the trigger window.")
    if failure_counter["SELF_CORRELATION"] >= max(2, len(rows) // 4):
        suggestions.append("Self-correlation is elevated. Diversify thesis families and remove near-clone skeletons.")
    if failure_counter["CONCENTRATED_WEIGHT"] >= max(2, len(rows) // 4):
        suggestions.append("Concentrated weights show up often. Prefer rank, zscore, winsorize, or more stable denominators.")

    near_duplicates = {
        skeleton: members
        for skeleton, members in skeleton_groups.items()
        if skeleton and len(members) > 1
    }

    return {
        "summary": {
            "rows": len(rows),
            "pass_all_count": sum(1 for row in enriched if row["passed_all_checks"]),
            "failure_counts": dict(sorted(failure_counter.items())),
            "near_duplicate_skeletons": len(near_duplicates),
        },
        "top_rows": top_rows,
        "near_duplicate_groups": near_duplicates,
        "suggestions": suggestions,
    }


def render_markdown(result: dict, csv_path: str) -> str:
    lines = []
    summary = result["summary"]
    lines.append("# Results Digest")
    lines.append("")
    lines.append(f"- Source: {csv_path}")
    lines.append(f"- Rows: {summary['rows']}")
    lines.append(f"- Pass-all count: {summary['pass_all_count']}")
    lines.append(f"- Near-duplicate skeleton groups: {summary['near_duplicate_skeletons']}")
    lines.append("")

    if summary["failure_counts"]:
        lines.append("## Failure counts")
        for name, count in summary["failure_counts"].items():
            lines.append(f"- {name}: {count}")
        lines.append("")

    lines.append("## Top rows")
    for row in result["top_rows"]:
        lines.append(
            f"- alpha_id={row['alpha_id'] or 'n/a'}, "
            f"score={row['research_score']}, "
            f"sharpe={row['sharpe']}, "
            f"fitness={row['fitness']}, "
            f"turnover={row['turnover']}, "
            f"checks={'PASS' if row['passed_all_checks'] else 'FAIL'}"
        )
        lines.append(f"  expr: {row['regular_code']}")
    lines.append("")

    if result["suggestions"]:
        lines.append("## Suggestions")
        for suggestion in result["suggestions"]:
            lines.append(f"- {suggestion}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize WorldQuant-style simulation result CSV files.",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Optional CSV path, or a directory containing simulation_results.csv or simulations.csv.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top rows to show.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format.",
    )
    args = parser.parse_args()

    try:
        csv_path = discover_csv(args.csv_path)
        rows = read_rows(csv_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not rows:
        print("No rows found in CSV.", file=sys.stderr)
        return 1

    result = build_summary(rows, args.top)
    if args.format == "markdown":
        print(render_markdown(result, csv_path))
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
