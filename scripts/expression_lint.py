#!/usr/bin/env python3
"""Lint and compare WorldQuant-style FastExpr batches."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict

IDENTIFIER_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
WINDOW_RE = re.compile(r"(?<=[,(])\s*(\d+)\s*(?=[,)])")


def normalize_expression(expr: str) -> str:
    return re.sub(r"\s+", "", expr.strip())


def skeletonize(expr: str) -> str:
    normalized = normalize_expression(expr)
    return re.sub(r"\d+(?:\.\d+)?", "N", normalized)


def infer_family(expr: str) -> list[str]:
    lower = expr.lower()
    families = []
    if "ts_corr" in lower or "correlation" in lower:
        families.append("correlation")
    if "ts_regression" in lower or "beta" in lower:
        families.append("residual")
    if "volume" in lower or "vwap" in lower:
        families.append("liquidity")
    if "returns" in lower or "reverse(" in lower:
        families.append("reversal")
    if "rank(" in lower or "zscore(" in lower or "winsorize(" in lower:
        families.append("normalization")
    if "ts_std_dev" in lower or "vol" in lower:
        families.append("volatility")
    return families or ["uncategorized"]


def lint_expression(expr: str) -> dict:
    normalized = normalize_expression(expr)
    skeleton = skeletonize(expr)
    identifiers = IDENTIFIER_RE.findall(expr)
    windows = [int(value) for value in WINDOW_RE.findall(expr)]
    warnings = []

    paren_balance = normalized.count("(") - normalized.count(")")
    if paren_balance != 0:
        warnings.append("unbalanced_parentheses")
    if len(normalized) < 12:
        warnings.append("very_short_expression")
    if len(normalized) > 220:
        warnings.append("long_expression")
    if len(identifiers) < 2:
        warnings.append("low_operator_density")
    if windows and len(set(windows)) == 1 and len(windows) >= 3:
        warnings.append("single_window_reused_many_times")

    return {
        "expression": expr,
        "normalized": normalized,
        "skeleton": skeleton,
        "length": len(normalized),
        "operators": identifiers,
        "operator_count": len(identifiers),
        "unique_operators": sorted(set(identifiers)),
        "windows": windows,
        "families": infer_family(expr),
        "warnings": warnings,
    }


def load_expressions(args: argparse.Namespace) -> list[str]:
    expressions = []

    for expr in args.expr:
        if expr.strip():
            expressions.append(expr.strip())

    if args.file:
        with open(args.file, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                expressions.append(line)

    if not expressions and not sys.stdin.isatty():
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            expressions.append(line)

    return expressions


def build_summary(expressions: list[str]) -> dict:
    reports = [lint_expression(expr) for expr in expressions]
    exact_groups = defaultdict(list)
    skeleton_groups = defaultdict(list)
    family_counter = Counter()
    warning_counter = Counter()

    for index, report in enumerate(reports, start=1):
        report["id"] = index
        exact_groups[report["normalized"]].append(index)
        skeleton_groups[report["skeleton"]].append(index)
        family_counter.update(report["families"])
        warning_counter.update(report["warnings"])

    for report in reports:
        report["exact_duplicate_group"] = exact_groups[report["normalized"]]
        report["near_duplicate_group"] = skeleton_groups[report["skeleton"]]

    return {
        "summary": {
            "total_expressions": len(reports),
            "exact_duplicate_groups": [
                ids for ids in exact_groups.values() if len(ids) > 1
            ],
            "near_duplicate_groups": [
                ids for ids in skeleton_groups.values() if len(ids) > 1
            ],
            "family_counts": dict(sorted(family_counter.items())),
            "warning_counts": dict(sorted(warning_counter.items())),
        },
        "expressions": reports,
    }


def render_markdown(result: dict) -> str:
    lines = []
    summary = result["summary"]
    lines.append("# Expression Lint")
    lines.append("")
    lines.append(f"- Total expressions: {summary['total_expressions']}")
    lines.append(f"- Exact duplicate groups: {len(summary['exact_duplicate_groups'])}")
    lines.append(f"- Near-duplicate groups: {len(summary['near_duplicate_groups'])}")
    lines.append("")

    if summary["family_counts"]:
        lines.append("## Family counts")
        for name, count in summary["family_counts"].items():
            lines.append(f"- {name}: {count}")
        lines.append("")

    if summary["warning_counts"]:
        lines.append("## Warning counts")
        for name, count in summary["warning_counts"].items():
            lines.append(f"- {name}: {count}")
        lines.append("")

    lines.append("## Expressions")
    for report in result["expressions"]:
        lines.append(
            f"- #{report['id']}: len={report['length']}, "
            f"families={','.join(report['families'])}, "
            f"warnings={','.join(report['warnings']) or 'none'}"
        )
        lines.append(f"  expr: {report['expression']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lint and compare batches of WorldQuant-style expressions.",
    )
    parser.add_argument(
        "--expr",
        action="append",
        default=[],
        help="Expression to lint. Repeat for multiple expressions.",
    )
    parser.add_argument(
        "--file",
        help="Optional text file with one expression per line.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format.",
    )
    args = parser.parse_args()

    expressions = load_expressions(args)
    if not expressions:
        print("No expressions provided.", file=sys.stderr)
        return 1

    result = build_summary(expressions)
    if args.format == "markdown":
        print(render_markdown(result))
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
