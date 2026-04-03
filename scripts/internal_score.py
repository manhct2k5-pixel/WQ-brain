#!/usr/bin/env python3
"""Score one or more alpha expressions locally without contacting WorldQuant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.internal_scoring import HistoryIndex, format_scoring_settings, parse_scoring_settings, score_expression


def load_expressions(expression: str | None, file_path: str | None) -> list[str]:
    expressions = []
    if expression:
        expressions.append(expression.strip())
    if file_path:
        for raw_line in Path(file_path).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            expressions.append(line)

    deduped = []
    seen = set()
    for item in expressions:
        if not item or item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped


def render_markdown(results: list[dict], top: int) -> str:
    lines = ["# Internal Alpha Score", ""]
    ranked = sorted(
        results,
        key=lambda item: (item.get("alpha_score", 0.0), item.get("sharpe", 0.0), item.get("fitness", 0.0)),
        reverse=True,
    )

    for index, result in enumerate(ranked[:top], start=1):
        lines.append(f"## Candidate {index}")
        lines.append(f"- expression: {result['expression']}")
        lines.append(f"- verdict: {result['verdict']} ({result['confidence']})")
        lines.append(f"- alpha_type: {result['alpha_type']}")
        lines.append(f"- settings: {result['settings']['label']}")
        lines.append(f"- alpha_score: {result['alpha_score']}")
        lines.append(f"- sharpe: {result['sharpe']}")
        lines.append(f"- fitness: {result['fitness']}")
        lines.append(f"- ic_proxy: {result['ic_proxy']}")
        lines.append(f"- turnover: {result['turnover']}")
        lines.append(f"- stability_proxy: {result['stability_proxy']}")
        lines.append(f"- capacity_proxy: {result['capacity_proxy']}")
        lines.append(f"- uniqueness_proxy: {result['uniqueness_proxy']}")
        lines.append(f"- ensemble_proxy: {result['ensemble_proxy']}")
        lines.append(f"- style_tags: {', '.join(result['style_tags']) if result['style_tags'] else 'none'}")
        shadow = result.get("surrogate_shadow", {})
        if shadow.get("status") == "ready":
            lines.append(
                "- surrogate_shadow: "
                f"{shadow.get('preview_verdict')} "
                f"(fitness={shadow.get('predicted_fitness')}, "
                f"sharpe={shadow.get('predicted_sharpe')}, "
                f"rows={shadow.get('training_rows')}, "
                f"alignment={shadow.get('alignment')})"
            )
        else:
            lines.append(f"- surrogate_shadow: {shadow.get('status', 'unavailable')}")
        lines.append("- checks:")
        for check_name in (
            "LOW_SHARPE",
            "LOW_FITNESS",
            "LOW_TURNOVER",
            "HIGH_TURNOVER",
            "CONCENTRATED_WEIGHT",
            "LOW_SUB_UNIVERSE_SHARPE",
            "SELF_CORRELATION",
            "MATCHES_COMPETITION",
        ):
            lines.append(f"  - {check_name}: {result[check_name]}")
        lines.append("- optimization_hints:")
        for hint in result["optimization_hints"]:
            lines.append(f"  - {hint}")
        if result["strengths"]:
            lines.append("- strengths:")
            for item in result["strengths"]:
                lines.append(f"  - {item}")
        if result["weaknesses"]:
            lines.append("- weaknesses:")
            for item in result["weaknesses"]:
                lines.append(f"  - {item}")
        if result["suggestions"]:
            lines.append("- suggestions:")
            for item in result["suggestions"]:
                lines.append(f"  - [{item['type']}] {item['description']}")
                lines.append(f"    example: {item['example']}")
        lines.append(f"- score_breakdown: {json.dumps(result['score_breakdown'], ensure_ascii=True)}")
        lines.append(f"- analysis: {result['analysis']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Score alpha expressions locally with the internal heuristic model.")
    parser.add_argument("--expression", help="Single alpha expression to score.")
    parser.add_argument("--file", help="Text file containing one alpha expression per line.")
    parser.add_argument("--csv", help="Optional history CSV path for duplicate and family context.")
    parser.add_argument(
        "--settings",
        default=None,
        help=(
            "Optional scoring settings string, for example "
            "'USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market'."
        ),
    )
    parser.add_argument("--top", type=int, default=5, help="Maximum number of rows to show in markdown mode.")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown", help="Output format.")
    args = parser.parse_args()

    expressions = load_expressions(args.expression, args.file)
    if not expressions:
        print("Provide --expression or --file with at least one alpha expression.", file=sys.stderr)
        return 1

    history_index = HistoryIndex.from_csv(args.csv)
    scoring_settings = parse_scoring_settings(args.settings)
    results = []
    for expression in expressions:
        result = score_expression(expression, history_index=history_index, settings=scoring_settings)
        results.append(result)
        history_index.observe_expression(expression, result)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        if args.settings:
            print(f"Using settings: {format_scoring_settings(scoring_settings)}\n")
        print(render_markdown(results, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
