#!/usr/bin/env python3
"""Render a markdown feed of all current alpha candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.daily_best import DEFAULT_AUTO_FIX_INPUT, load_auto_fix_candidates
from scripts.flow_utils import load_json as load_artifact_json

DEFAULT_INPUT = Path("artifacts/latest/evaluated_candidates.json")
DEFAULT_LEGACY_INPUT = Path("artifacts/lo_tiep_theo.json")


def _warn_json_issue(message: str) -> None:
    print(f"[alpha-feed] Warning: {message}", file=sys.stderr)


def load_json(path: Path) -> dict:
    return load_artifact_json(
        path,
        default={},
        context=f"alpha feed input {path}",
        warn=_warn_json_issue,
    )


def resolve_input_path(path: str | None) -> Path:
    if path:
        return Path(path)
    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT
    return DEFAULT_LEGACY_INPUT


def payload_candidates(payload: dict) -> list[dict]:
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, dict)]
    batch = payload.get("batch", {})
    return [item for item in batch.get("candidates", []) if isinstance(item, dict)]


def render_alpha_feed(payload: dict, *, limit: int | None = None, extra_candidates: list[dict] | None = None) -> str:
    memory = payload.get("memory", {})
    batch = payload.get("batch", {})
    candidates = list(payload_candidates(payload))
    candidates.sort(
        key=lambda item: (
            int(item.get("qualified", False)),
            float(item.get("confidence_score", 0.0)),
            float(item.get("candidate_score", 0.0)),
            float(item.get("novelty_score", 0.0)),
            float(item.get("style_alignment_score", 0.0)),
        ),
        reverse=True,
    )
    if limit is not None and limit > 0:
        candidates = candidates[:limit]

    lines = ["# Alpha Feed", ""]
    lines.append("- Output target: full markdown feed of current candidates.")
    lines.append("- Review the logic yourself before any manual submission.")
    lines.append(f"- candidates: {len(candidates)}")
    lines.append("")

    style_leaders = memory.get("style_leaders", [])
    if style_leaders:
        lines.append("## Style Leaders")
        for item in style_leaders[:5]:
            lines.append(f"- {item.get('tag')}: learning_score={item.get('learning_score')}")
        lines.append("")

    notes = batch.get("notes", [])
    if notes:
        lines.append("## Notes")
        for item in notes:
            lines.append(f"- {item}")
        lines.append("")

    actionable_auto_fix = [
        item
        for item in (extra_candidates or [])
        if item.get("repair_status") in {"submit_ready", "promising"}
    ]
    if actionable_auto_fix:
        lines.append("## Auto-Fix Actionable")
        for index, candidate in enumerate(actionable_auto_fix, start=1):
            local = candidate.get("local_metrics", {})
            lines.append(f"- #{index} [{candidate.get('thesis')}] {candidate.get('expression')}")
            lines.append(
                f"  repair_status: {candidate.get('repair_status')}, "
                f"alpha_score: {local.get('alpha_score')}, "
                f"sharpe: {local.get('sharpe')}, "
                f"fitness: {local.get('fitness')}"
            )
            lines.append(f"  why: {candidate.get('why')}")
            lines.append(f"  risk_tags: {', '.join(candidate.get('risk_tags', [])) or 'none'}")
        lines.append("")

    qualified_candidates = [item for item in candidates if item.get("qualified")]
    watchlist_candidates = [item for item in candidates if not item.get("qualified")]

    if qualified_candidates:
        lines.append("## Strictly Qualified")
        for index, candidate in enumerate(qualified_candidates, start=1):
            lines.append(f"- #{index} [{candidate.get('thesis')}] {candidate.get('expression')}")
            lines.append(f"  why: {candidate.get('why')}")
            lines.append(
                f"  confidence_score: {candidate.get('confidence_score')}, "
                f"candidate_score: {candidate.get('candidate_score')}, "
                f"novelty_score: {candidate.get('novelty_score')}, "
                f"style_alignment_score: {candidate.get('style_alignment_score')}"
            )
            lines.append(f"  seed_ready: {candidate.get('seed_ready')}")
            lines.append(f"  risk_tags: {', '.join(candidate.get('risk_tags', [])) or 'none'}")
        lines.append("")

    if watchlist_candidates:
        if not qualified_candidates:
            lines.append("## Watchlist")
            lines.append("- Reference only: these candidates did not pass the strict gate and should not be seeded or submitted as-is.")
        else:
            lines.append("## Watchlist")
        for index, candidate in enumerate(watchlist_candidates, start=1):
            lines.append(f"- #{index} [{candidate.get('thesis')}] {candidate.get('expression')}")
            lines.append(
                f"  confidence_score: {candidate.get('confidence_score')}, "
                f"quality_fail_reasons: {', '.join(candidate.get('quality_fail_reasons', [])) or 'none'}"
            )
            lines.append(f"  risk_tags: {', '.join(candidate.get('risk_tags', [])) or 'none'}")
        lines.append("")

    if not candidates:
        lines.append("No candidates are available in the current planning artifacts.")
        lines.append("")

    lines.append("## Files")
    lines.append("- artifacts/latest/bang_tin_alpha.md")
    lines.append("- artifacts/latest/evaluated_candidates.json")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a markdown feed of all current alpha candidates.")
    parser.add_argument("--input", help="Evaluated candidate JSON input. Defaults to artifacts/latest/evaluated_candidates.json, then artifacts/lo_tiep_theo.json.")
    parser.add_argument("--auto-fix-input", help="Optional legacy auto-fix candidate JSON to surface in the feed.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of candidates to include.")
    parser.add_argument("--output", help="Optional output markdown path.")
    args = parser.parse_args()

    payload_path = resolve_input_path(args.input)
    if not payload_path.exists():
        print(f"Planner JSON not found: {payload_path}", file=sys.stderr)
        return 1

    payload = load_json(payload_path)
    extra_candidates = load_auto_fix_candidates(Path(args.auto_fix_input)) if args.auto_fix_input else []
    markdown = render_alpha_feed(payload, limit=args.limit, extra_candidates=extra_candidates)

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
