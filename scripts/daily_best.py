#!/usr/bin/env python3
"""Render the top submit-ready daily alpha candidates from the latest planning artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.approve_seeds import load_seed_store, score_candidate
from scripts.flow_utils import load_json as load_artifact_json
from src.submit_gate import (
    SUBMIT_READY_MIN_ALPHA_SCORE,
    SUBMIT_READY_MIN_CONFIDENCE,
    SUBMIT_READY_MIN_FITNESS,
    SUBMIT_READY_MIN_SHARPE,
    SUBMIT_READY_VERDICTS,
    candidate_passes_submit_gate,
)
DEFAULT_TOP_CANDIDATES = 3
DEFAULT_INPUT = Path("artifacts/latest/evaluated_candidates.json")
DEFAULT_LEGACY_INPUT = Path("artifacts/lo_tiep_theo.json")
DEFAULT_AUTO_FIX_INPUT = Path("artifacts/auto_fix_candidates.json")


def _warn_json_issue(message: str) -> None:
    print(f"[daily-best] Warning: {message}", file=sys.stderr)


def _validate_candidate_payload(payload: dict) -> str | None:
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return None
    batch = payload.get("batch")
    if isinstance(batch, dict) and isinstance(batch.get("candidates"), list):
        return None
    return "expected candidates or batch.candidates to be a list"


def load_json(path: Path) -> dict:
    return load_artifact_json(
        path,
        default={},
        context=f"daily-best input {path}",
        validator=_validate_candidate_payload,
        warn=_warn_json_issue,
    )


def load_auto_fix_candidates(path: Path | None) -> list[dict]:
    if not path:
        return []
    payload = load_json(path)
    candidates = payload.get("candidates", [])
    return [item for item in candidates if isinstance(item, dict)]


def resolve_input_path(path: str | None) -> Path:
    if path:
        return Path(path)
    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT
    return DEFAULT_LEGACY_INPUT


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def verdict_rank(candidate: dict) -> int:
    verdict = str((candidate.get("local_metrics") or {}).get("verdict") or "").upper()
    ranks = {
        "PASS": 4,
        "LIKELY_PASS": 3,
        "BORDERLINE": 2,
        "FAIL": 1,
    }
    return ranks.get(verdict, 0)


def review_score(candidate: dict) -> float:
    local = candidate.get("local_metrics") or {}
    quality_bonus = 12.0 if candidate.get("quality_label") == "qualified" else 0.0
    return round(
        verdict_rank(candidate) * 30.0
        + to_float(local.get("alpha_score"))
        + quality_bonus
        + score_candidate(candidate),
        4,
    )


def is_submit_ready(candidate: dict) -> bool:
    return candidate_passes_submit_gate(candidate)


def submit_score(candidate: dict) -> float:
    local = candidate.get("local_metrics") or {}
    quality_bonus = 6.0 if candidate.get("quality_label") == "qualified" else 0.0
    return round(
        verdict_rank(candidate) * 40.0
        + to_float(local.get("alpha_score"))
        + 8.0 * to_float(local.get("fitness"))
        + 6.0 * to_float(local.get("sharpe"))
        + 4.0 * to_float(candidate.get("confidence_score"))
        + quality_bonus,
        4,
    )


def _payload_candidates(payload: dict) -> list[dict]:
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, dict)]
    return [item for item in (payload.get("batch", {}).get("candidates", []) or []) if isinstance(item, dict)]


def _candidate_pool(payload: dict, extra_candidates: list[dict] | None = None) -> list[dict]:
    merged = {}
    for raw_candidate in [*_payload_candidates(payload), *((extra_candidates or []))]:
        if not isinstance(raw_candidate, dict):
            continue
        expression = raw_candidate.get("compiled_expression") or raw_candidate.get("expression")
        if not expression:
            continue
        current = merged.get(expression)
        if current is None:
            merged[expression] = raw_candidate
            continue
        current_score = to_float((current.get("local_metrics") or {}).get("alpha_score"))
        new_score = to_float((raw_candidate.get("local_metrics") or {}).get("alpha_score"))
        if new_score >= current_score:
            merged[expression] = raw_candidate
    return list(merged.values())


def select_submit_ready_candidates(
    payload: dict,
    seed_store: dict,
    *,
    top: int = DEFAULT_TOP_CANDIDATES,
    extra_candidates: list[dict] | None = None,
) -> list[dict]:
    candidates = []
    existing_expressions = set(seed_store or {})
    for raw_candidate in _candidate_pool(payload, extra_candidates):
        if not is_submit_ready(raw_candidate):
            continue
        candidate = dict(raw_candidate)
        compiled_expression = candidate.get("compiled_expression") or candidate.get("expression")
        candidate["already_in_seed_store"] = compiled_expression in existing_expressions
        candidate["selection_score"] = score_candidate(candidate)
        candidate["review_score"] = review_score(candidate)
        candidate["submit_score"] = submit_score(candidate)
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            item.get("submit_score", 0.0),
            verdict_rank(item),
            to_float((item.get("local_metrics") or {}).get("alpha_score")),
            to_float((item.get("local_metrics") or {}).get("fitness")),
            to_float((item.get("local_metrics") or {}).get("sharpe")),
            to_float(item.get("confidence_score")),
            to_float(item.get("novelty_score")),
        ),
        reverse=True,
    )
    return candidates[:top]


def render_daily_best(
    payload: dict,
    seed_store: dict,
    *,
    title: str = "Top Submit-Ready Alphas Today",
    include_footer: bool = True,
    top: int = DEFAULT_TOP_CANDIDATES,
    extra_candidates: list[dict] | None = None,
) -> str:
    candidates = select_submit_ready_candidates(payload, seed_store, top=top, extra_candidates=extra_candidates)
    auto_fix_count = sum(1 for item in candidates if item.get("source") == "auto_fix_rewrite")

    lines = [f"# {title}", ""]
    lines.append(f"- Standard view: up to {top} best candidates that clear the submit-ready gate.")
    lines.append(
        f"- Submit-ready gate: seed_ready=true, quality_label=qualified, confidence>={SUBMIT_READY_MIN_CONFIDENCE}, "
        f"verdict in {sorted(SUBMIT_READY_VERDICTS)}, "
        f"alpha_score>={SUBMIT_READY_MIN_ALPHA_SCORE}, sharpe>={SUBMIT_READY_MIN_SHARPE}, "
        f"fitness>={SUBMIT_READY_MIN_FITNESS}."
    )
    lines.append(f"- Candidates shown: {len(candidates)}")
    if extra_candidates:
        lines.append(f"- Auto-fix candidates considered: {len(extra_candidates)}")
        lines.append(f"- Auto-fix candidates shown: {auto_fix_count}")
    lines.append("")

    if not candidates:
        lines.append("No candidate met the submit-ready gate today.")
        lines.append("")
    else:
        for index, candidate in enumerate(candidates, start=1):
            local = candidate.get("local_metrics", {})
            lines.append(f"## Candidate {index}")
            lines.append(f"- thesis: {candidate.get('thesis')}")
            lines.append(f"- expression: {candidate.get('expression')}")
            if candidate.get("source") == "auto_fix_rewrite":
                lines.append("- source: auto_fix_rewrite")
            lines.append(f"- settings: {candidate.get('settings') or 'n/a'}")
            lines.append(f"- verdict: {local.get('verdict')} ({local.get('confidence')})")
            lines.append(f"- alpha_score: {local.get('alpha_score')}")
            lines.append(f"- sharpe: {local.get('sharpe')}")
            lines.append(f"- fitness: {local.get('fitness')}")
            lines.append(f"- turnover: {local.get('turnover')}")
            lines.append(f"- submit_score: {candidate.get('submit_score')}")
            lines.append(f"- planner_confidence: {candidate.get('confidence_score')}")
            lines.append(f"- quality_label: {candidate.get('quality_label')}")
            lines.append(f"- seed_store_status: {'already seeded' if candidate.get('already_in_seed_store') else 'new candidate'}")
            lines.append(f"- why: {candidate.get('why')}")
            lines.append(f"- risk_tags: {', '.join(candidate.get('risk_tags', [])) or 'none'}")
            lines.append("")

    if include_footer:
        lines.append("## Canonical File")
        lines.append("- artifacts/latest/alpha_tot_nhat_hom_nay.md")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the top submit-ready daily alpha candidates.")
    parser.add_argument("--input", help="Evaluated candidate JSON input. Defaults to artifacts/latest/evaluated_candidates.json, then artifacts/lo_tiep_theo.json.")
    parser.add_argument("--seed-path", default="initial-population.pkl", help="Current seed store path.")
    parser.add_argument("--auto-fix-input", help="Optional legacy auto-fix candidate JSON to merge into the daily shortlist.")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_CANDIDATES, help="Maximum number of submit-ready candidates to show.")
    parser.add_argument("--output", help="Optional output markdown path.")
    args = parser.parse_args()

    payload_path = resolve_input_path(args.input)
    if not payload_path.exists():
        print(f"Planner JSON not found: {payload_path}", file=sys.stderr)
        return 1

    payload = load_json(payload_path)
    seed_store = load_seed_store(Path(args.seed_path))
    extra_candidates = load_auto_fix_candidates(Path(args.auto_fix_input)) if args.auto_fix_input else []
    markdown = render_daily_best(payload, seed_store, top=args.top, extra_candidates=extra_candidates)

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
