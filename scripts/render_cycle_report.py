#!/usr/bin/env python3
"""Render a compact markdown report focused on the top submit-ready daily alphas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.approve_seeds import load_seed_store
from scripts.daily_best import render_daily_best
from scripts.flow_utils import load_json as load_artifact_json


def load_json(path: Path) -> dict:
    return load_artifact_json(
        path,
        default={},
        required_fields=("batch",),
        context=f"cycle report planner artifact {path}",
        warn=lambda message: print(f"[render-cycle-report] Warning: {message}", file=sys.stderr),
    )


def build_report(repo_root: Path) -> str:
    artifacts_dir = repo_root / "artifacts"
    batch_payload = load_json(artifacts_dir / "lo_tiep_theo.json")
    seed_store = load_seed_store(repo_root / "initial-population.pkl")
    return render_daily_best(
        batch_payload,
        seed_store,
        title="Latest Submit-Ready Alpha Report",
        include_footer=False,
        top=3,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a compact cycle report.")
    parser.add_argument("--repo-root", default=str(ROOT_DIR), help="Repo root to read artifacts from.")
    parser.add_argument("--output", help="Optional output markdown path.")
    args = parser.parse_args()

    report = build_report(Path(args.repo_root))
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
