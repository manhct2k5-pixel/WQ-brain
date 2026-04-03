#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-feed}"
EXTRA_ARGS=("${@:2}")

cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in this shell."
  echo "Install it in WSL with:"
  echo "  sudo apt update && sudo apt install -y python3 python3-pip"
  exit 1
fi

refresh_artifacts() {
  mkdir -p artifacts
  if [ -f simulation_results.csv ] || [ -f simulations.csv ]; then
    python3 scripts/results_digest.py --format markdown | tee artifacts/tom_tat_moi_nhat.md
    python3 scripts/plan_next_batch.py \
      --format markdown \
      --memory artifacts/bo_nho_nghien_cuu.json \
      --write-memory artifacts/bo_nho_nghien_cuu.json \
      --write-batch artifacts/bieu_thuc_ung_vien.txt \
      --write-plan artifacts/lo_tiep_theo.json \
      | tee artifacts/lo_tiep_theo.md
    python3 scripts/render_cycle_report.py --output artifacts/bao_cao_moi_nhat.md >/dev/null
    python3 scripts/manual_review.py --input artifacts/lo_tiep_theo.json --output artifacts/duyet_tay.md >/dev/null
    python3 scripts/daily_best.py --input artifacts/lo_tiep_theo.json --output artifacts/alpha_tot_nhat_hom_nay.md >/dev/null
    python3 scripts/alpha_feed.py --input artifacts/lo_tiep_theo.json --output artifacts/bang_tin_alpha.md >/dev/null
  else
    echo "No simulation CSV found yet, skipping digest and next-batch planning."
  fi
}

new_run_id() {
  date '+%Y%m%d_%H%M%S'
}

has_extra_arg() {
  local needle="$1"
  for arg in "${EXTRA_ARGS[@]}"; do
    if [ "$arg" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

latest_publish_is_complete() {
  python3 - <<'PY'
import json
from pathlib import Path

metadata_path = Path("artifacts/latest/latest_metadata.json")
if not metadata_path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "complete" and payload.get("complete") else 1)
PY
}

run_orchestrator() {
  local profile="$1"
  local scoring="$2"
  local run_id
  run_id="$(new_run_id)"
  python3 scripts/orchestrator.py --profile "$profile" --run-id "$run_id" --scoring "$scoring" "${EXTRA_ARGS[@]}"
}

run_cleanup() {
  python3 scripts/cleanup_artifacts.py "$@"
}

run_cleanup_quiet() {
  if ! python3 scripts/cleanup_artifacts.py >/dev/null; then
    echo "Artifact cleanup skipped because scripts/cleanup_artifacts.py failed." >&2
  fi
}

requested_scoring() {
  local default_scoring="$1"
  local next_is_value=0
  for arg in "${EXTRA_ARGS[@]}"; do
    if [ "$next_is_value" -eq 1 ]; then
      echo "$arg"
      return 0
    fi
    if [ "$arg" = "--scoring" ]; then
      next_is_value=1
    fi
  done
  echo "$default_scoring"
}

doctor_for_orchestrator() {
  local scoring
  scoring="$(requested_scoring "$1")"
  if [ "$scoring" = "internal" ]; then
    python3 scripts/doctor.py --mode wsl --require-deps
  else
    python3 scripts/doctor.py --mode wsl --require-env --require-deps --check-auth
  fi
}

case "$ACTION" in
  doctor)
    python3 scripts/doctor.py --mode wsl
    ;;
  auth)
    python3 scripts/doctor.py --mode wsl --require-env --require-deps --check-auth
    ;;
  digest)
    python3 scripts/results_digest.py --format markdown
    ;;
  plan)
    mkdir -p artifacts
    python3 scripts/plan_next_batch.py \
      --format markdown \
      --memory artifacts/bo_nho_nghien_cuu.json \
      --write-memory artifacts/bo_nho_nghien_cuu.json \
      --write-batch artifacts/bieu_thuc_ung_vien.txt \
      --write-plan artifacts/lo_tiep_theo.json
    python3 scripts/render_cycle_report.py --output artifacts/bao_cao_moi_nhat.md >/dev/null
    python3 scripts/manual_review.py --input artifacts/lo_tiep_theo.json --output artifacts/duyet_tay.md >/dev/null
    ;;
  review)
    refresh_artifacts
    if [ -f artifacts/duyet_tay.md ]; then
      cat artifacts/duyet_tay.md
    fi
    ;;
  daily)
    if latest_publish_is_complete && [ -f artifacts/latest/alpha_tot_nhat_hom_nay.md ]; then
      cat artifacts/latest/alpha_tot_nhat_hom_nay.md
    else
      echo "artifacts/latest is not marked complete yet. Check artifacts/latest/latest_metadata.json."
    fi
    run_cleanup_quiet
    ;;
  feed)
    if latest_publish_is_complete && [ -f artifacts/latest/bang_tin_alpha.md ]; then
      cat artifacts/latest/bang_tin_alpha.md
    else
      echo "artifacts/latest is not marked complete yet. Check artifacts/latest/latest_metadata.json."
    fi
    ;;
  scout)
    python3 scripts/doctor.py --mode wsl --require-deps
    RUN_ID="$(new_run_id)"
    RUN_DIR="artifacts/recent_runs/${RUN_ID}"
    mkdir -p "$RUN_DIR" artifacts/latest
    python3 scripts/scout_ideas.py \
      --output "$RUN_DIR/scout_report.md" \
      --write-plan "$RUN_DIR/scout_candidates.json" \
      --write-batch "$RUN_DIR/scout_batch.txt" \
      --memory-path "$RUN_DIR/scout_memory.json" \
      "${EXTRA_ARGS[@]}"
    if [ -f "$RUN_DIR/scout_candidates.json" ]; then
      cp "$RUN_DIR/scout_candidates.json" artifacts/latest/scout_candidates.json
    fi
    if [ -f "$RUN_DIR/scout_report.md" ]; then
      cat "$RUN_DIR/scout_report.md"
    fi
    run_cleanup_quiet
    ;;
  scout-loop|until-ready)
    python3 scripts/doctor.py --mode wsl --require-deps
    python3 scripts/scout_loop.py --clear-stop-file --clear-stale-locks "${EXTRA_ARGS[@]}"
    ;;
  scout-stop)
    mkdir -p artifacts/_trinh_sat
    python3 scripts/runtime_control.py request-stop \
      --stop-file artifacts/_trinh_sat/DUNG \
      --status-file artifacts/_trinh_sat/scout_loop_status.json \
      --reason user_requested_stop >/dev/null
    echo "Requested scout-loop stop via artifacts/_trinh_sat/DUNG"
    ;;
  fix)
    RUN_ID="$(new_run_id)"
    RUN_DIR="artifacts/recent_runs/${RUN_ID}"
    mkdir -p "$RUN_DIR" artifacts/latest
    FIX_ARGS=("${EXTRA_ARGS[@]}")
    if ! has_extra_arg "--output"; then
      FIX_ARGS+=(--output "$RUN_DIR/fix_report.md")
    fi
    if has_extra_arg "--auto-rewrite" && ! has_extra_arg "--write-candidates"; then
      FIX_ARGS+=(--write-candidates "$RUN_DIR/auto_fix_candidates.json")
    fi
    python3 scripts/fix_alpha.py "${FIX_ARGS[@]}"
    if [ -f "$RUN_DIR/auto_fix_candidates.json" ]; then
      cp "$RUN_DIR/auto_fix_candidates.json" artifacts/latest/auto_fix_candidates.json
      cp "$RUN_DIR/auto_fix_candidates.json" artifacts/auto_fix_candidates.json
    fi
    if [ -f "$RUN_DIR/fix_report.md" ]; then
      cat "$RUN_DIR/fix_report.md"
    fi
    run_cleanup_quiet
    ;;
  score)
    python3 scripts/internal_score.py "${EXTRA_ARGS[@]}"
    ;;
  seed)
    python3 scripts/approve_seeds.py --input artifacts/lo_tiep_theo.json --top 4 --format markdown
    ;;
  seed-submit-ready)
    RUN_ID="$(new_run_id)"
    RUN_DIR="artifacts/recent_runs/${RUN_ID}"
    mkdir -p "$RUN_DIR"
    if ! latest_publish_is_complete; then
      echo "artifacts/latest is not marked complete yet. Refusing to seed from artifacts/latest/evaluated_candidates.json."
      exit 1
    fi
    python3 scripts/seed_submit_ready.py --input artifacts/latest/evaluated_candidates.json --top 3 --snapshot-path "$RUN_DIR/seed_population.pkl" --format markdown
    run_cleanup_quiet
    ;;
  cleanup)
    run_cleanup "${EXTRA_ARGS[@]}"
    ;;
  internal|offline)
    doctor_for_orchestrator internal
    run_orchestrator light internal
    run_cleanup_quiet
    ;;
  test)
    python3 scripts/doctor.py --mode wsl --require-env --require-deps
    python3 main.py --mode test "${EXTRA_ARGS[@]}"
    refresh_artifacts
    ;;
  careful)
    doctor_for_orchestrator worldquant
    run_orchestrator careful worldquant
    run_cleanup_quiet
    ;;
  smart)
    doctor_for_orchestrator worldquant
    run_orchestrator smart worldquant
    run_cleanup_quiet
    ;;
  light|run|cycle)
    doctor_for_orchestrator worldquant
    run_orchestrator light worldquant
    run_cleanup_quiet
    ;;
  full)
    doctor_for_orchestrator worldquant
    run_orchestrator full worldquant
    run_cleanup_quiet
    ;;
  max|turbo)
    doctor_for_orchestrator worldquant
    run_orchestrator turbo worldquant
    run_cleanup_quiet
    ;;
  loop)
    doctor_for_orchestrator worldquant
    python3 scripts/orchestrator_loop.py --clear-stop-file --clear-stale-locks "${EXTRA_ARGS[@]}"
    ;;
  loop-stop)
    mkdir -p artifacts/state
    python3 scripts/runtime_control.py request-stop \
      --stop-file artifacts/state/DUNG_LOOP \
      --status-file artifacts/state/orchestrator_loop_status.json \
      --reason user_requested_stop >/dev/null
    echo "Requested orchestrator loop stop via artifacts/state/DUNG_LOOP"
    ;;
  loop-status)
    if [ -f artifacts/state/orchestrator_loop_status.json ]; then
      cat artifacts/state/orchestrator_loop_status.json
    else
      echo "No orchestrator loop status found yet."
    fi
    ;;
  *)
    echo "Unknown action: $ACTION"
    echo "Usage: bash run_wsl.sh [doctor|auth|digest|plan|review|daily|feed|scout|scout-loop|until-ready|scout-stop|fix|score|seed|seed-submit-ready|cleanup|internal|offline|test|careful|smart|light|run|cycle|full|max|turbo|loop|loop-stop|loop-status]"
    exit 1
    ;;
esac
