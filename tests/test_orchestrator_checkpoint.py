import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import orchestrator


def _run_args(run_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        profile="light",
        run_id=run_id,
        csv_path=None,
        memory=None,
        seed_store="initial-population.pkl",
        auto_fix_input=None,
        scout_input=None,
        top=10,
        count=None,
        queue_limit=None,
        history_window=120,
        scoring="internal",
        timeout=300,
        daily_top=3,
        feed_limit=12,
        adaptive_controls={},
        source_bonus_adjustments={},
        source_quota_profile={},
        manual_overrides={},
    )


class TestOrchestratorCheckpoint(unittest.TestCase):
    def _patch_runtime(self, temp_dir: str):
        artifacts_dir = Path(temp_dir) / "artifacts"
        latest_dir = artifacts_dir / "latest"
        state_dir = artifacts_dir / "state"
        runs_dir = artifacts_dir / "runs"

        def fake_ensure_runtime_layout(run_id: str) -> Path:
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            latest_dir.mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)
            return run_dir

        return patch.multiple(
            orchestrator,
            ARTIFACTS_DIR=artifacts_dir,
            LATEST_DIR=latest_dir,
            GLOBAL_MEMORY_PATH=state_dir / "global_research_memory.json",
            LEGACY_MEMORY_PATH=artifacts_dir / "bo_nho_nghien_cuu.json",
            LEGACY_AUTO_FIX_PATH=artifacts_dir / "auto_fix_candidates.json",
            LEGACY_SCOUT_PATH=artifacts_dir / "_trinh_sat" / "du_lieu.json",
            LATEST_EVALUATED_PATH=latest_dir / "evaluated_candidates.json",
            ensure_runtime_layout=fake_ensure_runtime_layout,
        )

    def _planning_memory(self):
        return {
            "failure_counts": {},
            "blocked_skeletons": [],
            "blocked_families": [],
            "soft_blocked_skeletons": [],
            "soft_blocked_families": [],
            "preferred_skeletons": [],
            "historical_skeletons": [],
            "family_stats": {},
            "style_leaders": [],
            "seed_context": {
                "family_counts": {},
                "planned_family_counts": {},
                "seeded_skeletons": [],
                "planned_skeletons": [],
            },
            "window_rows": 10,
            "suggestions": [],
            "adaptive_controls": {},
        }

    def _planner_batch(self):
        return {
            "candidates": [
                {
                    "source": "planner",
                    "thesis": "Residual or de-beta structure",
                    "thesis_id": "residual_beta",
                    "why": "test",
                    "expression": "rank(close)",
                    "compiled_expression": "rank(close)",
                    "token_program": ["CLOSE", "RANK"],
                    "seed_ready": True,
                    "qualified": True,
                    "quality_label": "qualified",
                    "quality_fail_reasons": [],
                    "confidence_score": 0.82,
                    "candidate_score": 1.1,
                    "novelty_score": 0.9,
                    "style_alignment_score": 0.4,
                    "risk_tags": [],
                    "local_metrics": {
                        "verdict": "PASS",
                        "alpha_score": 78.0,
                        "sharpe": 1.9,
                        "fitness": 1.7,
                    },
                    "settings": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market",
                }
            ],
            "qualified_count": 1,
            "watchlist_count": 0,
            "lint_summary": {},
            "notes": [],
        }

    def _planner_watchlist_batch(self):
        return {
            "candidates": [
                {
                    "source": "planner",
                    "thesis": "VWAP dislocation",
                    "thesis_id": "vwap_dislocation",
                    "why": "watchlist near-miss",
                    "expression": "rank(ts_zscore(abs(close-vwap),21))",
                    "compiled_expression": "rank(ts_zscore(abs(close-vwap),21))",
                    "token_program": ["FACTOR_3", "TSZ_21", "RANK"],
                    "seed_ready": True,
                    "qualified": False,
                    "quality_label": "watchlist",
                    "quality_fail_reasons": ["local", "confidence"],
                    "confidence_score": 0.42,
                    "candidate_score": 0.88,
                    "novelty_score": 0.81,
                    "style_alignment_score": 0.36,
                    "risk_tags": ["weight_risk"],
                    "local_metrics": {
                        "alpha_id": "LOCAL-ABCD1234",
                        "verdict": "FAIL",
                        "confidence": "MEDIUM",
                        "alpha_score": 61.0,
                        "sharpe": 1.23,
                        "fitness": 0.94,
                        "turnover": 0.39,
                        "style_tags": ["vwap", "rank", "normalization"],
                        "LOW_SHARPE": "FAIL",
                        "LOW_FITNESS": "FAIL",
                        "LOW_TURNOVER": "PASS",
                        "HIGH_TURNOVER": "PASS",
                        "CONCENTRATED_WEIGHT": "FAIL",
                        "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                        "SELF_CORRELATION": "PASS",
                        "MATCHES_COMPETITION": "PASS",
                    },
                    "settings": "USA, TOP3000, Decay 3, Delay 1, Truncation 0.02, Neutralization Subindustry",
                }
            ],
            "qualified_count": 0,
            "watchlist_count": 1,
            "lint_summary": {},
            "notes": ["No candidate passed the strict quality gate."],
        }

    def test_run_pipeline_rerun_with_same_run_id_skips_completed_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with ExitStack() as stack:
                stack.enter_context(self._patch_runtime(temp_dir))
                discover_csv_mock = stack.enter_context(patch.object(orchestrator, "discover_csv", return_value="dummy.csv"))
                read_rows_mock = stack.enter_context(patch.object(orchestrator, "read_rows", return_value=[{"regular_code": "rank(close)"}]))
                stack.enter_context(patch.object(orchestrator, "load_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_seed_context", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_memory", return_value=self._planning_memory()))
                stack.enter_context(patch.object(orchestrator, "merge_memory", side_effect=lambda current, _previous: current))
                stack.enter_context(patch.object(orchestrator, "apply_adaptive_planning_controls", side_effect=lambda memory, _controls: memory))
                history_index_mock = stack.enter_context(patch.object(orchestrator.HistoryIndex, "from_csv", return_value=object()))
                build_batch_mock = stack.enter_context(patch.object(orchestrator, "build_batch", return_value=self._planner_batch()))
                merge_pool_mock = stack.enter_context(
                    patch.object(
                        orchestrator,
                        "merge_candidate_pool",
                        return_value={
                            "source_counts": {"planner": 1, "auto_fix_rewrite": 0, "scout": 0},
                            "filtered_counts": {},
                            "candidate_count": 1,
                            "candidates": list(self._planner_batch()["candidates"]),
                        },
                    )
                )

                def fake_evaluate_queue(queue_payload, **_kwargs):
                    candidate = dict(queue_payload["candidates"][0])
                    return [
                        {
                            "run_id": candidate["run_id"],
                            "batch_id": candidate["batch_id"],
                            "candidate_id": candidate["candidate_id"],
                            "source": candidate["source"],
                            "source_stages": candidate.get("source_stages", [candidate["source"]]),
                            "thesis": candidate["thesis"],
                            "thesis_id": candidate["thesis_id"],
                            "why": candidate["why"],
                            "expression": candidate["expression"],
                            "compiled_expression": candidate["compiled_expression"],
                            "token_program": candidate["token_program"],
                            "seed_ready": candidate["seed_ready"],
                            "qualified": candidate["qualified"],
                            "quality_label": candidate["quality_label"],
                            "quality_fail_reasons": candidate["quality_fail_reasons"],
                            "confidence_score": candidate["confidence_score"],
                            "candidate_score": candidate["candidate_score"],
                            "novelty_score": candidate["novelty_score"],
                            "style_alignment_score": candidate["style_alignment_score"],
                            "settings": candidate["settings"],
                            "risk_tags": candidate["risk_tags"],
                            "local_metrics": candidate["local_metrics"],
                            "evaluation_status": "COMPLETED",
                        }
                    ]

                evaluate_queue_mock = stack.enter_context(patch.object(orchestrator, "evaluate_queue", side_effect=fake_evaluate_queue))
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "_results_summary_payload",
                        return_value=({"summary": {"qualified_count": 1}}, "# Results"),
                    )
                )
                update_memory_mock = stack.enter_context(
                    patch.object(
                        orchestrator,
                        "update_research_memory",
                        return_value={
                            "_meta": {},
                            "working_memory": {},
                            "summary_memory": {},
                            "archive_log": {},
                            "planner_memory": {},
                        },
                    )
                )
                stack.enter_context(patch.object(orchestrator, "load_canonical_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "render_daily_best", return_value="# Daily"))
                stack.enter_context(patch.object(orchestrator, "render_alpha_feed", return_value="# Feed"))

                first_summary = orchestrator.run_pipeline(_run_args("resume_demo"))
                run_dir = Path(temp_dir) / "artifacts" / "runs" / "resume_demo"
                latest_dir = Path(temp_dir) / "artifacts" / "latest"
                queue_payload = json.loads((run_dir / "pending_simulation_queue.json").read_text(encoding="utf-8"))
                evaluated_payload = json.loads((run_dir / "evaluated_candidates.json").read_text(encoding="utf-8"))
                checkpoint = json.loads((run_dir / "orchestrator_checkpoint.json").read_text(encoding="utf-8"))
                latest_metadata = json.loads((latest_dir / "latest_metadata.json").read_text(encoding="utf-8"))
                latest_summary_exists = (latest_dir / "orchestrator_summary.json").exists()
                for stage_name in checkpoint["stages"]:
                    checkpoint["stages"][stage_name] = {"state": "pending"}
                (run_dir / "orchestrator_checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

                second_summary = orchestrator.run_pipeline(_run_args("resume_demo"))
                checkpoint = json.loads((run_dir / "orchestrator_checkpoint.json").read_text(encoding="utf-8"))

        self.assertEqual(first_summary["run_id"], "resume_demo")
        self.assertTrue(second_summary["resumed_from_checkpoint"])
        self.assertEqual(discover_csv_mock.call_count, 1)
        self.assertEqual(read_rows_mock.call_count, 1)
        self.assertEqual(history_index_mock.call_count, 1)
        self.assertEqual(build_batch_mock.call_count, 1)
        self.assertEqual(merge_pool_mock.call_count, 1)
        self.assertEqual(evaluate_queue_mock.call_count, 1)
        self.assertEqual(update_memory_mock.call_count, 1)
        self.assertIn("candidate_id", queue_payload["candidates"][0])
        self.assertIn("batch_id", queue_payload["candidates"][0])
        self.assertEqual(queue_payload["candidates"][0]["run_id"], "resume_demo")
        self.assertEqual(evaluated_payload["candidates"][0]["candidate_id"], queue_payload["candidates"][0]["candidate_id"])
        self.assertEqual(latest_metadata["status"], "complete")
        self.assertTrue(latest_metadata["complete"])
        self.assertEqual(latest_metadata["run_id"], "resume_demo")
        self.assertTrue(latest_summary_exists)
        self.assertEqual(checkpoint["stages"]["published"]["state"], "done")

    def test_run_pipeline_manual_exploratory_queue_is_active_for_internal_scoring(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with ExitStack() as stack:
                stack.enter_context(self._patch_runtime(temp_dir))
                stack.enter_context(patch.object(orchestrator, "discover_csv", return_value="dummy.csv"))
                stack.enter_context(patch.object(orchestrator, "read_rows", return_value=[{"regular_code": "rank(close)"}]))
                stack.enter_context(patch.object(orchestrator, "load_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_seed_context", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_memory", return_value=self._planning_memory()))
                stack.enter_context(patch.object(orchestrator, "merge_memory", side_effect=lambda current, _previous: current))
                stack.enter_context(patch.object(orchestrator, "apply_adaptive_planning_controls", side_effect=lambda memory, _controls: memory))
                stack.enter_context(patch.object(orchestrator.HistoryIndex, "from_csv", return_value=object()))
                stack.enter_context(patch.object(orchestrator, "build_batch", return_value=self._planner_batch()))
                captured_merge = {}

                def fake_merge_candidate_pool(**kwargs):
                    captured_merge.update(kwargs)
                    return {
                        "source_counts": {"planner": 1, "auto_fix_rewrite": 0, "scout": 0},
                        "filtered_counts": {},
                        "candidate_count": 1,
                        "candidates": list(kwargs.get("planner_candidates", [])),
                    }

                stack.enter_context(patch.object(orchestrator, "merge_candidate_pool", side_effect=fake_merge_candidate_pool))

                def fake_evaluate_queue(queue_payload, **_kwargs):
                    candidate = dict(queue_payload["candidates"][0])
                    return [
                        {
                            "run_id": candidate["run_id"],
                            "batch_id": candidate["batch_id"],
                            "candidate_id": candidate["candidate_id"],
                            "source": candidate["source"],
                            "source_stages": candidate.get("source_stages", [candidate["source"]]),
                            "thesis": candidate["thesis"],
                            "thesis_id": candidate["thesis_id"],
                            "why": candidate["why"],
                            "expression": candidate["expression"],
                            "compiled_expression": candidate["compiled_expression"],
                            "token_program": candidate["token_program"],
                            "seed_ready": candidate["seed_ready"],
                            "qualified": candidate["qualified"],
                            "quality_label": candidate["quality_label"],
                            "quality_fail_reasons": candidate["quality_fail_reasons"],
                            "confidence_score": candidate["confidence_score"],
                            "candidate_score": candidate["candidate_score"],
                            "novelty_score": candidate["novelty_score"],
                            "style_alignment_score": candidate["style_alignment_score"],
                            "settings": candidate["settings"],
                            "risk_tags": candidate["risk_tags"],
                            "local_metrics": candidate["local_metrics"],
                            "evaluation_status": "COMPLETED",
                        }
                    ]

                stack.enter_context(patch.object(orchestrator, "evaluate_queue", side_effect=fake_evaluate_queue))
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "_results_summary_payload",
                        return_value=({"summary": {"qualified_count": 1}}, "# Results"),
                    )
                )
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "update_research_memory",
                        return_value={
                            "_meta": {},
                            "working_memory": {},
                            "summary_memory": {},
                            "archive_log": {},
                            "planner_memory": {},
                        },
                    )
                )
                stack.enter_context(patch.object(orchestrator, "load_canonical_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "render_daily_best", return_value="# Daily"))
                stack.enter_context(patch.object(orchestrator, "render_alpha_feed", return_value="# Feed"))

                args = _run_args("internal_exploratory")
                args.scoring = "internal"
                args.manual_overrides = {"allow_exploratory_queue": True, "exploratory_queue_limit": 3}

                orchestrator.run_pipeline(args)

        self.assertTrue(captured_merge["exploratory_queue"]["active"])
        self.assertEqual(captured_merge["exploratory_queue"]["limit"], 3)

    def test_run_pipeline_can_resume_from_evaluated_stage_without_replanning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with ExitStack() as stack:
                stack.enter_context(self._patch_runtime(temp_dir))
                stack.enter_context(patch.object(orchestrator, "discover_csv", return_value="dummy.csv"))
                stack.enter_context(patch.object(orchestrator, "read_rows", return_value=[{"regular_code": "rank(close)"}]))
                stack.enter_context(patch.object(orchestrator, "load_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_seed_context", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_memory", return_value=self._planning_memory()))
                stack.enter_context(patch.object(orchestrator, "merge_memory", side_effect=lambda current, _previous: current))
                stack.enter_context(patch.object(orchestrator, "apply_adaptive_planning_controls", side_effect=lambda memory, _controls: memory))
                build_batch_mock = stack.enter_context(patch.object(orchestrator, "build_batch", return_value=self._planner_batch()))
                merge_pool_mock = stack.enter_context(
                    patch.object(
                        orchestrator,
                        "merge_candidate_pool",
                        return_value={
                            "source_counts": {"planner": 1, "auto_fix_rewrite": 0, "scout": 0},
                            "filtered_counts": {},
                            "candidate_count": 1,
                            "candidates": list(self._planner_batch()["candidates"]),
                        },
                    )
                )

                first_summary = orchestrator.run_pipeline(_run_args("resume_partial"))

                run_dir = Path(temp_dir) / "artifacts" / "runs" / "resume_partial"
                checkpoint_path = run_dir / "orchestrator_checkpoint.json"
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                checkpoint["stages"]["evaluated"] = {"state": "pending"}
                checkpoint["stages"]["memory_updated"] = {"state": "pending"}
                checkpoint["stages"]["published"] = {"state": "pending"}
                checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
                (run_dir / "simulation_results.jsonl").unlink(missing_ok=True)
                (run_dir / "results_summary.json").unlink(missing_ok=True)
                (run_dir / "results_summary.md").unlink(missing_ok=True)
                (run_dir / "evaluated_candidates.json").unlink(missing_ok=True)
                (run_dir / "orchestrator_summary.json").unlink(missing_ok=True)

                def fake_evaluate_queue(queue_payload, **_kwargs):
                    candidate = dict(queue_payload["candidates"][0])
                    return [
                        {
                            "run_id": candidate["run_id"],
                            "batch_id": candidate["batch_id"],
                            "candidate_id": candidate["candidate_id"],
                            "source": candidate["source"],
                            "source_stages": candidate.get("source_stages", [candidate["source"]]),
                            "thesis": candidate["thesis"],
                            "thesis_id": candidate["thesis_id"],
                            "why": candidate["why"],
                            "expression": candidate["expression"],
                            "compiled_expression": candidate["compiled_expression"],
                            "token_program": candidate["token_program"],
                            "seed_ready": candidate["seed_ready"],
                            "qualified": candidate["qualified"],
                            "quality_label": candidate["quality_label"],
                            "quality_fail_reasons": candidate["quality_fail_reasons"],
                            "confidence_score": candidate["confidence_score"],
                            "candidate_score": candidate["candidate_score"],
                            "novelty_score": candidate["novelty_score"],
                            "style_alignment_score": candidate["style_alignment_score"],
                            "settings": candidate["settings"],
                            "risk_tags": candidate["risk_tags"],
                            "local_metrics": candidate["local_metrics"],
                            "evaluation_status": "COMPLETED",
                        }
                    ]

                evaluate_queue_mock = stack.enter_context(patch.object(orchestrator, "evaluate_queue", side_effect=fake_evaluate_queue))
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "_results_summary_payload",
                        return_value=({"summary": {"qualified_count": 1}}, "# Results"),
                    )
                )
                update_memory_mock = stack.enter_context(
                    patch.object(
                        orchestrator,
                        "update_research_memory",
                        return_value={
                            "_meta": {},
                            "working_memory": {},
                            "summary_memory": {},
                            "archive_log": {},
                            "planner_memory": {},
                        },
                    )
                )
                stack.enter_context(patch.object(orchestrator, "load_canonical_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "render_daily_best", return_value="# Daily"))
                stack.enter_context(patch.object(orchestrator, "render_alpha_feed", return_value="# Feed"))

                resumed_summary = orchestrator.run_pipeline(_run_args("resume_partial"))

        self.assertEqual(first_summary["run_id"], "resume_partial")
        self.assertIn("planned", resumed_summary["resumed_stages"])
        self.assertIn("merged", resumed_summary["resumed_stages"])
        self.assertIn("evaluated", resumed_summary["executed_stages"])
        self.assertIn("memory_updated", resumed_summary["resumed_stages"])
        self.assertEqual(build_batch_mock.call_count, 1)
        self.assertEqual(merge_pool_mock.call_count, 1)
        self.assertEqual(evaluate_queue_mock.call_count, 1)
        self.assertEqual(update_memory_mock.call_count, 0)

    def test_run_pipeline_republishes_when_latest_metadata_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with ExitStack() as stack:
                stack.enter_context(self._patch_runtime(temp_dir))
                stack.enter_context(patch.object(orchestrator, "discover_csv", return_value="dummy.csv"))
                stack.enter_context(patch.object(orchestrator, "read_rows", return_value=[{"regular_code": "rank(close)"}]))
                stack.enter_context(patch.object(orchestrator, "load_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_seed_context", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_memory", return_value=self._planning_memory()))
                stack.enter_context(patch.object(orchestrator, "merge_memory", side_effect=lambda current, _previous: current))
                stack.enter_context(patch.object(orchestrator, "apply_adaptive_planning_controls", side_effect=lambda memory, _controls: memory))
                stack.enter_context(patch.object(orchestrator.HistoryIndex, "from_csv", return_value=object()))
                stack.enter_context(patch.object(orchestrator, "build_batch", return_value=self._planner_batch()))
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "merge_candidate_pool",
                        return_value={
                            "source_counts": {"planner": 1, "auto_fix_rewrite": 0, "scout": 0},
                            "filtered_counts": {},
                            "candidate_count": 1,
                            "candidates": list(self._planner_batch()["candidates"]),
                        },
                    )
                )

                def fake_evaluate_queue(queue_payload, **_kwargs):
                    candidate = dict(queue_payload["candidates"][0])
                    return [
                        {
                            "run_id": candidate["run_id"],
                            "batch_id": candidate["batch_id"],
                            "candidate_id": candidate["candidate_id"],
                            "source": candidate["source"],
                            "source_stages": candidate.get("source_stages", [candidate["source"]]),
                            "thesis": candidate["thesis"],
                            "thesis_id": candidate["thesis_id"],
                            "why": candidate["why"],
                            "expression": candidate["expression"],
                            "compiled_expression": candidate["compiled_expression"],
                            "token_program": candidate["token_program"],
                            "seed_ready": candidate["seed_ready"],
                            "qualified": candidate["qualified"],
                            "quality_label": candidate["quality_label"],
                            "quality_fail_reasons": candidate["quality_fail_reasons"],
                            "confidence_score": candidate["confidence_score"],
                            "candidate_score": candidate["candidate_score"],
                            "novelty_score": candidate["novelty_score"],
                            "style_alignment_score": candidate["style_alignment_score"],
                            "settings": candidate["settings"],
                            "risk_tags": candidate["risk_tags"],
                            "local_metrics": candidate["local_metrics"],
                            "evaluation_status": "COMPLETED",
                        }
                    ]

                stack.enter_context(patch.object(orchestrator, "evaluate_queue", side_effect=fake_evaluate_queue))
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "_results_summary_payload",
                        return_value=({"summary": {"qualified_count": 1}}, "# Results"),
                    )
                )
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "update_research_memory",
                        return_value={
                            "_meta": {},
                            "working_memory": {},
                            "summary_memory": {},
                            "archive_log": {},
                            "planner_memory": {},
                        },
                    )
                )
                stack.enter_context(patch.object(orchestrator, "load_canonical_seed_store", return_value={}))
                daily_mock = stack.enter_context(patch.object(orchestrator, "render_daily_best", return_value="# Daily"))
                feed_mock = stack.enter_context(patch.object(orchestrator, "render_alpha_feed", return_value="# Feed"))

                orchestrator.run_pipeline(_run_args("resume_publish_only"))

                run_dir = Path(temp_dir) / "artifacts" / "runs" / "resume_publish_only"
                latest_dir = Path(temp_dir) / "artifacts" / "latest"
                checkpoint_path = run_dir / "orchestrator_checkpoint.json"
                metadata_path = latest_dir / "latest_metadata.json"

                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                checkpoint["stages"]["published"] = {"state": "pending"}
                checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

                latest_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                latest_metadata["status"] = "publishing"
                latest_metadata["complete"] = False
                latest_metadata["published_at"] = ""
                metadata_path.write_text(json.dumps(latest_metadata, indent=2), encoding="utf-8")

                resumed_summary = orchestrator.run_pipeline(_run_args("resume_publish_only"))
                repaired_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertIn("published", resumed_summary["executed_stages"])
        self.assertNotIn("published", resumed_summary["resumed_stages"])
        self.assertEqual(repaired_metadata["status"], "complete")
        self.assertTrue(repaired_metadata["complete"])
        self.assertGreaterEqual(daily_mock.call_count, 2)
        self.assertGreaterEqual(feed_mock.call_count, 2)

    def test_run_pipeline_applies_manual_planning_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with ExitStack() as stack:
                stack.enter_context(self._patch_runtime(temp_dir))
                stack.enter_context(patch.object(orchestrator, "discover_csv", return_value="dummy.csv"))
                stack.enter_context(patch.object(orchestrator, "read_rows", return_value=[{"regular_code": "rank(close)"}]))
                stack.enter_context(patch.object(orchestrator, "load_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_seed_context", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_memory", return_value=self._planning_memory()))
                stack.enter_context(patch.object(orchestrator, "merge_memory", side_effect=lambda current, _previous: current))
                captured_controls = {}

                def fake_apply_controls(memory, controls):
                    captured_controls.update(controls or {})
                    adjusted = dict(memory)
                    adjusted["adaptive_controls"] = dict(controls or {})
                    return adjusted

                stack.enter_context(patch.object(orchestrator, "apply_adaptive_planning_controls", side_effect=fake_apply_controls))
                stack.enter_context(patch.object(orchestrator.HistoryIndex, "from_csv", return_value=object()))
                stack.enter_context(patch.object(orchestrator, "build_batch", return_value=self._planner_batch()))
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "merge_candidate_pool",
                        return_value={
                            "source_counts": {"planner": 1, "auto_fix_rewrite": 0, "scout": 0},
                            "filtered_counts": {},
                            "candidate_count": 1,
                            "candidates": list(self._planner_batch()["candidates"]),
                        },
                    )
                )

                def fake_evaluate_queue(queue_payload, **_kwargs):
                    candidate = dict(queue_payload["candidates"][0])
                    return [
                        {
                            "run_id": candidate["run_id"],
                            "batch_id": candidate["batch_id"],
                            "candidate_id": candidate["candidate_id"],
                            "source": candidate["source"],
                            "source_stages": candidate.get("source_stages", [candidate["source"]]),
                            "thesis": candidate["thesis"],
                            "thesis_id": candidate["thesis_id"],
                            "why": candidate["why"],
                            "expression": candidate["expression"],
                            "compiled_expression": candidate["compiled_expression"],
                            "token_program": candidate["token_program"],
                            "seed_ready": candidate["seed_ready"],
                            "qualified": candidate["qualified"],
                            "quality_label": candidate["quality_label"],
                            "quality_fail_reasons": candidate["quality_fail_reasons"],
                            "confidence_score": candidate["confidence_score"],
                            "candidate_score": candidate["candidate_score"],
                            "novelty_score": candidate["novelty_score"],
                            "style_alignment_score": candidate["style_alignment_score"],
                            "settings": candidate["settings"],
                            "risk_tags": candidate["risk_tags"],
                            "local_metrics": candidate["local_metrics"],
                            "evaluation_status": "COMPLETED",
                        }
                    ]

                stack.enter_context(patch.object(orchestrator, "evaluate_queue", side_effect=fake_evaluate_queue))
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "_results_summary_payload",
                        return_value=({"summary": {"qualified_count": 1}}, "# Results"),
                    )
                )
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "update_research_memory",
                        return_value={
                            "_meta": {},
                            "working_memory": {},
                            "summary_memory": {},
                            "archive_log": {},
                            "planner_memory": {},
                        },
                    )
                )
                stack.enter_context(patch.object(orchestrator, "load_canonical_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "render_daily_best", return_value="# Daily"))
                stack.enter_context(patch.object(orchestrator, "render_alpha_feed", return_value="# Feed"))

                args = _run_args("manual_planning_override")
                args.manual_overrides = {"increase_explore": True, "ignore_block_list": True}
                summary = orchestrator.run_pipeline(args)

        self.assertGreaterEqual(captured_controls["exploration_boost"], 0.18)
        self.assertGreaterEqual(captured_controls["exploration_weight_multiplier"], 1.6)
        self.assertTrue(captured_controls["ignore_block_list"])
        self.assertIn("manual_increase_explore", captured_controls["reason_codes"])
        self.assertIn("manual_ignore_block_list", captured_controls["reason_codes"])
        self.assertTrue(summary["manual_overrides"]["increase_explore"])
        self.assertTrue(summary["manual_overrides"]["ignore_block_list"])

    def test_run_pipeline_manual_only_fix_disables_scout_and_freezes_memory_updates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with ExitStack() as stack:
                stack.enter_context(self._patch_runtime(temp_dir))
                stack.enter_context(patch.object(orchestrator, "discover_csv", return_value="dummy.csv"))
                stack.enter_context(patch.object(orchestrator, "read_rows", return_value=[{"regular_code": "rank(close)"}]))
                stack.enter_context(patch.object(orchestrator, "load_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_seed_context", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_memory", return_value=self._planning_memory()))
                stack.enter_context(patch.object(orchestrator, "merge_memory", side_effect=lambda current, _previous: current))
                stack.enter_context(patch.object(orchestrator, "apply_adaptive_planning_controls", side_effect=lambda memory, _controls: memory))
                stack.enter_context(patch.object(orchestrator.HistoryIndex, "from_csv", return_value=object()))
                stack.enter_context(patch.object(orchestrator, "build_batch", return_value=self._planner_batch()))
                captured_merge = {}

                def fake_merge_candidate_pool(**kwargs):
                    captured_merge.update(kwargs)
                    auto_fix_candidates = list(kwargs.get("auto_fix_candidates", []))
                    return {
                        "source_counts": {
                            "planner": len(kwargs.get("planner_candidates", [])),
                            "auto_fix_rewrite": len(auto_fix_candidates),
                            "scout": len(kwargs.get("scout_candidates", [])),
                        },
                        "filtered_counts": {},
                        "candidate_count": len(auto_fix_candidates),
                        "candidates": auto_fix_candidates,
                    }

                stack.enter_context(patch.object(orchestrator, "merge_candidate_pool", side_effect=fake_merge_candidate_pool))

                def fake_evaluate_queue(queue_payload, **_kwargs):
                    candidate = dict(queue_payload["candidates"][0])
                    return [
                        {
                            "run_id": candidate["run_id"],
                            "batch_id": candidate["batch_id"],
                            "candidate_id": candidate["candidate_id"],
                            "source": candidate["source"],
                            "source_stages": candidate.get("source_stages", [candidate["source"]]),
                            "thesis": candidate["thesis"],
                            "thesis_id": candidate["thesis_id"],
                            "why": candidate["why"],
                            "expression": candidate["expression"],
                            "compiled_expression": candidate["compiled_expression"],
                            "token_program": candidate["token_program"],
                            "seed_ready": candidate["seed_ready"],
                            "qualified": candidate["qualified"],
                            "quality_label": candidate["quality_label"],
                            "quality_fail_reasons": candidate["quality_fail_reasons"],
                            "confidence_score": candidate["confidence_score"],
                            "candidate_score": candidate["candidate_score"],
                            "novelty_score": candidate["novelty_score"],
                            "style_alignment_score": candidate["style_alignment_score"],
                            "settings": candidate["settings"],
                            "risk_tags": candidate["risk_tags"],
                            "local_metrics": candidate["local_metrics"],
                            "evaluation_status": "COMPLETED",
                        }
                    ]

                stack.enter_context(patch.object(orchestrator, "evaluate_queue", side_effect=fake_evaluate_queue))
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "_results_summary_payload",
                        return_value=({"summary": {"qualified_count": 1}}, "# Results"),
                    )
                )
                update_memory_mock = stack.enter_context(
                    patch.object(
                        orchestrator,
                        "update_research_memory",
                        return_value={
                            "_meta": {},
                            "working_memory": {},
                            "summary_memory": {},
                            "archive_log": {},
                            "planner_memory": {},
                        },
                    )
                )
                stack.enter_context(patch.object(orchestrator, "load_canonical_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "render_daily_best", return_value="# Daily"))
                stack.enter_context(patch.object(orchestrator, "render_alpha_feed", return_value="# Feed"))

                auto_fix_input = Path(temp_dir) / "manual_auto_fix.json"
                scout_input = Path(temp_dir) / "manual_scout.json"
                auto_fix_candidate = {
                    **self._planner_batch()["candidates"][0],
                    "source": "auto_fix_rewrite",
                }
                auto_fix_input.write_text(json.dumps({"candidates": [auto_fix_candidate]}), encoding="utf-8")
                scout_input.write_text(
                    json.dumps({"selected": [{"expression": "rank(volume)", "source": "scout"}]}),
                    encoding="utf-8",
                )

                args = _run_args("manual_fix_only")
                args.auto_fix_input = str(auto_fix_input)
                args.scout_input = str(scout_input)
                args.manual_overrides = {"only_fix": True, "freeze_memory_update": True}

                summary = orchestrator.run_pipeline(args)
                run_dir = Path(temp_dir) / "artifacts" / "runs" / "manual_fix_only"
                checkpoint = json.loads((run_dir / "orchestrator_checkpoint.json").read_text(encoding="utf-8"))

        self.assertEqual(captured_merge["planner_candidates"], [])
        self.assertEqual(captured_merge["scout_candidates"], [])
        self.assertEqual(len(captured_merge["auto_fix_candidates"]), 1)
        self.assertEqual(update_memory_mock.call_count, 0)
        self.assertTrue(summary["manual_overrides"]["only_fix"])
        self.assertTrue(summary["manual_overrides"]["freeze_memory_update"])
        self.assertTrue(checkpoint["stages"]["memory_updated"]["details"]["frozen"])

    def test_run_pipeline_auto_generates_auto_fix_candidates_from_planner_watchlist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with ExitStack() as stack:
                stack.enter_context(self._patch_runtime(temp_dir))
                stack.enter_context(patch.object(orchestrator, "discover_csv", return_value="dummy.csv"))
                stack.enter_context(patch.object(orchestrator, "read_rows", return_value=[{"regular_code": "rank(close)"}]))
                stack.enter_context(patch.object(orchestrator, "load_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_seed_context", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_memory", return_value=self._planning_memory()))
                stack.enter_context(patch.object(orchestrator, "merge_memory", side_effect=lambda current, _previous: current))
                stack.enter_context(patch.object(orchestrator, "apply_adaptive_planning_controls", side_effect=lambda memory, _controls: memory))
                stack.enter_context(patch.object(orchestrator.HistoryIndex, "from_csv", return_value=object()))
                stack.enter_context(patch.object(orchestrator, "build_batch", return_value=self._planner_watchlist_batch()))
                build_auto_fix_payload_mock = stack.enter_context(
                    patch.object(
                        orchestrator,
                        "build_auto_fix_payload",
                        return_value={"candidates": [{"expression": "rank(ts_sum(close,10))"}]},
                    )
                )
                generated_auto_fix_candidate = {
                    "source": "auto_fix_rewrite",
                    "thesis": "Auto-fix rewrite [Technical Indicator]",
                    "thesis_id": "technical_indicator",
                    "why": "Auto-fixed from planner watchlist.",
                    "expression": "rank(ts_sum(close,10))",
                    "compiled_expression": "rank(ts_sum(close,10))",
                    "token_program": ["CLOSE", "TSS_10", "RANK"],
                    "seed_ready": True,
                    "qualified": True,
                    "quality_label": "qualified",
                    "quality_fail_reasons": [],
                    "confidence_score": 0.76,
                    "candidate_score": 0.91,
                    "novelty_score": 0.73,
                    "style_alignment_score": 0.58,
                    "risk_tags": [],
                    "repair_status": "submit_ready",
                    "local_metrics": {
                        "verdict": "PASS",
                        "confidence": "HIGH",
                        "alpha_score": 71.0,
                        "sharpe": 1.55,
                        "fitness": 1.13,
                        "turnover": 0.29,
                    },
                    "settings": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market",
                }
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "build_actionable_auto_fix_candidates",
                        return_value=[generated_auto_fix_candidate],
                    )
                )
                captured_merge = {}

                def fake_merge_candidate_pool(**kwargs):
                    captured_merge.update(kwargs)
                    auto_fix_candidates = list(kwargs.get("auto_fix_candidates", []))
                    return {
                        "source_counts": {
                            "planner": len(kwargs.get("planner_candidates", [])),
                            "auto_fix_rewrite": len(auto_fix_candidates),
                            "scout": len(kwargs.get("scout_candidates", [])),
                        },
                        "filtered_counts": {},
                        "candidate_count": len(auto_fix_candidates),
                        "candidates": auto_fix_candidates,
                    }

                stack.enter_context(patch.object(orchestrator, "merge_candidate_pool", side_effect=fake_merge_candidate_pool))

                def fake_evaluate_queue(queue_payload, **_kwargs):
                    candidate = dict(queue_payload["candidates"][0])
                    return [
                        {
                            "run_id": candidate["run_id"],
                            "batch_id": candidate["batch_id"],
                            "candidate_id": candidate["candidate_id"],
                            "source": candidate["source"],
                            "source_stages": candidate.get("source_stages", [candidate["source"]]),
                            "thesis": candidate["thesis"],
                            "thesis_id": candidate["thesis_id"],
                            "why": candidate["why"],
                            "expression": candidate["expression"],
                            "compiled_expression": candidate["compiled_expression"],
                            "token_program": candidate["token_program"],
                            "seed_ready": candidate["seed_ready"],
                            "qualified": candidate["qualified"],
                            "quality_label": candidate["quality_label"],
                            "quality_fail_reasons": candidate["quality_fail_reasons"],
                            "confidence_score": candidate["confidence_score"],
                            "candidate_score": candidate["candidate_score"],
                            "novelty_score": candidate["novelty_score"],
                            "style_alignment_score": candidate["style_alignment_score"],
                            "settings": candidate["settings"],
                            "risk_tags": candidate["risk_tags"],
                            "local_metrics": candidate["local_metrics"],
                            "evaluation_status": "COMPLETED",
                        }
                    ]

                stack.enter_context(patch.object(orchestrator, "evaluate_queue", side_effect=fake_evaluate_queue))
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "_results_summary_payload",
                        return_value=({"summary": {"qualified_count": 1}}, "# Results"),
                    )
                )
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "update_research_memory",
                        return_value={
                            "_meta": {},
                            "working_memory": {},
                            "summary_memory": {},
                            "archive_log": {},
                            "planner_memory": {},
                        },
                    )
                )
                stack.enter_context(patch.object(orchestrator, "load_canonical_seed_store", return_value={}))
                report_capture = {}

                def fake_render_daily_best(payload, seed_store, **kwargs):
                    report_capture["daily_extra"] = list(kwargs.get("extra_candidates") or [])
                    return "# Daily"

                def fake_render_alpha_feed(payload, **kwargs):
                    report_capture["feed_extra"] = list(kwargs.get("extra_candidates") or [])
                    return "# Feed"

                stack.enter_context(patch.object(orchestrator, "render_daily_best", side_effect=fake_render_daily_best))
                stack.enter_context(patch.object(orchestrator, "render_alpha_feed", side_effect=fake_render_alpha_feed))

                summary = orchestrator.run_pipeline(_run_args("auto_fix_synth"))
                run_dir = Path(temp_dir) / "artifacts" / "runs" / "auto_fix_synth"
                auto_fix_payload = json.loads((run_dir / "auto_fix_candidates.json").read_text(encoding="utf-8"))

        self.assertEqual(build_auto_fix_payload_mock.call_count, 1)
        self.assertEqual(len(captured_merge["auto_fix_candidates"]), 1)
        self.assertEqual(captured_merge["auto_fix_candidates"][0]["source"], "auto_fix_rewrite")
        self.assertEqual(auto_fix_payload["orchestrator_generation"]["generated_candidate_count"], 1)
        self.assertEqual(auto_fix_payload["orchestrator_generation"]["context_count"], 1)
        self.assertEqual(auto_fix_payload["candidates"][0]["expression"], "rank(ts_sum(close,10))")
        self.assertEqual(report_capture["daily_extra"][0]["expression"], "rank(ts_sum(close,10))")
        self.assertEqual(report_capture["feed_extra"][0]["expression"], "rank(ts_sum(close,10))")
        self.assertEqual(summary["auto_fix_candidates_generated"], 1)
        self.assertEqual(summary["auto_fix_candidates_available"], 1)

    def test_run_pipeline_can_recover_from_crash_mid_publish(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with ExitStack() as stack:
                stack.enter_context(self._patch_runtime(temp_dir))
                stack.enter_context(patch.object(orchestrator, "discover_csv", return_value="dummy.csv"))
                stack.enter_context(patch.object(orchestrator, "read_rows", return_value=[{"regular_code": "rank(close)"}]))
                stack.enter_context(patch.object(orchestrator, "load_seed_store", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_seed_context", return_value={}))
                stack.enter_context(patch.object(orchestrator, "build_memory", return_value=self._planning_memory()))
                stack.enter_context(patch.object(orchestrator, "merge_memory", side_effect=lambda current, _previous: current))
                stack.enter_context(patch.object(orchestrator, "apply_adaptive_planning_controls", side_effect=lambda memory, _controls: memory))
                stack.enter_context(patch.object(orchestrator.HistoryIndex, "from_csv", return_value=object()))
                stack.enter_context(patch.object(orchestrator, "build_batch", return_value=self._planner_batch()))
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "merge_candidate_pool",
                        return_value={
                            "source_counts": {"planner": 1, "auto_fix_rewrite": 0, "scout": 0},
                            "filtered_counts": {},
                            "candidate_count": 1,
                            "candidates": list(self._planner_batch()["candidates"]),
                        },
                    )
                )

                def fake_evaluate_queue(queue_payload, **_kwargs):
                    candidate = dict(queue_payload["candidates"][0])
                    return [
                        {
                            "run_id": candidate["run_id"],
                            "batch_id": candidate["batch_id"],
                            "candidate_id": candidate["candidate_id"],
                            "source": candidate["source"],
                            "source_stages": candidate.get("source_stages", [candidate["source"]]),
                            "thesis": candidate["thesis"],
                            "thesis_id": candidate["thesis_id"],
                            "why": candidate["why"],
                            "expression": candidate["expression"],
                            "compiled_expression": candidate["compiled_expression"],
                            "token_program": candidate["token_program"],
                            "seed_ready": candidate["seed_ready"],
                            "qualified": candidate["qualified"],
                            "quality_label": candidate["quality_label"],
                            "quality_fail_reasons": candidate["quality_fail_reasons"],
                            "confidence_score": candidate["confidence_score"],
                            "candidate_score": candidate["candidate_score"],
                            "novelty_score": candidate["novelty_score"],
                            "style_alignment_score": candidate["style_alignment_score"],
                            "settings": candidate["settings"],
                            "risk_tags": candidate["risk_tags"],
                            "local_metrics": candidate["local_metrics"],
                            "evaluation_status": "COMPLETED",
                        }
                    ]

                stack.enter_context(patch.object(orchestrator, "evaluate_queue", side_effect=fake_evaluate_queue))
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "_results_summary_payload",
                        return_value=({"summary": {"qualified_count": 1}}, "# Results"),
                    )
                )
                stack.enter_context(
                    patch.object(
                        orchestrator,
                        "update_research_memory",
                        return_value={
                            "_meta": {},
                            "working_memory": {},
                            "summary_memory": {},
                            "archive_log": {},
                            "planner_memory": {},
                        },
                    )
                )
                stack.enter_context(patch.object(orchestrator, "load_canonical_seed_store", return_value={}))
                daily_mock = stack.enter_context(patch.object(orchestrator, "render_daily_best", return_value="# Daily"))
                feed_mock = stack.enter_context(patch.object(orchestrator, "render_alpha_feed", return_value="# Feed"))

                latest_dir = Path(temp_dir) / "artifacts" / "latest"
                original_copy_file = orchestrator.copy_file
                failed_once = {"raised": False}

                def flaky_copy(source, target, *, use_lock=True):
                    target_path = Path(target)
                    if not failed_once["raised"] and target_path == latest_dir / "bang_tin_alpha.md":
                        failed_once["raised"] = True
                        raise RuntimeError("simulated publish crash")
                    return original_copy_file(source, target, use_lock=use_lock)

                with patch.object(orchestrator, "copy_file", side_effect=flaky_copy):
                    with self.assertRaisesRegex(RuntimeError, "simulated publish crash"):
                        orchestrator.run_pipeline(_run_args("resume_publish_crash"))

                run_dir = Path(temp_dir) / "artifacts" / "runs" / "resume_publish_crash"
                metadata_path = latest_dir / "latest_metadata.json"
                checkpoint_path = run_dir / "orchestrator_checkpoint.json"
                crashed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                crashed_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

                resumed_summary = orchestrator.run_pipeline(_run_args("resume_publish_crash"))
                repaired_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(crashed_metadata["status"], "publishing")
        self.assertFalse(crashed_metadata["complete"])
        self.assertNotEqual(crashed_checkpoint["stages"]["published"]["state"], "done")
        self.assertIn("published", resumed_summary["executed_stages"])
        self.assertEqual(repaired_metadata["status"], "complete")
        self.assertTrue(repaired_metadata["complete"])
        self.assertGreaterEqual(daily_mock.call_count, 2)
        self.assertGreaterEqual(feed_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
