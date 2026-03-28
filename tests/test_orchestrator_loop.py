import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import orchestrator_loop
from scripts.runtime_control import write_signal_file


class TestOrchestratorLoop(unittest.TestCase):
    @patch(
        "scripts.orchestrator_loop.run_pipeline",
        return_value={
            "run_id": "light_20260327_210000",
            "queue_candidates": 4,
            "submit_ready_candidates": 1,
            "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
            "feed_report": "artifacts/latest/bang_tin_alpha.md",
        },
    )
    def test_loop_runs_one_round_and_writes_status(self, run_pipeline_mock):
        with tempfile.TemporaryDirectory() as temp_dir:
            stop_file = Path(temp_dir) / "DUNG_LOOP"
            status_file = Path(temp_dir) / "loop_status.json"
            with patch(
                "sys.argv",
                [
                    "orchestrator_loop.py",
                    "--scoring",
                    "internal",
                    "--max-rounds",
                    "1",
                    "--interval-minutes",
                    "0",
                    "--stop-file",
                    str(stop_file),
                    "--status-file",
                    str(status_file),
                ],
            ):
                exit_code = orchestrator_loop.main()
            payload = json.loads(status_file.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_pipeline_mock.call_count, 1)
        self.assertEqual(payload["state"], "stopped")
        self.assertEqual(payload["reason"], "max_rounds")
        self.assertEqual(payload["summary"]["submit_ready_candidates"], 1)
        self.assertIsInstance(payload["pid"], int)
        run_args = run_pipeline_mock.call_args.args[0]
        self.assertEqual(run_args.scoring, "internal")

    @patch(
        "scripts.orchestrator_loop.run_pipeline",
        return_value={
            "run_id": "light_20260327_210000",
            "queue_candidates": 1,
            "submit_ready_candidates": 0,
            "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
            "feed_report": "artifacts/latest/bang_tin_alpha.md",
        },
    )
    def test_loop_forwards_manual_override_flags(self, run_pipeline_mock):
        with tempfile.TemporaryDirectory() as temp_dir:
            stop_file = Path(temp_dir) / "DUNG_LOOP"
            status_file = Path(temp_dir) / "loop_status.json"
            with patch(
                "sys.argv",
                [
                    "orchestrator_loop.py",
                    "--scoring",
                    "internal",
                    "--max-rounds",
                    "1",
                    "--interval-minutes",
                    "0",
                    "--manual-only-fix",
                    "--manual-increase-explore",
                    "--manual-freeze-memory-update",
                    "--manual-ignore-block-list",
                    "--stop-file",
                    str(stop_file),
                    "--status-file",
                    str(status_file),
                ],
            ):
                exit_code = orchestrator_loop.main()
            payload = json.loads(status_file.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        run_args = run_pipeline_mock.call_args.args[0]
        self.assertTrue(run_args.manual_overrides["active"])
        self.assertTrue(run_args.manual_overrides["only_fix"])
        self.assertTrue(run_args.manual_overrides["disable_scout"])
        self.assertTrue(run_args.manual_overrides["increase_explore"])
        self.assertTrue(run_args.manual_overrides["freeze_memory_update"])
        self.assertTrue(run_args.manual_overrides["ignore_block_list"])
        self.assertTrue(payload["manual_overrides"]["only_fix"])

    @patch(
        "scripts.orchestrator_loop.run_pipeline",
        return_value={
            "run_id": "light_20260327_210001",
            "queue_candidates": 2,
            "submit_ready_candidates": 0,
            "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
            "feed_report": "artifacts/latest/bang_tin_alpha.md",
        },
    )
    def test_loop_can_stop_via_stop_file_after_a_round(self, _run_pipeline_mock):
        with tempfile.TemporaryDirectory() as temp_dir:
            stop_file = Path(temp_dir) / "DUNG_LOOP"
            status_file = Path(temp_dir) / "loop_status.json"

            def fake_sleep(_seconds, stop_path):
                stop_path.write_text("", encoding="utf-8")
                return False

            with patch("scripts.orchestrator_loop._sleep_until_next_round", side_effect=fake_sleep):
                with patch(
                    "sys.argv",
                    [
                        "orchestrator_loop.py",
                        "--scoring",
                        "internal",
                        "--interval-minutes",
                        "1",
                        "--stop-file",
                        str(stop_file),
                        "--status-file",
                        str(status_file),
                    ],
                ):
                    exit_code = orchestrator_loop.main()
                payload = json.loads(status_file.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["state"], "stopped")
        self.assertEqual(payload["reason"], "stop_file_detected")

    @patch("scripts.orchestrator_loop.run_pipeline", side_effect=RuntimeError("boom"))
    def test_loop_stops_after_max_consecutive_failures(self, _run_pipeline_mock):
        with tempfile.TemporaryDirectory() as temp_dir:
            stop_file = Path(temp_dir) / "DUNG_LOOP"
            status_file = Path(temp_dir) / "loop_status.json"
            with patch(
                "sys.argv",
                [
                    "orchestrator_loop.py",
                    "--scoring",
                    "internal",
                    "--interval-minutes",
                    "0",
                    "--max-consecutive-failures",
                    "2",
                    "--stop-file",
                    str(stop_file),
                    "--status-file",
                    str(status_file),
                ],
            ):
                exit_code = orchestrator_loop.main()
            payload = json.loads(status_file.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["state"], "stopped")
        self.assertEqual(payload["reason"], "max_consecutive_failures")

    @patch(
        "scripts.orchestrator_loop.run_pipeline",
        return_value={
            "run_id": "light_20260327_210002",
            "queue_candidates": 1,
            "submit_ready_candidates": 0,
            "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
            "feed_report": "artifacts/latest/bang_tin_alpha.md",
        },
    )
    def test_loop_removes_stale_stop_file_before_starting(self, run_pipeline_mock):
        with tempfile.TemporaryDirectory() as temp_dir:
            stop_file = Path(temp_dir) / "DUNG_LOOP"
            status_file = Path(temp_dir) / "loop_status.json"
            write_signal_file(
                stop_file,
                kind="stop_request",
                reason="user_requested_stop",
                run_id="old_run",
                target_pid=999999,
            )
            with patch("scripts.orchestrator_loop.clear_stale_stop_file", wraps=orchestrator_loop.clear_stale_stop_file), patch(
                "scripts.runtime_control.pid_is_alive",
                return_value=False,
            ), patch(
                "sys.argv",
                [
                    "orchestrator_loop.py",
                    "--scoring",
                    "internal",
                    "--max-rounds",
                    "1",
                    "--interval-minutes",
                    "0",
                    "--stop-file",
                    str(stop_file),
                    "--status-file",
                    str(status_file),
                ],
            ):
                exit_code = orchestrator_loop.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_pipeline_mock.call_count, 1)
        self.assertFalse(stop_file.exists())

    def test_loop_writes_system_and_error_logs(self):
        def fake_run_pipeline(_args):
            print("system stdout line")
            print("system stderr line", file=sys.stderr)
            return {
                "run_id": "light_20260327_210003",
                "queue_candidates": 3,
                "submit_ready_candidates": 1,
                "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
                "feed_report": "artifacts/latest/bang_tin_alpha.md",
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            stop_file = Path(temp_dir) / "DUNG_LOOP"
            status_file = Path(temp_dir) / "loop_status.json"
            with patch("scripts.orchestrator_loop.run_pipeline", side_effect=fake_run_pipeline), patch(
                "sys.argv",
                [
                    "orchestrator_loop.py",
                    "--scoring",
                    "internal",
                    "--max-rounds",
                    "1",
                    "--interval-minutes",
                    "0",
                    "--stop-file",
                    str(stop_file),
                    "--status-file",
                    str(status_file),
                ],
            ):
                exit_code = orchestrator_loop.main()
            system_log = (Path(temp_dir) / "system.log").read_text(encoding="utf-8")
            error_log = (Path(temp_dir) / "error.log").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("system stdout line", system_log)
        self.assertIn("system stderr line", error_log)
        self.assertIn("run_id=light_", system_log)

    def test_loop_logs_traceback_on_round_crash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stop_file = Path(temp_dir) / "DUNG_LOOP"
            status_file = Path(temp_dir) / "loop_status.json"
            with patch("scripts.orchestrator_loop.run_pipeline", side_effect=RuntimeError("boom")), patch(
                "sys.argv",
                [
                    "orchestrator_loop.py",
                    "--scoring",
                    "internal",
                    "--interval-minutes",
                    "0",
                    "--max-consecutive-failures",
                    "1",
                    "--stop-file",
                    str(stop_file),
                    "--status-file",
                    str(status_file),
                ],
            ):
                exit_code = orchestrator_loop.main()
            error_log = (Path(temp_dir) / "error.log").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 1)
        self.assertIn("Round 1 crashed", error_log)
        self.assertIn("Traceback", error_log)

    @patch("scripts.orchestrator_loop.run_pipeline")
    def test_loop_records_health_warning_when_submit_ready_is_stale(self, run_pipeline_mock):
        with tempfile.TemporaryDirectory() as temp_dir:
            latest_metadata = Path(temp_dir) / "latest_metadata.json"
            global_memory = Path(temp_dir) / "global_research_memory.json"
            latest_metadata.write_text(
                json.dumps({"status": "complete", "complete": True, "artifacts": {}}),
                encoding="utf-8",
            )
            global_memory.write_text(
                json.dumps(
                    {
                        "working_memory": {},
                        "summary_memory": {},
                        "archive_log": {},
                        "planner_memory": {},
                    }
                ),
                encoding="utf-8",
            )
            run_pipeline_mock.side_effect = [
                {
                    "run_id": "light_20260328_100001",
                    "queue_candidates": 4,
                    "evaluated_candidates": 4,
                    "submit_ready_candidates": 0,
                    "latest_metadata": str(latest_metadata),
                    "global_memory_path": str(global_memory),
                    "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
                    "feed_report": "artifacts/latest/bang_tin_alpha.md",
                },
                {
                    "run_id": "light_20260328_100002",
                    "queue_candidates": 4,
                    "evaluated_candidates": 4,
                    "submit_ready_candidates": 0,
                    "latest_metadata": str(latest_metadata),
                    "global_memory_path": str(global_memory),
                    "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
                    "feed_report": "artifacts/latest/bang_tin_alpha.md",
                },
            ]

            stop_file = Path(temp_dir) / "DUNG_LOOP"
            status_file = Path(temp_dir) / "loop_status.json"
            with patch(
                "sys.argv",
                [
                    "orchestrator_loop.py",
                    "--scoring",
                    "internal",
                    "--max-rounds",
                    "2",
                    "--interval-minutes",
                    "0",
                    "--health-no-pass-warning-rounds",
                    "2",
                    "--stop-file",
                    str(stop_file),
                    "--status-file",
                    str(status_file),
                ],
            ):
                exit_code = orchestrator_loop.main()
            payload = json.loads(status_file.read_text(encoding="utf-8"))
            system_log = (Path(temp_dir) / "system.log").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["health"]["highest_severity"], "warning")
        self.assertEqual(payload["health"]["checks"]["submit_ready_freshness"]["severity"], "warning")
        self.assertIn("[health][warning]", system_log)

    @patch("scripts.orchestrator_loop.run_pipeline")
    def test_loop_enables_recovery_mode_after_consecutive_dry_rounds(self, run_pipeline_mock):
        run_pipeline_mock.side_effect = [
            {
                "run_id": "light_20260328_000001",
                "queue_candidates": 4,
                "evaluated_candidates": 4,
                "submit_ready_candidates": 0,
                "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
                "feed_report": "artifacts/latest/bang_tin_alpha.md",
            },
            {
                "run_id": "light_20260328_000002",
                "queue_candidates": 4,
                "evaluated_candidates": 4,
                "submit_ready_candidates": 0,
                "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
                "feed_report": "artifacts/latest/bang_tin_alpha.md",
            },
            {
                "run_id": "light_20260328_000003",
                "queue_candidates": 4,
                "evaluated_candidates": 4,
                "submit_ready_candidates": 0,
                "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
                "feed_report": "artifacts/latest/bang_tin_alpha.md",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            stop_file = Path(temp_dir) / "DUNG_LOOP"
            status_file = Path(temp_dir) / "loop_status.json"
            with patch(
                "sys.argv",
                [
                    "orchestrator_loop.py",
                    "--scoring",
                    "internal",
                    "--max-rounds",
                    "3",
                    "--interval-minutes",
                    "0",
                    "--stagnation-no-submit-ready-rounds",
                    "2",
                    "--stagnation-window",
                    "0",
                    "--stop-file",
                    str(stop_file),
                    "--status-file",
                    str(status_file),
                ],
            ):
                exit_code = orchestrator_loop.main()
            payload = json.loads(status_file.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_pipeline_mock.call_count, 3)
        third_args = run_pipeline_mock.call_args_list[2].args[0]
        self.assertEqual(third_args.adaptive_controls["mode"], "stagnation_recovery")
        self.assertEqual(third_args.source_bonus_adjustments["scout"], 8.0)
        self.assertEqual(third_args.source_quota_profile["planner"], 0.3)
        self.assertEqual(third_args.source_quota_profile["auto_fix_rewrite"], 0.3)
        self.assertEqual(third_args.source_quota_profile["scout"], 0.4)
        self.assertTrue(payload["stagnation"]["active"])
        self.assertIn("consecutive_no_submit_ready", payload["stagnation"]["reason_codes"])

    @patch("scripts.orchestrator_loop.run_pipeline")
    def test_loop_enables_recovery_mode_when_rolling_pass_rate_is_too_low(self, run_pipeline_mock):
        run_pipeline_mock.side_effect = [
            {
                "run_id": "light_20260328_000011",
                "queue_candidates": 6,
                "evaluated_candidates": 20,
                "submit_ready_candidates": 1,
                "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
                "feed_report": "artifacts/latest/bang_tin_alpha.md",
            },
            {
                "run_id": "light_20260328_000012",
                "queue_candidates": 6,
                "evaluated_candidates": 20,
                "submit_ready_candidates": 1,
                "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
                "feed_report": "artifacts/latest/bang_tin_alpha.md",
            },
            {
                "run_id": "light_20260328_000013",
                "queue_candidates": 6,
                "evaluated_candidates": 20,
                "submit_ready_candidates": 1,
                "daily_report": "artifacts/latest/alpha_tot_nhat_hom_nay.md",
                "feed_report": "artifacts/latest/bang_tin_alpha.md",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            stop_file = Path(temp_dir) / "DUNG_LOOP"
            status_file = Path(temp_dir) / "loop_status.json"
            with patch(
                "sys.argv",
                [
                    "orchestrator_loop.py",
                    "--scoring",
                    "internal",
                    "--max-rounds",
                    "3",
                    "--interval-minutes",
                    "0",
                    "--stagnation-no-submit-ready-rounds",
                    "0",
                    "--stagnation-window",
                    "2",
                    "--stagnation-min-pass-rate",
                    "0.08",
                    "--stop-file",
                    str(stop_file),
                    "--status-file",
                    str(status_file),
                ],
            ):
                exit_code = orchestrator_loop.main()
            payload = json.loads(status_file.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_pipeline_mock.call_count, 3)
        third_args = run_pipeline_mock.call_args_list[2].args[0]
        self.assertEqual(third_args.adaptive_controls["mode"], "stagnation_recovery")
        self.assertIn("low_pass_rate", payload["stagnation"]["reason_codes"])
