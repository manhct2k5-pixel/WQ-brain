import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import scout_loop


class TestScoutLoop(unittest.TestCase):
    @patch("scripts.scout_loop._load_payload", return_value={"selected": []})
    @patch("scripts.scout_loop.subprocess.run", return_value=SimpleNamespace(returncode=0))
    def test_loop_adds_feedback_health_gate_by_default(self, run_mock, _load_payload_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "scout_loop_status.json"
            with patch(
                "sys.argv",
                ["scout_loop.py", "--max-rounds", "1", "--count", "4", "--status-file", str(status_file)],
            ):
                exit_code = scout_loop.main()

        self.assertEqual(exit_code, 0)
        command = run_mock.call_args.args[0]
        self.assertIn("--require-feedback-healthy", command)
        self.assertIn("--count", command)

    @patch("scripts.scout_loop._load_payload", return_value={"selected": []})
    @patch("scripts.scout_loop.subprocess.run", return_value=SimpleNamespace(returncode=0))
    def test_loop_can_allow_degraded_feedback_explicitly(self, run_mock, _load_payload_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "scout_loop_status.json"
            with patch(
                "sys.argv",
                ["scout_loop.py", "--max-rounds", "1", "--allow-degraded-feedback", "--count", "4", "--status-file", str(status_file)],
            ):
                exit_code = scout_loop.main()

        self.assertEqual(exit_code, 0)
        command = run_mock.call_args.args[0]
        self.assertNotIn("--require-feedback-healthy", command)

    @patch("scripts.scout_loop._load_payload", return_value={"selected": [{"selection_mode": "strict"}]})
    @patch("scripts.scout_loop.subprocess.run", return_value=SimpleNamespace(returncode=0))
    def test_loop_can_ignore_strict_pick_stop_when_manual_stop_only(self, run_mock, _load_payload_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "scout_loop_status.json"
            with patch("builtins.print") as print_mock, patch(
                "sys.argv",
                ["scout_loop.py", "--max-rounds", "1", "--manual-stop-only", "--count", "4", "--status-file", str(status_file)],
            ):
                exit_code = scout_loop.main()

        self.assertEqual(exit_code, 0)
        printed = "\n".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list)
        self.assertIn("manual-stop-only is enabled", printed)
        self.assertNotIn("Found enough strict picks", printed)

    @patch("scripts.scout_loop.subprocess.run", return_value=SimpleNamespace(returncode=0))
    def test_loop_warns_and_skips_stop_checks_when_plan_json_is_invalid(self, _run_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "du_lieu.json"
            status_file = Path(tmpdir) / "scout_loop_status.json"
            plan_path.write_text('{"selected":', encoding="utf-8")

            with patch("builtins.print") as print_mock, patch(
                "sys.argv",
                ["scout_loop.py", "--max-rounds", "1", "--plan-path", str(plan_path), "--status-file", str(status_file)],
            ):
                exit_code = scout_loop.main()

        self.assertEqual(exit_code, 0)
        printed = "\n".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list)
        self.assertIn("Warning", printed)
        self.assertIn("Skipping stop-condition checks for this round", printed)

    @patch("scripts.scout_loop._load_payload", return_value={"selected": []})
    @patch("scripts.scout_loop.subprocess.run", return_value=SimpleNamespace(returncode=0))
    def test_loop_writes_status_file_with_pid(self, _run_mock, _load_payload_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "scout_loop_status.json"
            with patch(
                "sys.argv",
                ["scout_loop.py", "--max-rounds", "1", "--status-file", str(status_file)],
            ):
                exit_code = scout_loop.main()
            payload = json.loads(status_file.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["state"], "stopped")
        self.assertEqual(payload["reason"], "max_rounds")
        self.assertIsInstance(payload["pid"], int)

    @patch("scripts.scout_loop._load_payload", return_value={"selected": []})
    @patch(
        "scripts.scout_loop.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="child stdout line\n", stderr="child stderr line\n"),
    )
    def test_loop_writes_system_and_error_logs(self, _run_mock, _load_payload_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "scout_loop_status.json"
            with patch(
                "sys.argv",
                ["scout_loop.py", "--max-rounds", "1", "--status-file", str(status_file)],
            ):
                exit_code = scout_loop.main()
            system_log = (Path(tmpdir) / "system.log").read_text(encoding="utf-8")
            error_log = (Path(tmpdir) / "error.log").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("child stdout line", system_log)
        self.assertIn("child stderr line", error_log)
        self.assertIn("run_id=scout_loop_", system_log)

    @patch("scripts.scout_loop._load_payload", return_value={"selected": []})
    @patch(
        "scripts.scout_loop.subprocess.run",
        side_effect=subprocess.TimeoutExpired(
            cmd=["python", "scripts/scout_ideas.py"],
            timeout=7,
            output="partial scout stdout\n",
            stderr="partial scout stderr\n",
        ),
    )
    def test_loop_stops_when_scout_subprocess_times_out(self, _run_mock, _load_payload_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "scout_loop_status.json"
            with patch(
                "sys.argv",
                ["scout_loop.py", "--max-rounds", "1", "--subprocess-timeout", "7", "--status-file", str(status_file)],
            ):
                exit_code = scout_loop.main()
            payload = json.loads(status_file.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 124)
        self.assertEqual(payload["state"], "stopped")
        self.assertEqual(payload["reason"], "scout_timeout")
        self.assertIn("timeout=7", payload["error"])
