import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.research_cycle import inspect_run_output, run_subprocess, should_stop_after_round, write_status


class TestResearchCycle(unittest.TestCase):
    def test_inspect_run_output_detects_auth_and_rate_limit(self):
        output = "Status Code: 429\nSIMULATION_LIMIT_EXCEEDED\nStatus Code: 401\nINVALID_CREDENTIALS"
        events = inspect_run_output(output)
        self.assertTrue(events["auth_failed"])
        self.assertGreaterEqual(events["rate_limit_events"], 2)

    def test_should_stop_after_round_handles_pending_backlog_and_no_improvement(self):
        pending_stop = should_stop_after_round(
            before_summary={"rows": 10, "pending_count": 0, "best_research_score": 0.8},
            after_summary={"rows": 14, "pending_count": 3, "best_research_score": 1.0},
            round_events={"auth_failed": False, "rate_limit_events": 0},
            max_pending_ratio=0.35,
        )
        self.assertEqual(pending_stop, "pending_backlog")

        no_improvement_stop = should_stop_after_round(
            before_summary={"rows": 10, "pending_count": 0, "best_research_score": 0.8},
            after_summary={"rows": 12, "pending_count": 0, "best_research_score": 0.8},
            round_events={"auth_failed": False, "rate_limit_events": 0},
            max_pending_ratio=0.35,
        )
        self.assertEqual(no_improvement_stop, "no_improvement")

    def test_write_status_writes_json_artifact(self):
        payload = {
            "profile": "light",
            "rounds_completed": 1,
            "new_rows": 4,
            "best_research_score": 1.23,
            "stop_reason": "completed",
            "next_action": "Review artifacts/lo_tiep_theo.json",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trang_thai_chay.json"
            write_status(payload, output_path)
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], 1)
            self.assertEqual(written["profile"], "light")
            self.assertEqual(written["stop_reason"], "completed")

    def test_run_subprocess_returns_timeout_result_when_child_exceeds_deadline(self):
        timeout_error = subprocess.TimeoutExpired(
            cmd=["python", "slow.py"],
            timeout=5,
            output="partial stdout\n",
        )

        with patch("scripts.research_cycle.subprocess.run", side_effect=timeout_error):
            result = run_subprocess(["python", "slow.py"], cwd=Path("."), timeout=5)

        self.assertEqual(result.returncode, 124)
        self.assertIn("partial stdout", result.stdout)
        self.assertIn("timed out after 5 seconds", result.stdout)
        self.assertEqual(result.stderr, "timeout")


if __name__ == "__main__":
    unittest.main()
