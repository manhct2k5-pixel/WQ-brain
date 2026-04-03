import json
import tempfile
import unittest
from pathlib import Path

from scripts.system_health import classify_api_failure, evaluate_orchestrator_loop_health


class TestSystemHealth(unittest.TestCase):
    def test_classify_api_failure_detects_worldquant_auth_errors(self):
        self.assertTrue(classify_api_failure("WorldQuant authentication failed in simulate_batch (status=401, INVALID_CREDENTIALS)."))
        self.assertTrue(classify_api_failure("Status Code: 429\nSIMULATION_LIMIT_EXCEEDED"))
        self.assertFalse(classify_api_failure("ValueError: malformed local config"))

    def test_evaluate_orchestrator_loop_health_escalates_api_streak_and_memory_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            latest_metadata = Path(tmpdir) / "latest_metadata.json"
            global_memory = Path(tmpdir) / "global_memory.json"
            latest_metadata.write_text(
                json.dumps({"status": "complete", "complete": True, "artifacts": {}}),
                encoding="utf-8",
            )
            global_memory.write_text("x" * 4096, encoding="utf-8")

            health = evaluate_orchestrator_loop_health(
                summary={
                    "latest_metadata": str(latest_metadata),
                    "global_memory_path": str(global_memory),
                },
                round_index=5,
                scoring_backend="worldquant",
                consecutive_api_failures=3,
                last_submit_ready_round=0,
                last_submit_ready_at=None,
                memory_warning_mb=0.001,
                memory_error_mb=0.002,
                memory_critical_mb=0.003,
                api_warning_streak=1,
                api_error_streak=2,
                api_critical_streak=4,
                no_pass_warning_rounds=3,
                no_pass_error_rounds=6,
                no_pass_critical_rounds=10,
            )

        self.assertEqual(health["highest_severity"], "critical")
        self.assertEqual(health["checks"]["memory_file_size"]["severity"], "critical")
        self.assertEqual(health["checks"]["api_error_streak"]["severity"], "error")
        self.assertEqual(health["checks"]["submit_ready_freshness"]["severity"], "warning")

    def test_evaluate_orchestrator_loop_health_warns_when_memory_payload_missing_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            latest_metadata = Path(tmpdir) / "latest_metadata.json"
            global_memory = Path(tmpdir) / "global_memory.json"
            latest_metadata.write_text(
                json.dumps({"status": "complete", "complete": True, "artifacts": {}}),
                encoding="utf-8",
            )
            global_memory.write_text(json.dumps({"working_memory": {}}), encoding="utf-8")

            health = evaluate_orchestrator_loop_health(
                summary={
                    "latest_metadata": str(latest_metadata),
                    "global_memory_path": str(global_memory),
                },
                round_index=0,
                scoring_backend="internal",
                consecutive_api_failures=0,
                last_submit_ready_round=0,
                last_submit_ready_at=None,
            )

        self.assertEqual(health["highest_severity"], "warning")
        self.assertEqual(health["checks"]["memory_payload_shape"]["severity"], "warning")
        self.assertIn("summary_memory", health["checks"]["memory_payload_shape"]["missing_fields"])

    def test_evaluate_orchestrator_loop_health_errors_when_memory_json_is_partial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            latest_metadata = Path(tmpdir) / "latest_metadata.json"
            global_memory = Path(tmpdir) / "global_memory.json"
            latest_metadata.write_text(
                json.dumps({"status": "complete", "complete": True, "artifacts": {}}),
                encoding="utf-8",
            )
            global_memory.write_text('{"working_memory":', encoding="utf-8")

            health = evaluate_orchestrator_loop_health(
                summary={
                    "latest_metadata": str(latest_metadata),
                    "global_memory_path": str(global_memory),
                },
                round_index=0,
                scoring_backend="internal",
                consecutive_api_failures=0,
                last_submit_ready_round=0,
                last_submit_ready_at=None,
            )

        self.assertEqual(health["highest_severity"], "error")
        self.assertEqual(health["checks"]["memory_payload_shape"]["severity"], "error")
        self.assertEqual(health["checks"]["memory_payload_shape"]["status"], "unreadable")


if __name__ == "__main__":
    unittest.main()
