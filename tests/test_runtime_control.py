import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import runtime_control


class TestRuntimeControl(unittest.TestCase):
    def test_request_stop_file_uses_status_pid_and_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stop_file = Path(tmpdir) / "DUNG_LOOP"
            status_file = Path(tmpdir) / "status.json"
            status_file.write_text(
                json.dumps(
                    {
                        "pid": 43210,
                        "run_id": "light_20260328_010000_r001",
                    }
                ),
                encoding="utf-8",
            )

            with patch("scripts.runtime_control.pid_is_alive", return_value=True):
                payload = runtime_control.request_stop_file(stop_file, status_file=status_file)

            saved = json.loads(stop_file.read_text(encoding="utf-8"))

        self.assertEqual(payload["target_pid"], 43210)
        self.assertEqual(payload["run_id"], "light_20260328_010000_r001")
        self.assertEqual(saved["target_pid"], 43210)
        self.assertEqual(saved["reason"], "user_requested_stop")

    def test_clear_stale_stop_file_removes_dead_target_pid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stop_file = Path(tmpdir) / "DUNG_LOOP"
            stop_file.write_text(
                json.dumps(
                    {
                        "kind": "stop_request",
                        "pid": 111,
                        "target_pid": 222,
                        "timestamp": "2026-03-28T01:00:00",
                        "run_id": "loop_demo",
                        "reason": "user_requested_stop",
                    }
                ),
                encoding="utf-8",
            )
            warnings = []

            with patch("scripts.runtime_control.pid_is_alive", return_value=False):
                removed = runtime_control.clear_stale_stop_file(stop_file, warn=warnings.append)

            self.assertTrue(removed)
            self.assertFalse(stop_file.exists())
            self.assertTrue(any("Removed stale stop file" in message for message in warnings))

    def test_clear_stale_lock_files_removes_dead_pid_and_legacy_empty_locks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dead_lock = root / ".dead.lock"
            dead_lock.write_text(
                json.dumps(
                    {
                        "kind": "file_lock",
                        "pid": 123,
                        "timestamp": "2026-03-28T01:00:00",
                    }
                ),
                encoding="utf-8",
            )
            legacy_lock = root / ".legacy.lock"
            legacy_lock.write_text("", encoding="utf-8")
            live_lock = root / ".live.lock"
            live_lock.write_text(
                json.dumps(
                    {
                        "kind": "file_lock",
                        "pid": 456,
                        "timestamp": "2026-03-28T01:00:00",
                    }
                ),
                encoding="utf-8",
            )

            def fake_pid_is_alive(pid):
                return int(pid) == 456

            with patch("scripts.runtime_control.pid_is_alive", side_effect=fake_pid_is_alive):
                removed = runtime_control.clear_stale_lock_files([root])

            self.assertEqual(len(removed), 2)
            self.assertFalse(dead_lock.exists())
            self.assertFalse(legacy_lock.exists())
            self.assertTrue(live_lock.exists())


if __name__ == "__main__":
    unittest.main()
