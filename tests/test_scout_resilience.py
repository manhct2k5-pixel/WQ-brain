import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import scout_ideas


class TestScoutResilience(unittest.TestCase):
    def test_load_json_returns_empty_dict_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "broken.json"
            path.write_text('{"broken":', encoding="utf-8")

            self.assertEqual(scout_ideas.load_json(path), {})

    def test_save_json_round_trips_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "payload.json"
            payload = {"entries": {"alpha": {"score": 0.82}}, "status": "ok"}

            scout_ideas.save_json(path, payload)

            self.assertEqual(scout_ideas.load_json(path), payload)

    def test_main_feedback_health_failure_does_not_claim_archive_written_when_archive_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "trinh_sat_hang_ngay.md"
            plan_path = root / "du_lieu.json"
            batch_path = root / "bieu_thuc_da_chon.txt"
            memory_path = root / "bo_nho.json"
            brain_feedback_path = root / "phan_hoi_brain.json"
            learning_path = root / "kien_thuc_tu_hoc.json"
            fetch_cache_path = root / "cache_tim_kiem.json"
            report_state_path = root / "trang_thai_bao_cao.json"
            archive_root = root / "bao_cao_ngay"
            generic_memory_path = root / "bo_nho_nghien_cuu.json"
            submitted_alphas_path = root / "alpha_da_gui.json"
            history_path = root / "lich_su.jsonl"

            feedback_status = {
                "status": "schema_error",
                "health": "degraded",
                "hard_block": True,
                "reason": "missing_context_columns",
                "recommended_action": "upgrade_feedback_schema",
                "missing_context_columns": ["region", "universe"],
                "message": "Loaded 0 simulation rows. Missing context columns: region, universe.",
            }

            argv = [
                "scout_ideas.py",
                "--memory",
                str(generic_memory_path),
                "--memory-path",
                str(memory_path),
                "--history-path",
                str(history_path),
                "--submitted-alphas",
                str(submitted_alphas_path),
                "--output",
                str(output_path),
                "--write-plan",
                str(plan_path),
                "--write-batch",
                str(batch_path),
                "--write-brain-feedback",
                str(brain_feedback_path),
                "--write-learning",
                str(learning_path),
                "--fetch-cache",
                str(fetch_cache_path),
                "--report-state-path",
                str(report_state_path),
                "--archive-root",
                str(archive_root),
                "--require-feedback-healthy",
            ]

            with patch("sys.argv", argv), patch(
                "scripts.scout_ideas.load_memory",
                return_value={"style_leaders": [], "failure_counts": {}},
            ), patch(
                "scripts.scout_ideas.load_brain_feedback_rows",
                return_value=([], {"status": "schema_error", "missing_context_columns": ["region", "universe"]}),
            ), patch(
                "scripts.scout_ideas.assess_brain_feedback_health",
                return_value=feedback_status,
            ), patch(
                "scripts.scout_ideas.write_report_archive",
                side_effect=OSError("disk full"),
            ):
                exit_code = scout_ideas.main()

            self.assertEqual(exit_code, 2)
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["halt_reason"], "feedback_health_check")
            self.assertFalse(payload["archive_status"]["written"])
            self.assertEqual(payload["archive_status"]["reason"], "archive_write_failed")
            self.assertEqual(payload["archived_paths"], {})
            self.assertEqual(batch_path.read_text(encoding="utf-8"), "")
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
