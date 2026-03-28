import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from scripts.cleanup_artifacts import cleanup_artifacts


def _iso_days_ago(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")


class TestCleanupArtifacts(unittest.TestCase):
    def test_cleanup_migrates_legacy_runs_archives_old_runs_and_compresses_large_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_root = Path(tmpdir) / "artifacts"
            legacy_run = artifacts_root / "runs" / "legacy_old"
            legacy_run.mkdir(parents=True, exist_ok=True)
            (legacy_run / "orchestrator_summary.json").write_text(
                '{"generated_at": "%s", "submit_ready_candidates": 0}' % _iso_days_ago(10),
                encoding="utf-8",
            )
            (legacy_run / "simulation_results.jsonl").write_text('{"value": 1}\n' * 8, encoding="utf-8")
            (legacy_run / "scores.csv").write_text("a,b\n1,2\n" * 8, encoding="utf-8")

            summary = cleanup_artifacts(
                artifacts_root=artifacts_root,
                recent_days=7,
                archive_delete_days=30,
                important_archive_delete_days=60,
                compress_min_bytes=1,
            )

            archived_run = artifacts_root / "archive" / "legacy_old"
            self.assertEqual(len(summary["migrated_legacy_runs"]), 1)
            self.assertEqual(len(summary["archived_runs"]), 1)
            self.assertTrue(archived_run.exists())
            self.assertFalse((artifacts_root / "recent_runs" / "legacy_old").exists())
            self.assertTrue((archived_run / "simulation_results.jsonl.gz").exists())
            self.assertTrue((archived_run / "scores.csv.gz").exists())
            self.assertFalse((archived_run / "simulation_results.jsonl").exists())
            self.assertFalse((archived_run / "scores.csv").exists())
            self.assertTrue((artifacts_root / "state" / "artifact_cleanup_status.json").exists())

    def test_cleanup_keeps_important_archives_longer_than_standard_archives(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_root = Path(tmpdir) / "artifacts"
            archive_dir = artifacts_root / "archive"
            standard_run = archive_dir / "standard_old"
            important_run = archive_dir / "important_old"
            standard_run.mkdir(parents=True, exist_ok=True)
            important_run.mkdir(parents=True, exist_ok=True)
            old_ts = (datetime.now() - timedelta(days=40)).timestamp()
            (standard_run / "archive_metadata.json").write_text(
                '{"original_last_updated_ts": %s}' % old_ts,
                encoding="utf-8",
            )
            (important_run / "archive_metadata.json").write_text(
                '{"original_last_updated_ts": %s}' % old_ts,
                encoding="utf-8",
            )
            (important_run / "retention_tag.json").write_text(
                '{"tag": "important", "set_at": "%s"}' % _iso_days_ago(40),
                encoding="utf-8",
            )

            cleanup_artifacts(
                artifacts_root=artifacts_root,
                recent_days=7,
                archive_delete_days=30,
                important_archive_delete_days=60,
                compress_min_bytes=1,
            )
            self.assertFalse(standard_run.exists())
            self.assertTrue(important_run.exists())

    def test_cleanup_keeps_recent_runs_under_recent_days(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_root = Path(tmpdir) / "artifacts"
            recent_run = artifacts_root / "recent_runs" / "recent_ok"
            recent_run.mkdir(parents=True, exist_ok=True)
            (recent_run / "orchestrator_summary.json").write_text(
                '{"generated_at": "%s", "submit_ready_candidates": 0}' % _iso_days_ago(2),
                encoding="utf-8",
            )

            summary = cleanup_artifacts(
                artifacts_root=artifacts_root,
                recent_days=7,
                archive_delete_days=30,
                important_archive_delete_days=60,
                compress_min_bytes=1,
            )
            self.assertTrue(recent_run.exists())
            self.assertEqual(len(summary["archived_runs"]), 0)
            self.assertEqual(len(summary["kept_recent_runs"]), 1)

    def test_cleanup_removes_stale_temp_files_and_reports_footprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_root = Path(tmpdir) / "artifacts"
            recent_run = artifacts_root / "recent_runs" / "recent_ok"
            recent_run.mkdir(parents=True, exist_ok=True)
            temp_file = recent_run / ".orchestrator_summary.json.deadbeef.tmp"
            temp_file.write_text("stale temp", encoding="utf-8")
            (recent_run / "orchestrator_summary.json").write_text(
                '{"generated_at": "%s", "submit_ready_candidates": 0}' % _iso_days_ago(1),
                encoding="utf-8",
            )

            summary = cleanup_artifacts(
                artifacts_root=artifacts_root,
                recent_days=7,
                archive_delete_days=30,
                important_archive_delete_days=60,
                compress_min_bytes=1,
                temp_file_max_age_hours=0,
            )

            self.assertFalse(temp_file.exists())
            self.assertEqual(len(summary["temp_files_removed"]), 1)
            self.assertGreater(summary["artifact_footprint"]["total"]["bytes"], 0)
            self.assertEqual(summary["artifact_footprint"]["recent_run_count"], 1)

    def test_cleanup_archives_oldest_recent_runs_when_max_recent_runs_is_exceeded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_root = Path(tmpdir) / "artifacts"
            newer_run = artifacts_root / "recent_runs" / "newer_run"
            older_run = artifacts_root / "recent_runs" / "older_run"
            newer_run.mkdir(parents=True, exist_ok=True)
            older_run.mkdir(parents=True, exist_ok=True)
            (newer_run / "orchestrator_summary.json").write_text(
                '{"generated_at": "%s", "submit_ready_candidates": 0}' % _iso_days_ago(1),
                encoding="utf-8",
            )
            (older_run / "orchestrator_summary.json").write_text(
                '{"generated_at": "%s", "submit_ready_candidates": 0}' % _iso_days_ago(2),
                encoding="utf-8",
            )

            summary = cleanup_artifacts(
                artifacts_root=artifacts_root,
                recent_days=30,
                archive_delete_days=30,
                important_archive_delete_days=60,
                compress_min_bytes=1,
                max_recent_runs=1,
            )

            self.assertTrue(newer_run.exists())
            self.assertFalse(older_run.exists())
            self.assertTrue((artifacts_root / "archive" / "older_run").exists())
            self.assertEqual(len(summary["kept_recent_runs"]), 1)
            self.assertEqual(len(summary["archived_runs"]), 1)
            self.assertEqual(summary["archived_runs"][0]["archive_reason"], "max_recent_runs")


if __name__ == "__main__":
    unittest.main()
