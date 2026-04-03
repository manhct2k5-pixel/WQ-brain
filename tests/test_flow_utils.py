import json
import tempfile
import unittest
from pathlib import Path

from scripts import flow_utils


class TestFlowUtils(unittest.TestCase):
    def test_load_json_returns_default_and_warns_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "broken.json"
            path.write_text('{"broken":', encoding="utf-8")
            warnings = []

            payload = flow_utils.load_json(
                path,
                default={},
                context="broken payload",
                warn=warnings.append,
            )

        self.assertEqual(payload, {})
        self.assertTrue(any("broken payload" in message for message in warnings))
        self.assertTrue(any("invalid JSON" in message for message in warnings))

    def test_load_json_validates_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.json"
            flow_utils.atomic_write_json(path, {"selected": []})
            warnings = []

            payload = flow_utils.load_json(
                path,
                default={},
                required_fields=("batch", "selected"),
                context="scout payload",
                warn=warnings.append,
            )

        self.assertEqual(payload, {})
        self.assertTrue(any("missing required fields: batch" in message for message in warnings))

    def test_atomic_write_json_round_trips_object_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "payload.json"
            payload = {"batch": {"qualified_count": 2}, "selected": [{"expression": "rank(close)"}]}

            flow_utils.atomic_write_json(path, payload)

            reloaded = flow_utils.load_json(
                path,
                default={},
                required_fields=("batch", "selected"),
                context="roundtrip payload",
            )

        self.assertEqual(reloaded, payload)

    def test_atomic_write_json_adds_schema_version_for_known_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "planned_candidates.json"
            payload = {"batch": {"candidates": []}}

            flow_utils.atomic_write_json(path, payload)

            written = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(written["schema_version"], flow_utils.JSON_ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(written["batch"], payload["batch"])

    def test_load_json_migrates_legacy_artifact_without_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evaluated_candidates.json"
            path.write_text(json.dumps({"candidates": [], "summary": {}}), encoding="utf-8")
            warnings = []

            payload = flow_utils.load_json(
                path,
                default={},
                context="legacy evaluated payload",
                warn=warnings.append,
            )

        self.assertEqual(payload["schema_version"], flow_utils.JSON_ARTIFACT_SCHEMA_VERSION)
        self.assertTrue(any("missing schema_version" in message for message in warnings))

    def test_load_json_rejects_unsupported_newer_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evaluated_candidates.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": flow_utils.JSON_ARTIFACT_SCHEMA_VERSION + 1,
                        "candidates": [],
                        "summary": {},
                    }
                ),
                encoding="utf-8",
            )
            warnings = []

            payload = flow_utils.load_json(
                path,
                default={"fallback": True},
                context="future evaluated payload",
                warn=warnings.append,
            )

        self.assertEqual(payload, {"fallback": True})
        self.assertTrue(any("unsupported schema_version" in message for message in warnings))

    def test_load_json_quarantines_invalid_artifact_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifacts" / "latest" / "evaluated_candidates.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"broken":', encoding="utf-8")
            warnings = []

            payload = flow_utils.load_json(
                path,
                default={},
                context="bad evaluated artifact",
                warn=warnings.append,
            )

            quarantine_dir = Path(tmpdir) / "artifacts" / "quarantine" / "files"
            quarantined_files = [item for item in quarantine_dir.glob("*.json") if not item.name.endswith(".meta.json")]
            metadata_files = list(quarantine_dir.glob("*.meta.json"))
            metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))

        self.assertEqual(payload, {})
        self.assertFalse(path.exists())
        self.assertEqual(len(quarantined_files), 1)
        self.assertEqual(len(metadata_files), 1)
        self.assertEqual(metadata["reason"], "invalid_json")
        self.assertEqual(metadata["action"], "moved")
        self.assertTrue(any("Quarantined file" in message for message in warnings))

    def test_read_jsonl_quarantines_invalid_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifacts" / "recent_runs" / "demo" / "simulation_results.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"value": 1}\n{"broken":\n', encoding="utf-8")

            records = flow_utils.read_jsonl(path)

            quarantine_dir = Path(tmpdir) / "artifacts" / "quarantine" / "jsonl_lines"
            quarantined = list(quarantine_dir.glob("*.json"))
            payload = json.loads(quarantined[0].read_text(encoding="utf-8"))

        self.assertEqual(records, [{"value": 1}])
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(payload["reason"], "invalid_jsonl_line")
        self.assertEqual(payload["payload"]["line_number"], 2)

    def test_copy_file_replaces_target_contents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.txt"
            target = Path(tmpdir) / "target.txt"
            source.write_text("fresh content", encoding="utf-8")
            target.write_text("stale content", encoding="utf-8")

            flow_utils.copy_file(source, target)

            self.assertEqual(target.read_text(encoding="utf-8"), "fresh content")

    def test_latest_publish_is_complete_checks_metadata_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            latest_dir = Path(tmpdir) / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            report_path = latest_dir / "alpha_tot_nhat_hom_nay.md"
            report_path.write_text("# Daily", encoding="utf-8")
            flow_utils.atomic_write_json(
                flow_utils.latest_metadata_path(latest_dir),
                {
                    "status": "complete",
                    "complete": True,
                    "run_id": "demo_run",
                    "batch_id": "batch_demo",
                    "artifacts": {
                        "daily_report": str(report_path),
                    },
                },
            )

            self.assertTrue(
                flow_utils.latest_publish_is_complete(
                    latest_dir=latest_dir,
                    run_id="demo_run",
                    batch_id="batch_demo",
                    required_artifact_names=("daily_report",),
                )
            )
            self.assertFalse(
                flow_utils.latest_publish_is_complete(
                    latest_dir=latest_dir,
                    run_id="other_run",
                    required_artifact_names=("daily_report",),
                )
            )

    def test_latest_publish_is_complete_returns_false_for_partial_latest_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            latest_dir = Path(tmpdir) / "artifacts" / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            report_path = latest_dir / "alpha_tot_nhat_hom_nay.md"
            report_path.write_text("# Daily", encoding="utf-8")
            metadata_path = flow_utils.latest_metadata_path(latest_dir)
            metadata_path.write_text('{"status": "complete"', encoding="utf-8")
            warnings = []

            is_complete = flow_utils.latest_publish_is_complete(
                latest_dir=latest_dir,
                run_id="demo_run",
                required_artifact_names=("daily_report",),
                warn=warnings.append,
            )

            quarantine_dir = Path(tmpdir) / "artifacts" / "quarantine" / "files"
            metadata_files = list(quarantine_dir.glob("*.meta.json"))
            metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))

        self.assertFalse(is_complete)
        self.assertEqual(len(metadata_files), 1)
        self.assertEqual(metadata["reason"], "invalid_json")
        self.assertTrue(any("invalid JSON" in message for message in warnings))

    def test_ensure_runtime_layout_creates_recent_runs_archive_and_state_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_runs_dir = flow_utils.RUNS_DIR
            original_recent_runs_dir = flow_utils.RECENT_RUNS_DIR
            original_archive_dir = flow_utils.ARCHIVE_DIR
            original_latest_dir = flow_utils.LATEST_DIR
            original_state_dir = flow_utils.STATE_DIR
            try:
                flow_utils.RECENT_RUNS_DIR = Path(tmpdir) / "artifacts" / "recent_runs"
                flow_utils.RUNS_DIR = flow_utils.RECENT_RUNS_DIR
                flow_utils.ARCHIVE_DIR = Path(tmpdir) / "artifacts" / "archive"
                flow_utils.LATEST_DIR = Path(tmpdir) / "artifacts" / "latest"
                flow_utils.STATE_DIR = Path(tmpdir) / "artifacts" / "state"

                run_dir = flow_utils.ensure_runtime_layout("demo_run")
                self.assertEqual(run_dir, Path(tmpdir) / "artifacts" / "recent_runs" / "demo_run")
                self.assertTrue((Path(tmpdir) / "artifacts" / "recent_runs").exists())
                self.assertTrue((Path(tmpdir) / "artifacts" / "archive").exists())
                self.assertTrue((Path(tmpdir) / "artifacts" / "latest").exists())
                self.assertTrue((Path(tmpdir) / "artifacts" / "state").exists())
            finally:
                flow_utils.RUNS_DIR = original_runs_dir
                flow_utils.RECENT_RUNS_DIR = original_recent_runs_dir
                flow_utils.ARCHIVE_DIR = original_archive_dir
                flow_utils.LATEST_DIR = original_latest_dir
                flow_utils.STATE_DIR = original_state_dir


if __name__ == "__main__":
    unittest.main()
