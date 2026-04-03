import tempfile
import unittest
from pathlib import Path

from scripts.system_logging import configure_runtime_logging


class TestSystemLogging(unittest.TestCase):
    def test_configure_runtime_logging_rotates_system_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = configure_runtime_logging(
                "rotation_test",
                log_dir=tmpdir,
                run_id_getter=lambda: "run_demo",
                log_max_bytes=120,
                log_backup_count=1,
            )
            try:
                for _ in range(3):
                    bundle.logger.info("x" * 160)
            finally:
                bundle.close()

            self.assertTrue((Path(tmpdir) / "system.log").exists())
            self.assertTrue((Path(tmpdir) / "system.log.1").exists())


if __name__ == "__main__":
    unittest.main()
