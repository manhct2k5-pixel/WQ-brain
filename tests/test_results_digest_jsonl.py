import json
import tempfile
import unittest
from pathlib import Path

from scripts.results_digest import read_result_rows


class TestResultsDigestJsonl(unittest.TestCase):
    def test_read_result_rows_supports_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "simulation_results.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "alpha_id": "LOCAL-123",
                        "regular_code": "rank(close)",
                        "fitness": 1.2,
                        "sharpe": 1.5,
                        "turnover": 0.18,
                        "LOW_SHARPE": "PASS",
                        "LOW_FITNESS": "PASS",
                        "LOW_TURNOVER": "PASS",
                        "HIGH_TURNOVER": "PASS",
                        "CONCENTRATED_WEIGHT": "PASS",
                        "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                        "SELF_CORRELATION": "PASS",
                        "MATCHES_COMPETITION": "PASS",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = read_result_rows(str(path))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["alpha_id"], "LOCAL-123")


if __name__ == "__main__":
    unittest.main()
