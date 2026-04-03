import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.simulate_batch import _resolve_worldquant_credentials, evaluate_queue


class TestSimulateBatch(unittest.TestCase):
    def test_resolve_worldquant_credentials_loads_dotenv_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("USERNAME=test_user\nPASSWORD=test_pass\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                username, password = _resolve_worldquant_credentials(env_path)

        self.assertEqual(username, "test_user")
        self.assertEqual(password, "test_pass")

    @patch("scripts.simulate_batch.save_alpha_to_csv")
    @patch("scripts.simulate_batch.score_expressions_batch")
    def test_evaluate_queue_skips_lower_priority_candidates_when_local_score_limit_applies(self, score_batch_mock, save_csv_mock):
        score_batch_mock.return_value = (
            [
                {
                    "expression": "rank(ts_mean(returns,21))",
                    "verdict": "PASS",
                    "alpha_score": 78.0,
                    "sharpe": 1.8,
                    "fitness": 1.5,
                    "turnover": 0.22,
                    "settings": {"label": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market"},
                }
            ],
            {
                "task_count": 1,
                "worker_count": 1,
                "mode": "sequential",
                "prepare_seconds": 0.0,
                "scoring_seconds": 0.01,
                "total_seconds": 0.01,
                "bottleneck_hint": "single_process_cpu",
            },
        )
        queue_payload = {
            "candidates": [
                {
                    "run_id": "run_demo",
                    "batch_id": "batch_demo",
                    "candidate_id": "cand_top",
                    "source": "planner",
                    "thesis": "Top candidate",
                    "thesis_id": "residual_beta",
                    "why": "top score",
                    "expression": "rank(ts_mean(returns,21))",
                    "compiled_expression": "rank(ts_mean(returns,21))",
                    "queue_rank": 1,
                    "priority_score": 91.0,
                    "token_program": ["RETURNS", "TS_MEAN_21", "RANK"],
                    "seed_ready": True,
                    "qualified": True,
                    "quality_label": "qualified",
                    "quality_fail_reasons": [],
                    "settings": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market",
                },
                {
                    "run_id": "run_demo",
                    "batch_id": "batch_demo",
                    "candidate_id": "cand_skip",
                    "source": "scout",
                    "thesis": "Overflow candidate",
                    "thesis_id": "technical_indicator",
                    "why": "lower score",
                    "expression": "rank(ts_zscore(abs(close-vwap),21))",
                    "compiled_expression": "rank(ts_zscore(abs(close-vwap),21))",
                    "queue_rank": 2,
                    "priority_score": 63.0,
                    "token_program": ["CLOSE", "VWAP", "SUB", "ABS", "TS_ZSCORE_21", "RANK"],
                    "seed_ready": False,
                    "qualified": False,
                    "quality_label": "watchlist",
                    "quality_fail_reasons": ["planner_watchlist"],
                    "settings": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market",
                },
            ]
        }

        records = evaluate_queue(
            queue_payload,
            backend="internal",
            local_score_limit=1,
            max_local_score_workers=1,
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["candidate_id"], "cand_top")
        self.assertEqual(records[0]["evaluation_status"], "COMPLETED")
        self.assertEqual(records[0]["local_metrics"]["alpha_score"], 78.0)
        self.assertEqual(records[1]["candidate_id"], "cand_skip")
        self.assertEqual(records[1]["evaluation_status"], "SKIPPED_LOCAL_SCORE_LIMIT")
        self.assertTrue(records[1]["local_scoring_skipped"])
        self.assertEqual(records[1]["local_scoring_skip_reason"], "local_score_limit")
        self.assertEqual(score_batch_mock.call_args.kwargs["max_workers"], 1)
        self.assertEqual(score_batch_mock.call_args.kwargs["min_parallel_tasks"], 4)
        self.assertEqual(save_csv_mock.call_count, 1)
        self.assertEqual(evaluate_queue.last_local_scoring_profile["candidate_count"], 2)
        self.assertEqual(evaluate_queue.last_local_scoring_profile["scored_candidates"], 1)
        self.assertEqual(evaluate_queue.last_local_scoring_profile["skipped_candidates"], 1)
        self.assertTrue(evaluate_queue.last_local_scoring_profile["limit_applied"])


if __name__ == "__main__":
    unittest.main()
