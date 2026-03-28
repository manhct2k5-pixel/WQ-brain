import unittest

from scripts.build_evaluated_pool import build_evaluated_pool


class TestBuildEvaluatedPool(unittest.TestCase):
    def test_build_evaluated_pool_counts_submit_ready_candidates(self):
        records = [
            {
                "source": "planner",
                "thesis": "Residual or de-beta structure",
                "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.82,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 78.0,
                    "sharpe": 1.92,
                    "fitness": 1.66,
                },
                "evaluation_status": "COMPLETED",
            },
            {
                "source": "scout",
                "thesis": "VWAP dislocation",
                "expression": "rank(ts_zscore(abs(close-vwap),21))",
                "seed_ready": True,
                "qualified": False,
                "quality_label": "watchlist",
                "confidence_score": 0.34,
                "local_metrics": {
                    "verdict": "FAIL",
                    "alpha_score": 24.0,
                    "sharpe": 0.4,
                    "fitness": 0.2,
                },
                "evaluation_status": "COMPLETED",
            },
        ]

        payload = build_evaluated_pool(records, memory={}, results_summary={})

        self.assertEqual(payload["summary"]["candidate_count"], 2)
        self.assertEqual(payload["summary"]["submit_ready_count"], 1)
        self.assertEqual(payload["candidates"][0]["thesis"], "Residual or de-beta structure")

    def test_build_evaluated_pool_recomputes_quality_from_evaluated_metrics(self):
        records = [
            {
                "source": "auto_fix_rewrite",
                "thesis": "Auto-fix rewrite [VWAP dislocation]",
                "expression": "rank(winsorize(ts_zscore(abs(close-vwap),21),std=5))",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.95,
                "local_metrics": {
                    "verdict": "FAIL",
                    "alpha_score": 34.3,
                    "sharpe": 0.56,
                    "fitness": 0.4,
                },
                "evaluation_status": "COMPLETED",
            }
        ]

        payload = build_evaluated_pool(records, memory={}, results_summary={})
        candidate = payload["candidates"][0]

        self.assertFalse(candidate["evaluated_submit_ready"])
        self.assertFalse(candidate["qualified"])
        self.assertEqual(candidate["quality_label"], "watchlist")
        self.assertIn("verdict=FAIL", candidate["quality_fail_reasons"])
        self.assertEqual(candidate["lineage"]["stage_results"]["planning"]["quality_label"], "qualified")
        self.assertFalse(candidate["lineage"]["stage_results"]["evaluation"]["submit_ready"])
        self.assertIn("verdict=FAIL", candidate["lineage"]["stage_results"]["evaluation"]["fail_reasons"])

    def test_build_evaluated_pool_can_promote_candidate_that_only_failed_old_planner_flag(self):
        records = [
            {
                "run_id": "resume_demo",
                "batch_id": "batch_123",
                "candidate_id": "cand_123",
                "source": "planner",
                "thesis": "Price-volume divergence",
                "expression": "rank(winsorize(ts_corr(ts_rank(volume,21),ts_rank(close,21),21),std=5))",
                "seed_ready": True,
                "qualified": False,
                "quality_label": "watchlist",
                "confidence_score": 0.8452,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 72.3,
                    "sharpe": 1.79,
                    "fitness": 1.64,
                },
                "evaluation_status": "COMPLETED",
            }
        ]

        payload = build_evaluated_pool(records, memory={}, results_summary={})
        candidate = payload["candidates"][0]

        self.assertTrue(candidate["evaluated_submit_ready"])
        self.assertTrue(candidate["qualified"])
        self.assertEqual(candidate["quality_label"], "qualified")
        self.assertEqual(candidate["quality_fail_reasons"], [])
        self.assertEqual(candidate["run_id"], "resume_demo")
        self.assertEqual(candidate["batch_id"], "batch_123")
        self.assertEqual(candidate["candidate_id"], "cand_123")

    def test_build_evaluated_pool_preserves_candidate_signatures_and_duplicate_metadata(self):
        records = [
            {
                "source": "planner",
                "thesis": "Residual or de-beta structure",
                "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "compiled_expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "normalized_expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "normalized_compiled_expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "expression_skeleton": "rank(ts_regression(close,beta_last_N_days_spy,N,lag=N,rettype=N))",
                "candidate_signature": "cand_sig",
                "structure_signature": "struct_sig",
                "expression_signature": "expr_sig",
                "skeleton_signature": "skel_sig",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.82,
                "recent_failure_penalty": 7.0,
                "recent_failure_reasons": ["recent_failed_skeleton_match"],
                "recent_failure_match_count": 2,
                "dedupe_match_types": ["skeleton"],
                "duplicate_candidate_count": 2,
                "merged_candidate_signatures": ["cand_sig", "other_sig"],
                "lineage": {
                    "origin": "planner",
                    "sources": ["planner"],
                    "hypothesis_id": "residual_beta",
                    "family": "residual_beta",
                    "stage_results": {
                        "planning": {
                            "qualified": True,
                            "quality_label": "qualified",
                            "fail_reasons": [],
                        }
                    },
                },
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 78.0,
                    "sharpe": 1.92,
                    "fitness": 1.66,
                },
                "evaluation_status": "COMPLETED",
            }
        ]

        payload = build_evaluated_pool(records, memory={}, results_summary={})
        candidate = payload["candidates"][0]

        self.assertEqual(candidate["candidate_signature"], "cand_sig")
        self.assertEqual(candidate["structure_signature"], "struct_sig")
        self.assertEqual(candidate["expression_signature"], "expr_sig")
        self.assertEqual(candidate["skeleton_signature"], "skel_sig")
        self.assertEqual(candidate["recent_failure_reasons"], ["recent_failed_skeleton_match"])
        self.assertEqual(candidate["dedupe_match_types"], ["skeleton"])
        self.assertEqual(candidate["duplicate_candidate_count"], 2)
        self.assertEqual(candidate["lineage"]["origin"], "planner")
        self.assertTrue(candidate["lineage"]["stage_results"]["evaluation"]["submit_ready"])


if __name__ == "__main__":
    unittest.main()
