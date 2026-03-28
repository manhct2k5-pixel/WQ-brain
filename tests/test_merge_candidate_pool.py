import unittest

from scripts.merge_candidate_pool import merge_candidate_pool


class TestMergeCandidatePool(unittest.TestCase):
    def test_merge_candidate_pool_dedupes_same_expression_and_preserves_sources(self):
        planner_candidates = [
            {
                "thesis": "Residual or de-beta structure",
                "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.74,
                "local_metrics": {
                    "alpha_score": 74.0,
                    "sharpe": 1.9,
                    "fitness": 1.7,
                },
            }
        ]
        auto_fix_candidates = [
            {
                "thesis": "Auto-fix rewrite [Residual]",
                "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.81,
                "local_metrics": {
                    "alpha_score": 78.0,
                    "sharpe": 2.0,
                    "fitness": 1.9,
                },
            }
        ]

        payload = merge_candidate_pool(
            planner_candidates=planner_candidates,
            auto_fix_candidates=auto_fix_candidates,
            scout_candidates=[],
        )

        self.assertEqual(payload["candidate_count"], 1)
        self.assertEqual(payload["candidates"][0]["source"], "auto_fix_rewrite")
        self.assertEqual(payload["candidates"][0]["source_stages"], ["auto_fix_rewrite", "planner"])
        self.assertEqual(payload["candidates"][0]["lineage"]["sources"], ["fix", "planner"])

    def test_merge_candidate_pool_keeps_parent_and_family_lineage_for_rewrites(self):
        auto_fix_candidates = [
            {
                "thesis": "Auto-fix rewrite [Technical Indicator]",
                "thesis_id": "technical_indicator",
                "family_id": "technical_indicator",
                "family": "Technical Indicator",
                "expression": "rank(ts_sum(close,10))",
                "source_expression": "rank(ts_zscore(abs(close-vwap),21))",
                "source_alpha_id": "A1",
                "repair_status": "submit_ready",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.81,
                "local_metrics": {
                    "alpha_score": 78.0,
                    "sharpe": 2.0,
                    "fitness": 1.8,
                },
            }
        ]

        payload = merge_candidate_pool(
            planner_candidates=[],
            auto_fix_candidates=auto_fix_candidates,
            scout_candidates=[],
        )

        self.assertEqual(payload["candidate_count"], 1)
        lineage = payload["candidates"][0]["lineage"]
        self.assertEqual(lineage["origin"], "fix")
        self.assertEqual(lineage["hypothesis_id"], "technical_indicator")
        self.assertEqual(lineage["family"], "technical_indicator")
        self.assertEqual(lineage["parents"][0]["alpha_id"], "A1")
        self.assertEqual(lineage["parents"][0]["expression"], "rank(ts_zscore(abs(close-vwap),21))")

    def test_merge_candidate_pool_filters_not_seed_ready_candidates(self):
        planner_candidates = [
            {
                "thesis": "Technical indicator blend",
                "expression": "rank(ts_sum(close,10))",
                "seed_ready": False,
                "qualified": False,
                "quality_label": "watchlist",
                "confidence_score": 0.9,
                "local_metrics": {
                    "alpha_score": 80.0,
                    "sharpe": 2.0,
                    "fitness": 2.0,
                },
            }
        ]

        payload = merge_candidate_pool(
            planner_candidates=planner_candidates,
            auto_fix_candidates=[],
            scout_candidates=[],
        )

        self.assertEqual(payload["candidate_count"], 0)
        self.assertEqual(payload["filtered_counts"], {"not_seed_ready": 1})

    def test_merge_candidate_pool_filters_stale_auto_fix_that_failed_recently(self):
        auto_fix_candidates = [
            {
                "thesis": "Auto-fix rewrite [VWAP dislocation]",
                "expression": "rank(winsorize(ts_zscore(abs(close-vwap),21),std=5))",
                "settings": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.95,
                "local_metrics": {
                    "alpha_score": 72.1,
                    "sharpe": 1.79,
                    "fitness": 1.73,
                },
            }
        ]
        prior_evaluated_candidates = [
            {
                "thesis": "Auto-fix rewrite [VWAP dislocation]",
                "expression": "rank(winsorize(ts_zscore(abs(close-vwap),21),std=5))",
                "settings": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market",
                "evaluated_submit_ready": False,
            }
        ]

        payload = merge_candidate_pool(
            planner_candidates=[],
            auto_fix_candidates=auto_fix_candidates,
            scout_candidates=[],
            prior_evaluated_candidates=prior_evaluated_candidates,
        )

        self.assertEqual(payload["candidate_count"], 0)
        self.assertEqual(payload["filtered_counts"], {"stale_auto_fix_failed_recently": 1})

    def test_merge_candidate_pool_can_boost_scout_priority_during_recovery(self):
        planner_candidates = [
            {
                "thesis": "Planner baseline",
                "expression": "rank(ts_sum(close,10))",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.72,
                "local_metrics": {
                    "alpha_score": 70.0,
                    "sharpe": 1.6,
                    "fitness": 1.5,
                },
            }
        ]
        scout_candidates = [
            {
                "thesis": "Scout recovery idea",
                "expression": "rank(ts_mean(high,10)-close)",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.69,
                "local_metrics": {
                    "alpha_score": 68.0,
                    "sharpe": 1.5,
                    "fitness": 1.4,
                },
            }
        ]

        payload = merge_candidate_pool(
            planner_candidates=planner_candidates,
            auto_fix_candidates=[],
            scout_candidates=scout_candidates,
            source_bonus_adjustments={"scout": 8.0},
        )

        self.assertEqual(payload["candidate_count"], 2)
        self.assertEqual(payload["candidates"][0]["source"], "scout")

    def test_merge_candidate_pool_balances_sources_with_soft_quotas(self):
        planner_expressions = [
            "rank(ts_sum(close,10))",
            "rank(ts_mean(close,13))",
            "rank(ts_delta(close,5))",
            "rank(close-vwap)",
        ]
        planner_candidates = [
            {
                "thesis": f"Planner {index}",
                "thesis_id": f"planner_family_{index}",
                "expression": expression,
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.78,
                "novelty_score": 0.45,
                "local_metrics": {
                    "alpha_score": 74.0 - index,
                    "sharpe": 1.7,
                    "fitness": 1.5,
                },
            }
            for index, expression in enumerate(planner_expressions)
        ]
        auto_fix_candidates = [
            {
                "thesis": "Fix candidate",
                "thesis_id": "technical_indicator",
                "expression": "rank(ts_mean(close,15))",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.77,
                "novelty_score": 0.82,
                "local_metrics": {
                    "alpha_score": 71.5,
                    "sharpe": 1.6,
                    "fitness": 1.45,
                },
            }
        ]
        scout_candidates = [
            {
                "thesis": "Scout candidate",
                "thesis_id": "shock_response",
                "expression": "rank(ts_mean(high,10)-close)",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.76,
                "novelty_score": 0.91,
                "local_metrics": {
                    "alpha_score": 71.0,
                    "sharpe": 1.55,
                    "fitness": 1.4,
                },
            }
        ]

        payload = merge_candidate_pool(
            planner_candidates=planner_candidates,
            auto_fix_candidates=auto_fix_candidates,
            scout_candidates=scout_candidates,
            limit=4,
            source_quota_profile={"planner": 0.5, "auto_fix_rewrite": 0.25, "scout": 0.25},
        )

        self.assertEqual(payload["candidate_count"], 4)
        self.assertEqual(payload["selected_source_counts"]["planner"], 2)
        self.assertEqual(payload["selected_source_counts"]["auto_fix_rewrite"], 1)
        self.assertEqual(payload["selected_source_counts"]["scout"], 1)

    def test_merge_candidate_pool_uses_source_history_in_selection_priority(self):
        planner_candidates = [
            {
                "thesis": "Planner retry",
                "expression": "rank(ts_sum(close,10))",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.80,
                "novelty_score": 0.52,
                "local_metrics": {
                    "alpha_score": 74.0,
                    "sharpe": 1.8,
                    "fitness": 1.6,
                },
            }
        ]
        auto_fix_candidates = [
            {
                "thesis": "Fix retry",
                "expression": "rank(ts_mean(close,10))",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.79,
                "novelty_score": 0.52,
                "local_metrics": {
                    "alpha_score": 73.0,
                    "sharpe": 1.8,
                    "fitness": 1.6,
                },
            }
        ]
        prior_evaluated_candidates = [
            {"source": "planner", "evaluated_submit_ready": False},
            {"source": "planner", "evaluated_submit_ready": False},
            {"source": "planner", "evaluated_submit_ready": False},
            {"source": "auto_fix_rewrite", "evaluated_submit_ready": True},
            {"source": "auto_fix_rewrite", "evaluated_submit_ready": True},
            {"source": "auto_fix_rewrite", "evaluated_submit_ready": True},
        ]

        payload = merge_candidate_pool(
            planner_candidates=planner_candidates,
            auto_fix_candidates=auto_fix_candidates,
            scout_candidates=[],
            prior_evaluated_candidates=prior_evaluated_candidates,
            source_quota_profile={"planner": 0.5, "auto_fix_rewrite": 0.5, "scout": 0.0},
        )

        self.assertEqual(payload["candidates"][0]["source"], "auto_fix_rewrite")
        self.assertEqual(payload["source_historical_pass_rates"]["planner"], 0.0)
        self.assertEqual(payload["source_historical_pass_rates"]["auto_fix_rewrite"], 1.0)
        self.assertGreater(payload["candidates"][0]["source_history_bonus"], 0.0)

    def test_merge_candidate_pool_filters_near_duplicate_skeletons_before_queue(self):
        planner_candidates = [
            {
                "thesis": "Momentum 10",
                "expression": "rank(ts_sum(close,10))",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.80,
                "local_metrics": {
                    "alpha_score": 71.0,
                    "sharpe": 1.7,
                    "fitness": 1.5,
                },
            },
            {
                "thesis": "Momentum 21",
                "expression": "rank(ts_sum(close,21))",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.78,
                "local_metrics": {
                    "alpha_score": 68.0,
                    "sharpe": 1.5,
                    "fitness": 1.3,
                },
            },
        ]

        payload = merge_candidate_pool(
            planner_candidates=planner_candidates,
            auto_fix_candidates=[],
            scout_candidates=[],
        )

        self.assertEqual(payload["candidate_count"], 1)
        self.assertEqual(payload["filtered_counts"], {"near_duplicate_skeleton_in_queue": 1})
        self.assertEqual(payload["candidates"][0]["expression"], "rank(ts_sum(close,10))")
        self.assertEqual(payload["candidates"][0]["dedupe_match_types"], ["skeleton"])
        self.assertEqual(payload["candidates"][0]["duplicate_candidate_count"], 2)

    def test_merge_candidate_pool_attaches_signatures_and_penalizes_recent_failed_duplicates(self):
        planner_candidates = [
            {
                "thesis": "Retry exact failed idea",
                "expression": "rank(ts_sum(close,10))",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.81,
                "local_metrics": {
                    "alpha_score": 74.0,
                    "sharpe": 1.8,
                    "fitness": 1.6,
                },
            },
            {
                "thesis": "Fresh idea",
                "expression": "rank(ts_mean(close,10))",
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "confidence_score": 0.78,
                "local_metrics": {
                    "alpha_score": 73.0,
                    "sharpe": 1.7,
                    "fitness": 1.5,
                },
            },
        ]
        prior_evaluated_candidates = [
            {
                "expression": "rank(ts_sum(close,10))",
                "compiled_expression": "rank(ts_sum(close,10))",
                "evaluated_submit_ready": False,
            }
        ]

        payload = merge_candidate_pool(
            planner_candidates=planner_candidates,
            auto_fix_candidates=[],
            scout_candidates=[],
            prior_evaluated_candidates=prior_evaluated_candidates,
        )

        self.assertEqual(payload["candidate_count"], 2)
        exact_retry = next(item for item in payload["candidates"] if item["expression"] == "rank(ts_sum(close,10))")
        fresh = next(item for item in payload["candidates"] if item["expression"] == "rank(ts_mean(close,10))")
        self.assertTrue(exact_retry["candidate_signature"])
        self.assertTrue(exact_retry["structure_signature"])
        self.assertTrue(exact_retry["skeleton_signature"])
        self.assertEqual(exact_retry["recent_failure_reasons"], ["recent_failed_exact_match"])
        self.assertGreater(exact_retry["recent_failure_penalty"], 0.0)
        self.assertLess(exact_retry["priority_score"], exact_retry["priority_score_before_recent_failures"])
        self.assertEqual(exact_retry["lineage"]["origin"], "planner")
        self.assertEqual(exact_retry["lineage"]["stage_results"]["planning"]["quality_label"], "qualified")
        self.assertEqual(fresh["recent_failure_reasons"], [])
        self.assertGreater(fresh["priority_score"], exact_retry["priority_score"])

    def test_merge_candidate_pool_quarantines_invalid_candidates_via_callback(self):
        quarantined = []
        payload = merge_candidate_pool(
            planner_candidates=[
                {
                    "thesis": "Broken planner candidate",
                    "seed_ready": True,
                    "qualified": True,
                }
            ],
            auto_fix_candidates=[],
            scout_candidates=[],
            quarantine_callback=quarantined.append,
        )

        self.assertEqual(payload["candidate_count"], 0)
        self.assertEqual(payload["quarantined_count"], 1)
        self.assertEqual(payload["quarantined_counts"], {"missing_expression": 1})
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0]["source"], "planner")
        self.assertEqual(quarantined[0]["reason"], "missing_expression")


if __name__ == "__main__":
    unittest.main()
