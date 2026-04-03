import json
from collections import Counter
import tempfile
import unittest
from unittest.mock import patch

from scripts.plan_next_batch import (
    apply_adaptive_planning_controls,
    build_batch,
    build_candidate,
    build_memory,
    evaluate_quality_gate,
    load_memory,
    merge_family_stats,
    merge_memory,
    planner_settings_label,
    rank_theses,
    summarize_local_metrics,
)
from scripts.results_digest import skeletonize
from src.internal_scoring import CHECK_COLUMNS
from src.internal_scoring import HistoryIndex


class TestPlanNextBatch(unittest.TestCase):
    def test_summarize_local_metrics_preserves_failure_flags_and_settings(self):
        result = {
            "alpha_id": "LOCAL-ABCD1234",
            "verdict": "FAIL",
            "confidence": "MEDIUM",
            "alpha_score": 61.0,
            "sharpe": 1.21,
            "fitness": 0.93,
            "turnover": 0.42,
            "returns": 0.08,
            "drawdown": 0.03,
            "margin": 0.02,
            "uniqueness_proxy": 0.64,
            "style_tags": ["vwap", "rank", "normalization"],
            "settings": {"label": "USA, TOP3000, Decay 3, Delay 1, Truncation 0.02, Neutralization Subindustry"},
            "surrogate_shadow": {"status": "ready", "preview_verdict": "PASS"},
            "surrogate_shadow_status": "ready",
            "surrogate_shadow_preview_verdict": "PASS",
            "surrogate_shadow_alignment": "aligned",
            "surrogate_shadow_penalty": 0.0,
            "surrogate_shadow_reasons": [],
            "surrogate_shadow_hard_signal": "none",
            "LOW_SHARPE": "FAIL",
            "LOW_FITNESS": "FAIL",
            "LOW_TURNOVER": "PASS",
            "HIGH_TURNOVER": "PASS",
            "CONCENTRATED_WEIGHT": "FAIL",
            "LOW_SUB_UNIVERSE_SHARPE": "PASS",
            "SELF_CORRELATION": "PASS",
            "MATCHES_COMPETITION": "PASS",
        }

        metrics = summarize_local_metrics(result)

        self.assertEqual(metrics["alpha_id"], "LOCAL-ABCD1234")
        self.assertEqual(metrics["settings"]["label"], result["settings"]["label"])
        for name in CHECK_COLUMNS:
            self.assertIn(name, metrics)
        self.assertEqual(metrics["LOW_SHARPE"], "FAIL")
        self.assertEqual(metrics["CONCENTRATED_WEIGHT"], "FAIL")

    def test_planner_settings_label_avoids_subindustry_for_worldquant_compatibility(self):
        self.assertIn("Neutralization Industry", planner_settings_label("reversal_conditioned"))
        self.assertIn("Neutralization Industry", planner_settings_label("vwap_dislocation"))

    def test_build_memory_ignores_pending_and_blocks_repeated_serious_failures(self):
        expression = "rank(ts_zscore(abs(close-vwap),21))"
        rows = [
            {
                "alpha_id": "A1",
                "regular_code": expression,
                "turnover": "0.21",
                "fitness": "0.4",
                "sharpe": "0.7",
                "LOW_SHARPE": "FAIL",
                "LOW_FITNESS": "FAIL",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "SELF_CORRELATION": "PASS",
                "MATCHES_COMPETITION": "PASS",
            },
            {
                "alpha_id": "A2",
                "regular_code": expression,
                "turnover": "0.22",
                "fitness": "0.5",
                "sharpe": "0.8",
                "LOW_SHARPE": "FAIL",
                "LOW_FITNESS": "FAIL",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "SELF_CORRELATION": "PASS",
                "MATCHES_COMPETITION": "PASS",
            },
            {
                "alpha_id": "A3",
                "regular_code": expression,
                "turnover": "0.20",
                "fitness": "0.6",
                "sharpe": "1.0",
                "LOW_SHARPE": "PENDING",
                "LOW_FITNESS": "PENDING",
                "LOW_TURNOVER": "PENDING",
                "HIGH_TURNOVER": "PENDING",
                "CONCENTRATED_WEIGHT": "PENDING",
                "LOW_SUB_UNIVERSE_SHARPE": "PENDING",
                "SELF_CORRELATION": "PENDING",
                "MATCHES_COMPETITION": "PENDING",
            },
        ]

        memory = build_memory(rows, top_n=3)
        self.assertIn(skeletonize(expression), memory["blocked_skeletons"])
        self.assertEqual(memory["family_stats"]["vwap_dislocation"]["completed"], 2)
        self.assertEqual(memory["family_stats"]["vwap_dislocation"]["attempts"], 3)
        self.assertEqual(memory["family_stats"]["vwap_dislocation"]["real_attempts"], 3)

    def test_matches_competition_blocks_immediately(self):
        expression = "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))"
        rows = [
            {
                "alpha_id": "A1",
                "regular_code": expression,
                "turnover": "0.18",
                "fitness": "0.8",
                "sharpe": "1.1",
                "LOW_SHARPE": "PASS",
                "LOW_FITNESS": "PASS",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "SELF_CORRELATION": "PASS",
                "MATCHES_COMPETITION": "FAIL",
            }
        ]

        memory = build_memory(rows, top_n=1)
        self.assertIn(skeletonize(expression), memory["blocked_skeletons"])

    def test_build_batch_outputs_seed_fields_and_caps_thesis(self):
        memory = {
            "failure_counts": {"SELF_CORRELATION": 3},
            "blocked_skeletons": [],
            "blocked_families": [],
            "preferred_skeletons": [],
            "historical_skeletons": [],
            "family_stats": {},
            "style_leaders": [],
            "seed_context": {
                "family_counts": {},
                "planned_family_counts": {},
                "seeded_skeletons": [],
                "planned_skeletons": [],
            },
            "window_rows": 10,
            "suggestions": [],
        }

        batch = build_batch(memory, max_candidates=8, include_local_metrics=True, history_index=HistoryIndex())
        self.assertGreater(len(batch["candidates"]), 0)

        thesis_counts = Counter(item["thesis_id"] for item in batch["candidates"])
        self.assertTrue(all(count <= 2 for count in thesis_counts.values()))
        self.assertEqual(
            len({item["skeleton"] for item in batch["candidates"]}),
            len(batch["candidates"]),
        )
        for item in batch["candidates"]:
            self.assertIn("novelty_score", item)
            self.assertIn("style_alignment_score", item)
            self.assertIn("confidence_score", item)
            self.assertIn("qualified", item)
            self.assertIn("risk_tags", item)
            self.assertIn("seed_ready", item)
            self.assertIn("token_program", item)
            self.assertIn("settings", item)
            self.assertIn("local_metrics", item)
            self.assertIn("alpha_score", item["local_metrics"])

    def test_build_batch_reduces_seed_bias_for_overseeded_family(self):
        memory = {
            "failure_counts": {"LOW_SHARPE": 3},
            "blocked_skeletons": [],
            "blocked_families": [],
            "preferred_skeletons": [],
            "historical_skeletons": [],
            "family_stats": {},
            "style_leaders": [],
            "seed_context": {
                "family_counts": {"reversal_conditioned": 4},
                "planned_family_counts": {"reversal_conditioned": 2},
                "seeded_skeletons": [],
                "planned_skeletons": [],
            },
            "window_rows": 10,
            "suggestions": [],
        }

        batch = build_batch(memory, max_candidates=8)
        reversal_candidates = [
            item for item in batch["candidates"] if item["thesis_id"] == "reversal_conditioned"
        ]
        self.assertLessEqual(len(reversal_candidates), 1)

    def test_build_batch_prefers_best_primary_candidate_globally(self):
        memory = {
            "seed_context": {
                "family_counts": {},
                "planned_family_counts": {},
                "seeded_skeletons": [],
                "planned_skeletons": [],
            },
            "suggestions": [],
        }
        ranked_theses = [
            {"id": "family_a", "label": "Family A", "why": "A", "variants": [{"variant_id": "a1"}], "blocked": False},
            {"id": "family_b", "label": "Family B", "why": "B", "variants": [{"variant_id": "b1"}], "blocked": False},
        ]

        def fake_build_candidate(thesis, variant, _memory, history_index=None):
            if thesis["id"] == "family_a":
                return {
                    "variant_id": variant["variant_id"],
                    "thesis_id": thesis["id"],
                    "thesis": thesis["label"],
                    "why": thesis["why"],
                    "expression": "rank(close)",
                    "token_program": ["CLOSE", "RANK"],
                    "novelty_score": 0.74,
                    "style_alignment_score": 0.18,
                    "risk_tags": [],
                    "seed_ready": True,
                    "candidate_score": 0.91,
                    "skeleton": "family_a",
                    "confidence_score": 0.44,
                    "qualified": False,
                    "quality_label": "watchlist",
                }
            return {
                "variant_id": variant["variant_id"],
                "thesis_id": thesis["id"],
                "thesis": thesis["label"],
                "why": thesis["why"],
                "expression": "rank(volume)",
                "token_program": ["VOLUME", "RANK"],
                "novelty_score": 0.70,
                "style_alignment_score": 0.22,
                "risk_tags": [],
                "seed_ready": True,
                "candidate_score": 0.62,
                "skeleton": "family_b",
                "confidence_score": 0.81,
                "qualified": True,
                "quality_label": "qualified",
            }

        with patch("scripts.plan_next_batch.rank_theses", return_value=ranked_theses), patch(
            "scripts.plan_next_batch.build_candidate", side_effect=fake_build_candidate
        ):
            batch = build_batch(memory, max_candidates=1)

        self.assertEqual(batch["qualified_count"], 1)
        self.assertEqual(batch["candidates"][0]["thesis_id"], "family_b")

    def test_build_batch_backfills_overflow_candidates_when_primary_caps_underfill(self):
        memory = {
            "seed_context": {
                "family_counts": {"family_a": 5},
                "planned_family_counts": {"family_a": 1},
                "seeded_skeletons": [],
                "planned_skeletons": [],
            },
            "suggestions": [],
        }
        ranked_theses = [
            {
                "id": "family_a",
                "label": "Family A",
                "why": "A",
                "variants": [{"variant_id": "a1"}, {"variant_id": "a2"}, {"variant_id": "a3"}],
                "blocked": False,
            }
        ]
        candidates = {
            "a1": {
                "variant_id": "a1",
                "thesis_id": "family_a",
                "thesis": "Family A",
                "why": "A",
                "expression": "rank(close)",
                "token_program": ["CLOSE", "RANK"],
                "novelty_score": 0.72,
                "style_alignment_score": 0.20,
                "risk_tags": [],
                "seed_ready": True,
                "candidate_score": 0.84,
                "skeleton": "family_a_1",
                "confidence_score": 0.61,
                "qualified": False,
                "quality_label": "watchlist",
            },
            "a2": {
                "variant_id": "a2",
                "thesis_id": "family_a",
                "thesis": "Family A",
                "why": "A",
                "expression": "rank(volume)",
                "token_program": ["VOLUME", "RANK"],
                "novelty_score": 0.69,
                "style_alignment_score": 0.19,
                "risk_tags": [],
                "seed_ready": True,
                "candidate_score": 0.77,
                "skeleton": "family_a_2",
                "confidence_score": 0.58,
                "qualified": False,
                "quality_label": "watchlist",
            },
            "a3": {
                "variant_id": "a3",
                "thesis_id": "family_a",
                "thesis": "Family A",
                "why": "A",
                "expression": "rank(vwap)",
                "token_program": ["VWAP", "RANK"],
                "novelty_score": 0.67,
                "style_alignment_score": 0.17,
                "risk_tags": [],
                "seed_ready": True,
                "candidate_score": 0.73,
                "skeleton": "family_a_3",
                "confidence_score": 0.55,
                "qualified": False,
                "quality_label": "watchlist",
            },
        }

        def fake_build_candidate(thesis, variant, _memory, history_index=None):
            return candidates[variant["variant_id"]]

        with patch("scripts.plan_next_batch.rank_theses", return_value=ranked_theses), patch(
            "scripts.plan_next_batch.build_candidate", side_effect=fake_build_candidate
        ):
            batch = build_batch(memory, max_candidates=3)

        self.assertEqual(len(batch["candidates"]), 3)
        self.assertTrue(any("underfilled the batch" in note for note in batch["notes"]))

    def test_rank_theses_learns_from_recent_winning_styles(self):
        rows = [
            {
                "alpha_id": "A1",
                "regular_code": "rank(winsorize(ts_corr(ts_rank(volume,10),ts_rank(close,10),10),std=5))",
                "turnover": "0.23",
                "fitness": "1.6",
                "sharpe": "2.1",
                "LOW_SHARPE": "PASS",
                "LOW_FITNESS": "PASS",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "SELF_CORRELATION": "PASS",
                "MATCHES_COMPETITION": "PASS",
            },
            {
                "alpha_id": "A2",
                "regular_code": "rank(ts_zscore(-ts_corr(ts_delta(volume,1),ts_delta(close,1),10),21))",
                "turnover": "0.24",
                "fitness": "1.5",
                "sharpe": "1.9",
                "LOW_SHARPE": "PASS",
                "LOW_FITNESS": "PASS",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "SELF_CORRELATION": "PASS",
                "MATCHES_COMPETITION": "PASS",
            },
        ]

        memory = build_memory(rows, top_n=2)
        ranked = rank_theses(memory)
        self.assertEqual(ranked[0]["id"], "pv_divergence")
        self.assertTrue(any(item["tag"] == "correlation" for item in memory["style_leaders"]))

    def test_rank_theses_promotes_simple_price_patterns_under_uniqueness_pressure(self):
        memory = {
            "failure_counts": {"SELF_CORRELATION": 4, "MATCHES_COMPETITION": 4},
            "blocked_families": [],
            "soft_blocked_families": [],
            "family_stats": {},
            "style_leaders": [],
            "seed_context": {
                "family_counts": {},
                "planned_family_counts": {},
                "seeded_skeletons": [],
                "planned_skeletons": [],
            },
            "window_rows": 12,
            "adaptive_controls": {},
        }

        ranked = rank_theses(memory)
        top_ids = [item["id"] for item in ranked[:3]]

        self.assertIn("simple_price_patterns", top_ids)
        self.assertTrue(any(item["id"] == "simple_price_patterns" and item["thesis_score"] > 0.0 for item in ranked[:3]))

    def test_build_memory_hard_blocks_family_after_three_consecutive_real_failures(self):
        expression = "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))"
        rows = [
            {
                "alpha_id": f"A{index}",
                "regular_code": expression,
                "turnover": "0.21",
                "fitness": "0.4",
                "sharpe": "0.8",
                "LOW_SHARPE": "FAIL",
                "LOW_FITNESS": "FAIL",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "FAIL",
                "SELF_CORRELATION": "PASS",
                "MATCHES_COMPETITION": "PASS",
            }
            for index in range(1, 4)
        ]

        memory = build_memory(rows, top_n=3)

        self.assertIn("residual_beta", memory["blocked_families"])
        self.assertEqual(memory["family_stats"]["residual_beta"]["real_fail_streak"], 3)
        self.assertEqual(memory["family_stats"]["residual_beta"]["real_fail_count"], 3)

    def test_build_batch_can_emit_technical_indicator_candidates(self):
        memory = {
            "failure_counts": {"LOW_SHARPE": 4, "LOW_FITNESS": 3},
            "blocked_skeletons": [],
            "blocked_families": [],
            "preferred_skeletons": [],
            "historical_skeletons": [],
            "family_stats": {},
            "style_leaders": [
                {"tag": "technical", "learning_score": 0.7},
                {"tag": "momentum", "learning_score": 0.68},
                {"tag": "band", "learning_score": 0.64},
            ],
            "seed_context": {
                "family_counts": {},
                "planned_family_counts": {},
                "seeded_skeletons": [],
                "planned_skeletons": [],
            },
            "window_rows": 10,
            "suggestions": [],
        }

        batch = build_batch(memory, max_candidates=8)
        self.assertTrue(any(item["thesis_id"] == "technical_indicator" for item in batch["candidates"]))

    def test_evaluate_quality_gate_requires_local_submit_gate(self):
        memory = {
            "family_stats": {},
        }
        strong_candidate = {
            "seed_ready": True,
            "candidate_score": 0.92,
            "novelty_score": 0.94,
            "style_alignment_score": 0.41,
            "risk_tags": [],
            "local_metrics": {
                "verdict": "PASS",
                "alpha_score": 76.0,
                "sharpe": 1.9,
                "fitness": 1.8,
            },
        }
        weak_local_candidate = {
            **strong_candidate,
            "local_metrics": {
                "verdict": "FAIL",
                "alpha_score": 22.0,
                "sharpe": 0.4,
                "fitness": 0.3,
            },
        }

        qualified = evaluate_quality_gate(strong_candidate, "technical_indicator", memory)
        rejected = evaluate_quality_gate(weak_local_candidate, "technical_indicator", memory)

        self.assertTrue(qualified["qualified"])
        self.assertEqual(qualified["quality_label"], "qualified")
        self.assertGreaterEqual(qualified["confidence_score"], 0.45)
        self.assertFalse(rejected["qualified"])
        self.assertEqual(rejected["quality_label"], "watchlist")
        self.assertIn("local", rejected["quality_fail_reasons"])
        self.assertLess(rejected["confidence_score"], 0.45)
        self.assertLess(rejected["confidence_score"], qualified["confidence_score"])

    def test_evaluate_quality_gate_allows_strong_local_override_for_moderate_novelty(self):
        memory = {
            "family_stats": {},
        }
        candidate = {
            "seed_ready": True,
            "candidate_score": 0.16,
            "novelty_score": 0.65,
            "style_alignment_score": 0.07,
            "risk_tags": ["turnover_risk", "weight_risk"],
            "local_metrics": {
                "verdict": "PASS",
                "alpha_score": 78.0,
                "sharpe": 1.9,
                "fitness": 1.8,
            },
        }

        result = evaluate_quality_gate(candidate, "shock_response", memory)

        self.assertTrue(result["qualified"])
        self.assertEqual(result["quality_label"], "qualified")
        self.assertTrue(result["quality_stages"]["statistical"])
        self.assertTrue(result["quality_stages"]["history_fit"])

    def test_evaluate_quality_gate_allows_strong_local_override_with_negative_candidate_score(self):
        memory = {
            "family_stats": {},
        }
        candidate = {
            "seed_ready": True,
            "candidate_score": -1.2,
            "novelty_score": 0.64,
            "style_alignment_score": 0.11,
            "risk_tags": ["turnover_risk", "weight_risk"],
            "local_metrics": {
                "verdict": "PASS",
                "alpha_score": 79.0,
                "sharpe": 2.0,
                "fitness": 1.9,
            },
        }

        result = evaluate_quality_gate(candidate, "shock_response", memory)

        self.assertTrue(result["qualified"])
        self.assertTrue(result["quality_stages"]["statistical"])
        self.assertGreaterEqual(result["confidence_score"], 0.45)

    def test_evaluate_quality_gate_rejects_surrogate_shadow_fail_even_with_strong_local_proxy(self):
        memory = {
            "family_stats": {},
        }
        candidate = {
            "seed_ready": True,
            "candidate_score": 0.24,
            "novelty_score": 0.82,
            "style_alignment_score": 0.16,
            "risk_tags": [],
            "local_metrics": {
                "verdict": "PASS",
                "alpha_score": 78.0,
                "sharpe": 1.9,
                "fitness": 1.8,
                "surrogate_shadow_status": "ready",
                "surrogate_shadow_preview_verdict": "FAIL",
                "surrogate_shadow_alignment": "more_cautious",
                "surrogate_shadow_hard_signal": "severe_mismatch",
                "surrogate_shadow": {
                    "status": "ready",
                    "preview_verdict": "FAIL",
                    "alignment": "more_cautious",
                },
            },
        }

        result = evaluate_quality_gate(candidate, "shock_response", memory)

        self.assertFalse(result["quality_stages"]["local"])
        self.assertFalse(result["qualified"])
        self.assertIn("local", result["quality_fail_reasons"])

    def test_merge_family_stats_downweights_stale_memory_when_current_signal_exists(self):
        current = {
            "technical_indicator": {
                "attempts": 2,
                "completed": 2,
                "pass_all_count": 2,
                "real_attempts": 2,
                "real_completed": 2,
                "real_pass_all_count": 2,
                "real_fail_count": 0,
                "real_fail_streak": 0,
                "avg_research_score": 2.0,
                "avg_sharpe": 2.1,
                "avg_fitness": 2.2,
                "serious_failures": 0,
                "failure_counts": {},
            }
        }
        previous = {
            "technical_indicator": {
                "attempts": 200,
                "completed": 40,
                "pass_all_count": 20,
                "real_attempts": 20,
                "real_completed": 20,
                "real_pass_all_count": 0,
                "real_fail_count": 20,
                "real_fail_streak": 3,
                "avg_research_score": 0.2,
                "avg_sharpe": 0.5,
                "avg_fitness": 0.3,
                "serious_failures": 220,
                "failure_counts": {"LOW_SHARPE": 180, "LOW_FITNESS": 170},
            }
        }

        merged = merge_family_stats(current, previous)["technical_indicator"]

        self.assertLess(merged["attempts"], 60)
        self.assertLess(merged["serious_failures"], 50)
        self.assertGreater(merged["avg_research_score"], 0.25)
        self.assertLess(merged["failure_counts"]["LOW_SHARPE"], 50)
        self.assertEqual(merged["real_fail_streak"], 0)

    def test_merge_family_stats_carries_real_fail_streak_across_windows_without_reset(self):
        current = {
            "technical_indicator": {
                "attempts": 1,
                "completed": 1,
                "pass_all_count": 0,
                "real_attempts": 1,
                "real_completed": 1,
                "real_pass_all_count": 0,
                "real_fail_count": 1,
                "real_fail_streak": 1,
                "avg_research_score": 0.4,
                "avg_sharpe": 0.6,
                "avg_fitness": 0.5,
                "serious_failures": 1,
                "failure_counts": {"LOW_SHARPE": 1},
            }
        }
        previous = {
            "technical_indicator": {
                "attempts": 6,
                "completed": 6,
                "pass_all_count": 0,
                "real_attempts": 2,
                "real_completed": 2,
                "real_pass_all_count": 0,
                "real_fail_count": 2,
                "real_fail_streak": 2,
                "avg_research_score": 0.3,
                "avg_sharpe": 0.5,
                "avg_fitness": 0.4,
                "serious_failures": 2,
                "failure_counts": {"LOW_SHARPE": 2},
            }
        }

        merged = merge_family_stats(current, previous)["technical_indicator"]

        self.assertEqual(merged["real_fail_streak"], 3)

    def test_merge_memory_does_not_keep_stale_blocklists(self):
        current = {
            "failure_counts": {"LOW_SHARPE": 1},
            "blocked_skeletons": [],
            "blocked_families": [],
            "preferred_skeletons": [],
            "historical_skeletons": [],
            "family_stats": {},
            "style_leaders": [],
            "seed_context": {},
        }
        previous = {
            "failure_counts": {"LOW_FITNESS": 2},
            "blocked_skeletons": ["old_skeleton"],
            "blocked_families": ["pv_divergence"],
            "preferred_skeletons": ["winner"],
            "historical_skeletons": ["history"],
            "family_stats": {},
            "style_leaders": [],
            "seed_context": {"family_counts": {"pv_divergence": 3}},
        }

        merged = merge_memory(current, previous)
        self.assertEqual(merged["blocked_skeletons"], [])
        self.assertEqual(merged["blocked_families"], [])
        self.assertIn("winner", merged["preferred_skeletons"])
        self.assertIn("history", merged["historical_skeletons"])
        self.assertEqual(merged["failure_counts"]["LOW_FITNESS"], 2)

    def test_merge_memory_rebuilds_family_blocklists_from_merged_real_fail_streak(self):
        current = {
            "failure_counts": {},
            "blocked_skeletons": [],
            "blocked_families": [],
            "preferred_skeletons": [],
            "historical_skeletons": [],
            "family_stats": {
                "technical_indicator": {
                    "attempts": 1,
                    "completed": 1,
                    "pass_all_count": 0,
                    "real_attempts": 1,
                    "real_completed": 1,
                    "real_pass_all_count": 0,
                    "real_fail_count": 1,
                    "real_fail_streak": 1,
                    "avg_research_score": 0.3,
                    "avg_sharpe": 0.5,
                    "avg_fitness": 0.4,
                    "serious_failures": 1,
                    "failure_counts": {"LOW_SHARPE": 1},
                }
            },
            "style_leaders": [],
            "seed_context": {},
        }
        previous = {
            "failure_counts": {},
            "blocked_skeletons": [],
            "blocked_families": [],
            "preferred_skeletons": [],
            "historical_skeletons": [],
            "family_stats": {
                "technical_indicator": {
                    "attempts": 6,
                    "completed": 6,
                    "pass_all_count": 0,
                    "real_attempts": 2,
                    "real_completed": 2,
                    "real_pass_all_count": 0,
                    "real_fail_count": 2,
                    "real_fail_streak": 2,
                    "avg_research_score": 0.2,
                    "avg_sharpe": 0.4,
                    "avg_fitness": 0.3,
                    "serious_failures": 2,
                    "failure_counts": {"LOW_SHARPE": 2},
                }
            },
            "style_leaders": [],
            "seed_context": {},
        }

        merged = merge_memory(current, previous)

        self.assertEqual(merged["family_stats"]["technical_indicator"]["real_fail_streak"], 3)
        self.assertIn("technical_indicator", merged["blocked_families"])

    def test_load_memory_reads_structured_planner_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/memory.json"
            payload = {
                "planner_memory": {
                    "failure_counts": {"LOW_SHARPE": 3},
                    "blocked_skeletons": ["hard_skeleton"],
                    "soft_blocked_skeletons": ["soft_skeleton"],
                    "blocked_families": ["technical_indicator"],
                    "soft_blocked_families": ["pv_divergence"],
                    "preferred_skeletons": ["winner"],
                    "historical_skeletons": ["history"],
                    "family_stats": {},
                    "style_leaders": [{"tag": "technical", "learning_score": 0.7}],
                    "seed_context": {},
                }
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            memory = load_memory(path)

        self.assertEqual(memory["blocked_skeletons"], ["hard_skeleton"])
        self.assertEqual(memory["soft_blocked_skeletons"], ["soft_skeleton"])
        self.assertEqual(memory["blocked_families"], ["technical_indicator"])
        self.assertEqual(memory["soft_blocked_families"], ["pv_divergence"])

    def test_build_candidate_keeps_soft_block_as_risk_not_hard_rejection(self):
        ranked = rank_theses(
            {
                "failure_counts": {},
                "blocked_skeletons": [],
                "soft_blocked_families": ["technical_indicator"],
                "blocked_families": [],
                "preferred_skeletons": [],
                "historical_skeletons": [],
                "family_stats": {},
                "style_leaders": [],
                "seed_context": {"seeded_skeletons": [], "planned_skeletons": [], "family_counts": {}, "planned_family_counts": {}},
                "window_rows": 10,
            }
        )
        thesis = next(item for item in ranked if item["id"] == "technical_indicator")
        variant = thesis["variants"][0]
        candidate = build_candidate(
            thesis,
            variant,
            {
                "failure_counts": {},
                "blocked_skeletons": [],
                "soft_blocked_skeletons": [],
                "soft_blocked_families": ["technical_indicator"],
                "blocked_families": [],
                "preferred_skeletons": [],
                "historical_skeletons": [],
                "family_stats": {},
                "style_leaders": [],
                "seed_context": {"seeded_skeletons": [], "planned_skeletons": [], "family_counts": {}, "planned_family_counts": {}},
                "window_rows": 10,
            },
        )

        self.assertIsNotNone(candidate)
        self.assertIn("soft_blocked_family_risk", candidate["risk_tags"])

    def test_apply_adaptive_planning_controls_reopens_weakest_soft_blocks(self):
        memory = {
            "failure_counts": {},
            "blocked_skeletons": [],
            "blocked_families": [],
            "soft_blocked_skeletons": ["soft_old", "soft_recent"],
            "soft_blocked_families": ["technical_indicator", "pv_divergence"],
            "preferred_skeletons": [],
            "historical_skeletons": [],
            "family_stats": {},
            "style_leaders": [],
            "seed_context": {},
            "suggestions": [],
            "block_scores": {
                "skeletons": {
                    "soft": {
                        "soft_old": {"score": 1.05, "updated_at": "2026-03-20T00:00:00", "level": "soft"},
                        "soft_recent": {"score": 1.6, "updated_at": "2026-03-27T00:00:00", "level": "soft"},
                    }
                },
                "families": {
                    "soft": {
                        "pv_divergence": {"score": 1.02, "updated_at": "2026-03-20T00:00:00", "level": "soft"},
                        "technical_indicator": {"score": 1.7, "updated_at": "2026-03-27T00:00:00", "level": "soft"},
                    }
                },
            },
        }

        adjusted = apply_adaptive_planning_controls(
            memory,
            {
                "warning": "Recovery mode active.",
                "exploration_boost": 0.12,
                "reopen_soft_blocked_families_count": 1,
                "reopen_soft_blocked_skeletons_count": 1,
            },
        )

        self.assertEqual(adjusted["soft_blocked_families"], ["technical_indicator"])
        self.assertEqual(adjusted["soft_blocked_skeletons"], ["soft_recent"])
        self.assertEqual(adjusted["adaptive_controls"]["reopened_soft_blocked_families"], ["pv_divergence"])
        self.assertEqual(adjusted["adaptive_controls"]["reopened_soft_blocked_skeletons"], ["soft_old"])
        self.assertEqual(adjusted["suggestions"][0], "Recovery mode active.")

    def test_apply_adaptive_planning_controls_can_ignore_block_lists(self):
        memory = {
            "failure_counts": {},
            "blocked_skeletons": ["hard_skeleton"],
            "blocked_families": ["technical_indicator"],
            "soft_blocked_skeletons": ["soft_skeleton"],
            "soft_blocked_families": ["pv_divergence"],
            "preferred_skeletons": [],
            "historical_skeletons": [],
            "family_stats": {},
            "style_leaders": [],
            "seed_context": {},
            "suggestions": [],
        }

        adjusted = apply_adaptive_planning_controls(
            memory,
            {
                "warning": "Manual override active.",
                "ignore_block_list": True,
                "exploration_boost": 0.18,
            },
        )

        self.assertEqual(adjusted["blocked_families"], [])
        self.assertEqual(adjusted["blocked_skeletons"], [])
        self.assertEqual(adjusted["soft_blocked_families"], [])
        self.assertEqual(adjusted["soft_blocked_skeletons"], [])
        self.assertTrue(adjusted["adaptive_controls"]["ignore_block_list"])
        self.assertEqual(adjusted["adaptive_controls"]["ignored_blocked_families"], ["technical_indicator"])
        self.assertEqual(adjusted["adaptive_controls"]["ignored_soft_blocked_skeletons"], ["soft_skeleton"])
        self.assertEqual(adjusted["suggestions"][0], "Manual override active.")


if __name__ == "__main__":
    unittest.main()
