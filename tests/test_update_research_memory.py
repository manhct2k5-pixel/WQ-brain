import unittest

from scripts.update_research_memory import update_research_memory


class TestUpdateResearchMemory(unittest.TestCase):
    def test_update_research_memory_creates_layered_memory_with_decay(self):
        current_snapshot = {
            "failure_counts": {"LOW_SHARPE": 2},
            "blocked_skeletons": [],
            "hard_blocked_skeletons": [],
            "soft_blocked_skeletons": ["recent_soft_skeleton"],
            "blocked_families": [],
            "hard_blocked_families": [],
            "soft_blocked_families": ["technical_indicator"],
            "preferred_skeletons": ["winner_now"],
            "historical_skeletons": ["history_now"],
            "family_stats": {
                "technical_indicator": {
                    "attempts": 2,
                    "completed": 2,
                    "pass_all_count": 0,
                    "avg_research_score": 0.4,
                    "avg_sharpe": 0.5,
                    "avg_fitness": 0.45,
                    "serious_failures": 2,
                    "failure_counts": {"LOW_SHARPE": 2},
                }
            },
            "style_leaders": [{"tag": "technical", "learning_score": 0.42}],
            "seed_context": {},
            "window_rows": 8,
            "top_rows": [{"regular_code": "rank(close)"}],
            "suggestions": ["Try fresher technical variants."],
        }
        previous_memory = {
            "_meta": {"updated_at": "2025-12-01T00:00:00"},
            "summary_memory": {
                "_meta": {"updated_at": "2025-12-01T00:00:00"},
                "failure_counts": {"LOW_SHARPE": 20, "LOW_FITNESS": 18},
                "family_stats": {
                    "technical_indicator": {
                        "attempts": 40,
                        "completed": 20,
                        "pass_all_count": 0,
                        "avg_research_score": 0.15,
                        "avg_sharpe": 0.2,
                        "avg_fitness": 0.18,
                        "serious_failures": 24,
                        "failure_counts": {"LOW_SHARPE": 20, "LOW_FITNESS": 18},
                    }
                },
                "style_leaders": [{"tag": "technical", "learning_score": 0.9}],
                "block_scores": {
                    "skeletons": {
                        "soft": {"old_soft_skeleton": {"score": 1.5, "updated_at": "2025-12-01T00:00:00", "level": "soft"}},
                        "hard": {"old_hard_skeleton": {"score": 3.0, "updated_at": "2025-12-01T00:00:00", "level": "hard"}},
                    },
                    "families": {
                        "soft": {"pv_divergence": {"score": 1.4, "updated_at": "2025-12-01T00:00:00", "level": "soft"}},
                        "hard": {"shock_response": {"score": 3.0, "updated_at": "2025-12-01T00:00:00", "level": "hard"}},
                    },
                },
            },
            "archive_log": {"recent_runs": [{"run_at": "2025-12-01T00:00:00"}]},
            "planner_memory": {
                "failure_counts": {"LOW_SHARPE": 20, "LOW_FITNESS": 18},
                "preferred_skeletons": ["winner_old"],
                "historical_skeletons": ["history_old"],
                "family_stats": {},
                "style_leaders": [{"tag": "technical", "learning_score": 0.9}],
                "seed_context": {},
            },
        }

        payload = update_research_memory(
            current_snapshot,
            current_snapshot=current_snapshot,
            previous_memory=previous_memory,
            results_summary={"summary": {"qualified_count": 1}},
            simulation_records=[{"expression": "rank(close)"}],
            updated_at="2026-03-27T23:55:00",
        )

        self.assertIn("working_memory", payload)
        self.assertIn("summary_memory", payload)
        self.assertIn("archive_log", payload)
        self.assertIn("planner_memory", payload)
        self.assertLess(payload["summary_memory"]["failure_counts"]["LOW_SHARPE"], 10)
        self.assertIn("recent_soft_skeleton", payload["summary_memory"]["soft_blocked_skeletons"])
        self.assertNotIn("old_hard_skeleton", payload["summary_memory"]["hard_blocked_skeletons"])
        self.assertEqual(len(payload["archive_log"]["recent_runs"]), 2)
        self.assertEqual(payload["planner_memory"]["blocked_families"], [])


if __name__ == "__main__":
    unittest.main()
