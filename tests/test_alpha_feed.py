import unittest

from scripts.alpha_feed import render_alpha_feed


class TestAlphaFeed(unittest.TestCase):
    def test_render_alpha_feed_lists_multiple_candidates(self):
        payload = {
            "memory": {
                "style_leaders": [{"tag": "technical", "learning_score": 0.71}],
            },
            "batch": {
                "notes": ["Example note"],
                "candidates": [
                    {
                        "thesis": "Technical indicator blend",
                        "expression": "rank(ts_zscore(((ts_mean(high, 10)-close)/(ts_mean(high,10)-ts_mean(low,10))),21))",
                        "why": "Uses oscillator, momentum, and band-style signals inspired by classic factor libraries.",
                        "candidate_score": 1.1,
                        "confidence_score": 0.72,
                        "novelty_score": 0.8,
                        "style_alignment_score": 0.7,
                        "risk_tags": [],
                        "seed_ready": True,
                        "qualified": True,
                        "quality_fail_reasons": [],
                    },
                    {
                        "thesis": "VWAP dislocation",
                        "expression": "rank(ts_zscore(abs(close-vwap),21))",
                        "why": "Looks for price dislocation versus intraday anchor prices.",
                        "candidate_score": 0.9,
                        "confidence_score": 0.31,
                        "novelty_score": 0.7,
                        "style_alignment_score": 0.5,
                        "risk_tags": ["weight_risk"],
                        "seed_ready": True,
                        "qualified": False,
                        "quality_fail_reasons": ["confidence"],
                    },
                ],
            },
        }

        markdown = render_alpha_feed(payload)
        self.assertIn("Alpha Feed", markdown)
        self.assertIn("Strictly Qualified", markdown)
        self.assertIn("Watchlist", markdown)
        self.assertIn("Technical indicator blend", markdown)
        self.assertIn("VWAP dislocation", markdown)

    def test_render_alpha_feed_surfaces_actionable_auto_fix_candidates(self):
        payload = {"memory": {}, "batch": {"notes": [], "candidates": []}}
        extra_candidates = [
            {
                "thesis": "Auto-fix rewrite [Technical Indicator]",
                "expression": "rank(ts_sum(close,10))",
                "why": "Auto-fixed from A1.",
                "repair_status": "promising",
                "risk_tags": ["unproven_style"],
                "local_metrics": {
                    "alpha_score": 61.0,
                    "sharpe": 1.25,
                    "fitness": 0.9,
                },
            }
        ]

        markdown = render_alpha_feed(payload, extra_candidates=extra_candidates)

        self.assertIn("Auto-Fix Actionable", markdown)
        self.assertIn("repair_status: promising", markdown)
        self.assertIn("rank(ts_sum(close,10))", markdown)

    def test_render_alpha_feed_accepts_evaluated_pool_payload(self):
        payload = {
            "memory": {"style_leaders": [{"tag": "residual", "learning_score": 0.88}]},
            "candidates": [
                {
                    "thesis": "Residual or de-beta structure",
                    "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                    "why": "Evaluated via queue.",
                    "confidence_score": 0.83,
                    "candidate_score": 1.2,
                    "novelty_score": 0.66,
                    "style_alignment_score": 0.71,
                    "seed_ready": True,
                    "qualified": True,
                    "quality_fail_reasons": [],
                    "risk_tags": [],
                }
            ],
        }

        markdown = render_alpha_feed(payload)

        self.assertIn("Residual or de-beta structure", markdown)
        self.assertIn("artifacts/latest/evaluated_candidates.json", markdown)


if __name__ == "__main__":
    unittest.main()
