import unittest

from scripts.daily_best import render_daily_best


class TestDailyBest(unittest.TestCase):
    def test_render_daily_best_outputs_top_submit_ready_candidates(self):
        payload = {
            "batch": {
                "candidates": [
                    {
                        "thesis": "Technical indicator blend",
                        "thesis_id": "technical_indicator",
                        "why": "Uses oscillator, momentum, and band-style signals inspired by classic factor libraries.",
                        "expression": "rank(ts_zscore(((ts_mean(high, 10)-close)/(ts_mean(high,10)-ts_mean(low,10))),21))",
                        "candidate_score": 1.1,
                        "confidence_score": 0.74,
                        "novelty_score": 0.8,
                        "style_alignment_score": 0.7,
                        "risk_tags": [],
                        "seed_ready": True,
                        "qualified": True,
                        "quality_label": "qualified",
                        "settings": "USA, TOP3000, Decay 6, Delay 1, Truncation 0.05, Neutralization Industry",
                        "local_metrics": {
                            "verdict": "PASS",
                            "confidence": "HIGH",
                            "alpha_score": 82.0,
                            "sharpe": 1.95,
                            "fitness": 1.72,
                            "turnover": 0.18,
                        },
                        "token_program": ["WILLIAMS", "TSZ_21", "RANK"],
                    },
                    {
                        "thesis": "Residual or de-beta structure",
                        "thesis_id": "residual_beta",
                        "why": "Reduces market-direction overlap.",
                        "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                        "candidate_score": 1.0,
                        "confidence_score": 0.66,
                        "novelty_score": 0.72,
                        "style_alignment_score": 0.69,
                        "risk_tags": ["turnover_risk", "weight_risk"],
                        "seed_ready": True,
                        "qualified": True,
                        "quality_label": "qualified",
                        "settings": "USA, TOP3000, Decay 7, Delay 1, Truncation 0.06, Neutralization Industry",
                        "local_metrics": {
                            "verdict": "PASS",
                            "confidence": "MEDIUM",
                            "alpha_score": 74.2,
                            "sharpe": 1.88,
                            "fitness": 1.94,
                            "turnover": 0.14,
                        },
                        "token_program": ["CLOSE", "BETA", "REG_RESD_63", "RANK"],
                    },
                    {
                        "thesis": "Simple price hypothesis",
                        "thesis_id": "simple_price",
                        "why": "Simple ranked price idea.",
                        "expression": "winsorize(rank(inverse(close)),std=5)",
                        "candidate_score": 0.93,
                        "confidence_score": 0.58,
                        "novelty_score": 0.76,
                        "style_alignment_score": 0.55,
                        "risk_tags": ["turnover_risk", "weight_risk"],
                        "seed_ready": True,
                        "qualified": True,
                        "quality_label": "qualified",
                        "settings": "USA, TOP1000, Decay 4, Delay 1, Truncation 0.03, Neutralization Industry",
                        "local_metrics": {
                            "verdict": "LIKELY_PASS",
                            "confidence": "MEDIUM",
                            "alpha_score": 66.1,
                            "sharpe": 1.58,
                            "fitness": 1.21,
                            "turnover": 0.19,
                        },
                        "token_program": ["CLOSE", "INVERSE", "RANK", "WINSORIZE"],
                    },
                    {
                        "thesis": "Shock response",
                        "thesis_id": "shock_response",
                        "why": "Reacts to unusual activity.",
                        "expression": "rank(ts_zscore(divide(ts_std_dev(returns,21),ts_mean(volume, 63)),21))",
                        "candidate_score": 0.9,
                        "confidence_score": 0.22,
                        "novelty_score": 0.7,
                        "style_alignment_score": 0.5,
                        "risk_tags": [],
                        "seed_ready": True,
                        "qualified": False,
                        "quality_label": "watchlist",
                        "settings": "USA, TOP1000, Decay 6, Delay 1, Truncation 0.04, Neutralization Industry",
                        "local_metrics": {
                            "verdict": "PASS",
                            "confidence": "MEDIUM",
                            "alpha_score": 79.0,
                            "sharpe": 2.0,
                            "fitness": 2.2,
                            "turnover": 0.12,
                        },
                        "token_program": ["RETURNS", "VOL", "RANK"],
                    },
                ]
            }
        }

        markdown = render_daily_best(payload, seed_store={})
        self.assertIn("Top Submit-Ready Alphas Today", markdown)
        self.assertIn("Submit-ready gate", markdown)
        self.assertIn("Candidates shown: 3", markdown)
        self.assertIn("## Candidate 1", markdown)
        self.assertIn("## Candidate 2", markdown)
        self.assertIn("## Candidate 3", markdown)
        self.assertIn("quality_label", markdown)
        self.assertIn("Technical indicator blend", markdown)
        self.assertIn("Residual or de-beta structure", markdown)
        self.assertIn("Simple price hypothesis", markdown)
        self.assertNotIn("Reacts to unusual activity.", markdown)
        self.assertIn("seed_store_status: new candidate", markdown)
        self.assertIn("settings: USA, TOP3000, Decay 6, Delay 1, Truncation 0.05, Neutralization Industry", markdown)
        self.assertIn("alpha_score: 82.0", markdown)
        self.assertIn("sharpe: 1.95", markdown)
        self.assertEqual(markdown.count("## Candidate"), 3)

    def test_render_daily_best_can_include_submit_ready_auto_fix_candidate(self):
        payload = {"batch": {"candidates": []}}
        extra_candidates = [
            {
                "thesis": "Auto-fix rewrite [Technical Indicator]",
                "thesis_id": "technical_indicator",
                "why": "Auto-fixed from A1.",
                "expression": "rank(ts_sum(close,10))",
                "candidate_score": 0.69,
                "confidence_score": 0.61,
                "novelty_score": 0.73,
                "style_alignment_score": 0.62,
                "risk_tags": [],
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "settings": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market",
                "source": "auto_fix_rewrite",
                "repair_status": "submit_ready",
                "local_metrics": {
                    "verdict": "PASS",
                    "confidence": "HIGH",
                    "alpha_score": 69.0,
                    "sharpe": 1.5,
                    "fitness": 1.1,
                    "turnover": 0.16,
                },
                "token_program": ["CLOSE", "TS_SUM_10", "RANK"],
            }
        ]

        markdown = render_daily_best(payload, seed_store={}, extra_candidates=extra_candidates)

        self.assertIn("Auto-fix candidates considered: 1", markdown)
        self.assertIn("Auto-fix candidates shown: 1", markdown)
        self.assertIn("source: auto_fix_rewrite", markdown)
        self.assertIn("Auto-fix rewrite [Technical Indicator]", markdown)

    def test_render_daily_best_accepts_evaluated_pool_payload(self):
        payload = {
            "candidates": [
                {
                    "thesis": "Residual or de-beta structure",
                    "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                    "why": "Evaluated through the run queue.",
                    "seed_ready": True,
                    "qualified": True,
                    "quality_label": "qualified",
                    "confidence_score": 0.81,
                    "local_metrics": {
                        "verdict": "PASS",
                        "confidence": "HIGH",
                        "alpha_score": 77.0,
                        "sharpe": 1.92,
                        "fitness": 1.88,
                        "turnover": 0.14,
                    },
                }
            ]
        }

        markdown = render_daily_best(payload, seed_store={})

        self.assertIn("Candidates shown: 1", markdown)
        self.assertIn("Residual or de-beta structure", markdown)


if __name__ == "__main__":
    unittest.main()
