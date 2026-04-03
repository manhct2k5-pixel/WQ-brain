import unittest

from scripts.seed_submit_ready import select_submit_ready_for_seed


class TestSeedSubmitReady(unittest.TestCase):
    def test_select_submit_ready_for_seed_keeps_only_submit_ready_candidates(self):
        payload = {
            "batch": {
                "candidates": [
                    {
                        "thesis": "Shock response",
                        "thesis_id": "shock_response",
                        "expression": "rank(ts_zscore(abs(close-vwap),21))",
                        "candidate_score": 0.95,
                        "confidence_score": 0.41,
                        "novelty_score": 0.84,
                        "style_alignment_score": 0.62,
                        "risk_tags": ["turnover_risk", "weight_risk"],
                        "seed_ready": True,
                        "qualified": False,
                        "quality_label": "watchlist",
                        "settings": "USA, TOP1000, Decay 6, Delay 1, Truncation 0.04, Neutralization Industry",
                        "local_metrics": {
                            "verdict": "PASS",
                            "confidence": "MEDIUM",
                            "alpha_score": 75.7,
                            "sharpe": 1.97,
                            "fitness": 2.1,
                            "turnover": 0.13,
                        },
                        "token_program": ["FACTOR_3", "TSZ_21", "RANK"],
                    },
                    {
                        "thesis": "Simple price hypothesis",
                        "thesis_id": "simple_price",
                        "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                        "candidate_score": 1.02,
                        "confidence_score": 0.85,
                        "novelty_score": 0.99,
                        "style_alignment_score": 0.33,
                        "risk_tags": [],
                        "seed_ready": True,
                        "qualified": True,
                        "quality_label": "qualified",
                        "settings": "USA, TOP1000, Decay 4, Delay 1, Truncation 0.03, Neutralization Industry",
                        "local_metrics": {
                            "verdict": "LIKELY_PASS",
                            "confidence": "MEDIUM",
                            "alpha_score": 65.2,
                            "sharpe": 1.58,
                            "fitness": 1.6,
                            "turnover": 0.17,
                        },
                        "token_program": ["CLOSE", "BETA", "REG_RESD_63", "RANK"],
                    },
                    {
                        "thesis": "Weak candidate",
                        "thesis_id": "weak",
                        "expression": "rank(close)",
                        "candidate_score": 0.1,
                        "confidence_score": 0.2,
                        "novelty_score": 0.1,
                        "style_alignment_score": 0.1,
                        "risk_tags": [],
                        "seed_ready": True,
                        "qualified": False,
                        "quality_label": "watchlist",
                        "local_metrics": {
                            "verdict": "FAIL",
                            "confidence": "HIGH",
                            "alpha_score": 20.0,
                            "sharpe": 0.3,
                            "fitness": 0.2,
                            "turnover": 0.1,
                        },
                        "token_program": ["CLOSE", "RANK"],
                    },
                ]
            }
        }

        approved, rejected = select_submit_ready_for_seed(payload, seed_store={}, top=3)
        self.assertEqual(len(approved), 1)
        self.assertEqual(len(rejected), 0)
        self.assertTrue(all(item["seed_source"] == "submit_ready_report" for item in approved))
        self.assertEqual(approved[0]["compiled_expression"], "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))")

    def test_select_submit_ready_for_seed_can_include_auto_fix_submit_ready_candidate(self):
        payload = {"batch": {"candidates": []}}
        extra_candidates = [
            {
                "thesis": "Auto-fix rewrite [Technical Indicator]",
                "thesis_id": "residual_beta",
                "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "candidate_score": 0.69,
                "confidence_score": 0.61,
                "novelty_score": 0.73,
                "style_alignment_score": 0.62,
                "risk_tags": [],
                "seed_ready": True,
                "qualified": True,
                "quality_label": "qualified",
                "source": "auto_fix_rewrite",
                "local_metrics": {
                    "verdict": "PASS",
                    "confidence": "HIGH",
                    "alpha_score": 69.0,
                    "sharpe": 1.5,
                    "fitness": 1.1,
                    "turnover": 0.16,
                },
                "token_program": ["CLOSE", "BETA", "REG_RESD_63", "RANK"],
            }
        ]

        approved, rejected = select_submit_ready_for_seed(payload, seed_store={}, top=3, extra_candidates=extra_candidates)

        self.assertEqual(len(approved), 1)
        self.assertEqual(len(rejected), 0)
        self.assertEqual(approved[0]["compiled_expression"], "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))")
        self.assertEqual(approved[0]["seed_source"], "submit_ready_report")

    def test_select_submit_ready_for_seed_accepts_evaluated_pool_payload(self):
        payload = {
            "candidates": [
                {
                    "thesis": "Residual or de-beta structure",
                    "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                    "seed_ready": True,
                    "qualified": True,
                    "quality_label": "qualified",
                    "confidence_score": 0.74,
                    "local_metrics": {
                        "verdict": "PASS",
                        "confidence": "HIGH",
                        "alpha_score": 78.0,
                        "sharpe": 1.9,
                        "fitness": 1.6,
                        "turnover": 0.14,
                    },
                    "token_program": ["CLOSE", "BETA", "REG_RESD_63", "RANK"],
                }
            ]
        }

        approved, rejected = select_submit_ready_for_seed(payload, seed_store={}, top=3)

        self.assertEqual(len(approved), 1)
        self.assertEqual(len(rejected), 0)


if __name__ == "__main__":
    unittest.main()
