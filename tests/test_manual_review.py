import unittest

from scripts.manual_review import render_manual_review


class TestManualReview(unittest.TestCase):
    def test_render_manual_review_includes_selected_candidates(self):
        payload = {
            "memory": {
                "style_leaders": [{"tag": "technical", "learning_score": 0.71}],
            },
            "batch": {
                "candidates": [
                    {
                        "thesis": "Technical indicator blend",
                        "thesis_id": "technical_indicator",
                        "why": "Uses oscillator, momentum, and band-style signals inspired by classic factor libraries.",
                        "expression": "rank(ts_zscore(((ts_mean(high, 10)-close)/(ts_mean(high,10)-ts_mean(low,10))),21))",
                        "candidate_score": 1.1,
                        "novelty_score": 0.8,
                        "style_alignment_score": 0.7,
                        "risk_tags": [],
                        "seed_ready": True,
                        "token_program": ["WILLIAMS", "TSZ_21", "RANK"],
                    }
                ]
            },
        }

        markdown = render_manual_review(payload, seed_store={}, top=4)
        self.assertIn("Manual Review Queue", markdown)
        self.assertIn("Review First", markdown)
        self.assertIn("Technical indicator blend", markdown)


if __name__ == "__main__":
    unittest.main()
