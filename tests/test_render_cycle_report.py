import json
import tempfile
import unittest
from pathlib import Path

from scripts.render_cycle_report import build_report


class TestRenderCycleReport(unittest.TestCase):
    def test_build_report_reads_status_and_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            artifacts = repo_root / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "lo_tiep_theo.json").write_text(
                json.dumps(
                    {
                        "memory": {
                            "style_leaders": [
                                {"tag": "technical", "learning_score": 0.72, "avg_research_score": 1.05, "top_hits": 2}
                            ]
                        },
                        "batch": {
                            "candidates": [
                                {
                                    "thesis": "VWAP dislocation",
                                    "thesis_id": "vwap_dislocation",
                                    "expression": "rank(ts_zscore(abs(close-vwap),21))",
                                    "candidate_score": 1.2,
                                    "confidence_score": 0.72,
                                    "qualified": True,
                                    "quality_label": "qualified",
                                    "settings": "USA, TOP200, Decay 3, Delay 1, Truncation 0.02, Neutralization Subindustry",
                                    "local_metrics": {
                                        "verdict": "LIKELY_PASS",
                                        "confidence": "MEDIUM",
                                        "alpha_score": 71.0,
                                        "sharpe": 1.41,
                                        "fitness": 1.18,
                                    },
                                    "novelty_score": 0.9,
                                    "style_alignment_score": 0.8,
                                    "risk_tags": [],
                                    "seed_ready": True,
                                    "token_program": ["FACTOR_3", "TSZ_21", "RANK"],
                                },
                                {
                                    "thesis": "Shock response",
                                    "thesis_id": "shock_response",
                                    "expression": "rank(ts_zscore(divide(ts_std_dev(returns,21),ts_mean(volume, 63)),21))",
                                    "candidate_score": 1.1,
                                    "confidence_score": 0.41,
                                    "qualified": False,
                                    "quality_label": "watchlist",
                                    "settings": "USA, TOP1000, Decay 6, Delay 1, Truncation 0.04, Neutralization Industry",
                                    "local_metrics": {
                                        "verdict": "PASS",
                                        "confidence": "MEDIUM",
                                        "alpha_score": 75.7,
                                        "sharpe": 1.97,
                                        "fitness": 2.1,
                                    },
                                    "novelty_score": 0.84,
                                    "style_alignment_score": 0.62,
                                    "risk_tags": ["turnover_risk", "weight_risk"],
                                    "seed_ready": True,
                                    "token_program": ["RET", "VOL", "RANK"],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = build_report(repo_root)
            self.assertIn("Latest Submit-Ready Alpha Report", report)
            self.assertIn("Candidates shown: 1", report)
            self.assertIn("## Candidate 1", report)
            self.assertIn("VWAP dislocation", report)
            self.assertIn("settings: USA, TOP200, Decay 3, Delay 1, Truncation 0.02, Neutralization Subindustry", report)
            self.assertIn("verdict: LIKELY_PASS (MEDIUM)", report)
            self.assertIn("alpha_score: 71.0", report)
            self.assertIn("sharpe: 1.41", report)


if __name__ == "__main__":
    unittest.main()
