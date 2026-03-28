import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.fix_alpha import build_actionable_auto_fix_candidates, build_auto_fix_payload, build_context, render_fix_report


class Args:
    def __init__(self, **kwargs):
        self.alpha_id = kwargs.get("alpha_id")
        self.csv = kwargs.get("csv")
        self.expression = kwargs.get("expression")
        self.errors = kwargs.get("errors", [])
        self.sharpe = kwargs.get("sharpe")
        self.fitness = kwargs.get("fitness")
        self.turnover = kwargs.get("turnover")
        self.settings = kwargs.get("settings")
        self.auto_rewrite = kwargs.get("auto_rewrite", False)
        self.top_rewrites = kwargs.get("top_rewrites", 5)


class TestFixAlpha(unittest.TestCase):
    def test_build_context_from_expression_and_errors(self):
        args = Args(
            expression="rank(ts_zscore(abs(close-vwap),21))",
            errors=["LOW_SHARPE", "CONCENTRATED_WEIGHT"],
        )
        context = build_context(args)
        self.assertEqual(context["family"], "vwap_dislocation")
        self.assertIn("vwap", context["style_tags"])
        self.assertEqual(context["failures"], ["LOW_SHARPE", "CONCENTRATED_WEIGHT"])

    def test_build_context_from_alpha_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "simulation_results.csv"
            csv_path.write_text(
                "alpha_id,regular_code,turnover,returns,drawdown,margin,fitness,sharpe,"
                "region,universe,delay,decay,neutralization,truncation,"
                "LOW_SHARPE,LOW_FITNESS,LOW_TURNOVER,HIGH_TURNOVER,CONCENTRATED_WEIGHT,"
                "LOW_SUB_UNIVERSE_SHARPE,SELF_CORRELATION,MATCHES_COMPETITION,date\n"
                "A1,\"rank(ts_zscore(abs(close-vwap),21))\",0.5,0.1,0.05,0.03,0.8,0.9,"
                "USA,TOP3000,1,6,Industry,0.03,"
                "FAIL,PASS,PASS,PASS,FAIL,PASS,PASS,PASS,2026-03-25 10:00:00\n",
                encoding="utf-8",
            )
            args = Args(alpha_id="A1", csv=str(csv_path))
            context = build_context(args)
            self.assertEqual(context["alpha_id"], "A1")
            self.assertEqual(context["family"], "vwap_dislocation")
            self.assertIn("LOW_SHARPE", context["failures"])
            self.assertIn("CONCENTRATED_WEIGHT", context["failures"])
            self.assertEqual(context["settings"]["universe"], "TOP3000")
            self.assertEqual(context["settings"]["decay"], "6")

    def test_render_fix_report_includes_rewrite_directions(self):
        context = {
            "alpha_id": None,
            "expression": "rank(ts_zscore(abs(close-vwap),21))",
            "failures": ["LOW_SHARPE", "CONCENTRATED_WEIGHT"],
            "family": "vwap_dislocation",
            "style_tags": {"vwap", "rank", "normalization"},
            "sharpe": 0.9,
            "fitness": 0.7,
            "turnover": 0.4,
        }
        markdown = render_fix_report(context)
        self.assertIn("Alpha Fix Guide", markdown)
        self.assertIn("How To Fix", markdown)
        self.assertIn("Rewrite Directions", markdown)
        self.assertIn("Optimization Loop", markdown)

    @patch(
        "scripts.fix_alpha.build_rewrite_candidates",
        return_value=[
            {
                "family_id": "technical_indicator",
                "family": "Technical Indicator",
                "why": "more normalized trend variant",
                "variant_id": "tech_candidate",
                "expression": "rank(ts_sum(close,10))",
                "style_tags": ["technical", "trend"],
            },
            {
                "family_id": "residual_beta",
                "family": "Residual Beta",
                "why": "less correlated residual variant",
                "variant_id": "residual_candidate",
                "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "style_tags": ["residual", "beta"],
            },
        ],
    )
    @patch("scripts.fix_alpha.score_expressions_batch")
    @patch("scripts.fix_alpha.score_expression")
    def test_build_auto_fix_payload_ranks_submit_ready_then_promising(self, score_mock, score_batch_mock, _rewrite_mock):
        score_mock.return_value = {
            "expression": "rank(ts_zscore(abs(close-vwap),21))",
            "verdict": "FAIL",
            "alpha_score": 52.0,
            "sharpe": 0.9,
            "fitness": 0.7,
            "settings": {"label": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market"},
        }
        score_batch_mock.return_value = [
            {
                "expression": "rank(ts_sum(close,10))",
                "verdict": "PASS",
                "alpha_score": 69.0,
                "sharpe": 1.5,
                "fitness": 1.1,
                "settings": {"label": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market"},
            },
            {
                "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "verdict": "PASS",
                "alpha_score": 61.0,
                "sharpe": 1.25,
                "fitness": 0.9,
                "settings": {"label": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market"},
            },
        ]

        context = {
            "expression": "rank(ts_zscore(abs(close-vwap),21))",
            "failures": ["LOW_SHARPE", "CONCENTRATED_WEIGHT"],
            "family": "vwap_dislocation",
            "style_tags": {"vwap", "rank", "normalization"},
            "settings": {"region": "USA", "universe": "TOP3000"},
            "resolved_csv": None,
        }

        payload = build_auto_fix_payload(context, top_rewrites=2)

        self.assertEqual(payload["candidates"][0]["repair_status"], "submit_ready")
        self.assertEqual(payload["candidates"][0]["expression"], "rank(ts_sum(close,10))")
        self.assertEqual(payload["candidates"][1]["repair_status"], "promising")
        self.assertEqual(payload["candidates"][1]["expression"], "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))")
        self.assertGreater(payload["candidates"][0]["improvement"]["alpha_score_delta"], 0.0)
        self.assertEqual(score_mock.call_count, 1)
        self.assertEqual(score_batch_mock.call_count, 1)

    def test_render_fix_report_includes_auto_rewrite_scoreboard(self):
        context = {
            "alpha_id": None,
            "expression": "rank(ts_zscore(abs(close-vwap),21))",
            "failures": ["LOW_SHARPE"],
            "family": "vwap_dislocation",
            "style_tags": {"vwap", "rank", "normalization"},
            "sharpe": 0.9,
            "fitness": 0.7,
            "turnover": 0.4,
        }
        auto_fix_payload = {
            "settings_label": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market",
            "baseline": {"verdict": "FAIL", "alpha_score": 52.0, "sharpe": 0.9, "fitness": 0.7},
            "candidates": [
                {
                    "family": "Technical Indicator",
                    "variant_id": "tech_candidate",
                    "expression": "rank(ts_sum(close,10))",
                    "why": "more normalized trend variant",
                    "repair_status": "submit_ready",
                    "submit_gap": 0.0,
                    "improvement": {"alpha_score_delta": 17.0, "sharpe_delta": 0.6, "fitness_delta": 0.4},
                    "result": {"alpha_score": 69.0, "sharpe": 1.5, "fitness": 1.1},
                }
            ],
        }

        markdown = render_fix_report(context, auto_fix_payload=auto_fix_payload)

        self.assertIn("Auto Rewrite Scoreboard", markdown)
        self.assertIn("submit_ready", markdown)
        self.assertIn("rank(ts_sum(close,10))", markdown)

    def test_build_actionable_auto_fix_candidates_keeps_only_actionable_repairs(self):
        context = {"alpha_id": "A1", "expression": "rank(ts_zscore(abs(close-vwap),21))"}
        auto_fix_payload = {
            "candidates": [
                {
                    "family": "Technical Indicator",
                    "family_id": "technical_indicator",
                    "variant_id": "tech_candidate",
                    "expression": "rank(ts_sum(close,10))",
                    "why": "more normalized trend variant",
                    "repair_status": "submit_ready",
                    "submit_gap": 0.0,
                    "improvement": {"alpha_score_delta": 17.0, "sharpe_delta": 0.6, "fitness_delta": 0.4},
                    "token_program": ["CLOSE", "TS_SUM_10", "RANK"],
                    "result": {
                        "verdict": "PASS",
                        "confidence": "HIGH",
                        "alpha_score": 69.0,
                        "sharpe": 1.5,
                        "fitness": 1.1,
                        "quality_proxy": 0.63,
                        "stability_proxy": 0.66,
                        "uniqueness_proxy": 0.72,
                        "settings": {"label": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market"},
                    },
                },
                {
                    "family": "VWAP",
                    "family_id": "vwap_dislocation",
                    "variant_id": "weak_candidate",
                    "expression": "rank(close-vwap)",
                    "why": "weak variant",
                    "repair_status": "weak",
                    "submit_gap": 10.0,
                    "improvement": {"alpha_score_delta": 2.0, "sharpe_delta": 0.1, "fitness_delta": 0.05},
                    "token_program": ["CLOSE", "VWAP", "SUB", "RANK"],
                    "result": {
                        "verdict": "FAIL",
                        "confidence": "LOW",
                        "alpha_score": 30.0,
                        "sharpe": 0.4,
                        "fitness": 0.2,
                        "quality_proxy": 0.21,
                        "stability_proxy": 0.18,
                        "uniqueness_proxy": 0.44,
                        "settings": {"label": "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market"},
                    },
                },
            ]
        }

        candidates = build_actionable_auto_fix_candidates(context, auto_fix_payload)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["source"], "auto_fix_rewrite")
        self.assertEqual(candidates[0]["repair_status"], "submit_ready")
        self.assertTrue(candidates[0]["qualified"])
        self.assertEqual(candidates[0]["lineage"]["origin"], "fix")
        self.assertEqual(candidates[0]["lineage"]["family"], "technical_indicator")
        self.assertEqual(
            candidates[0]["lineage"]["parents"][0]["expression"],
            "rank(ts_zscore(abs(close-vwap),21))",
        )
        self.assertEqual(candidates[0]["lineage"]["stage_results"]["planning"]["quality_label"], "qualified")

    def test_build_context_classifies_open_volume_corr_as_technical(self):
        args = Args(
            expression="winsorize(rank(subtract(0,ts_corr(open,volume,10))),std=5)",
            errors=["LOW_SHARPE"],
        )
        context = build_context(args)
        self.assertEqual(context["family"], "technical_indicator")
        self.assertIn("cross_sectional", context["style_tags"])


if __name__ == "__main__":
    unittest.main()
