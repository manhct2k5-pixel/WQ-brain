import csv
import os
import tempfile
import unittest
from pathlib import Path

from src.brain import RESULT_COLUMNS
from src.internal_scoring import HistoryIndex, build_internal_metric, score_expression, score_expressions_batch


def _write_surrogate_training_csv(path: Path, *, row_count: int, local_rows: int = 0) -> None:
    headers = RESULT_COLUMNS
    rows = []

    total_rows = row_count + local_rows
    for index in range(total_rows):
        is_local = index >= row_count
        slot = index % 4
        window = 5 + (index % 6) * 7

        if slot == 0:
            expression = f"rank(ts_mean(returns,{window}))"
            fitness = 0.35 + 0.012 * index
            sharpe = 0.45 + 0.015 * index
            returns = 0.03 + 0.0015 * index
            universe = "TOP1000"
            neutralization = "Industry"
        elif slot == 1:
            expression = f"rank(ts_zscore(abs(close-vwap),{window}))"
            fitness = -0.10 + 0.009 * index
            sharpe = -0.15 + 0.012 * index
            returns = -0.01 + 0.0010 * index
            universe = "TOP3000"
            neutralization = "Market"
        elif slot == 2:
            expression = f"rank(ts_corr(volume,returns,{window}))"
            fitness = 0.08 + 0.010 * index
            sharpe = 0.12 + 0.014 * index
            returns = 0.01 + 0.0012 * index
            universe = "TOP1000"
            neutralization = "Subindustry"
        else:
            expression = f"rank(ts_regression(close,beta_last_60_days_spy,{window},lag=0,rettype=0))"
            fitness = 0.22 + 0.011 * index
            sharpe = 0.30 + 0.013 * index
            returns = 0.02 + 0.0014 * index
            universe = "TOP500"
            neutralization = "Industry"

        rows.append(
            {
                "alpha_id": f"{'LOCAL-' if is_local else 'A'}{index:04d}",
                "regular_code": expression,
                "turnover": f"{0.18 + ((index % 5) * 0.04):.4f}",
                "returns": f"{returns:.4f}",
                "drawdown": f"{0.05 + ((index % 4) * 0.02):.4f}",
                "margin": f"{0.0001 * (index + 1):.6f}",
                "fitness": f"{fitness:.2f}",
                "sharpe": f"{sharpe:.2f}",
                "LOW_SHARPE": "PASS" if sharpe >= 1.0 else "FAIL",
                "LOW_FITNESS": "PASS" if fitness >= 1.0 else "FAIL",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS" if sharpe >= 0.7 else "FAIL",
                "SELF_CORRELATION": "PASS",
                "MATCHES_COMPETITION": "PASS",
                "date": f"2026-03-{(index % 27) + 1:02d} 10:00:00",
                "region": "USA",
                "universe": universe,
                "delay": "1",
                "decay": str(4 + (index % 3)),
                "neutralization": neutralization,
                "truncation": "0.02",
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


class TestInternalScoring(unittest.TestCase):
    def test_score_expression_returns_local_metrics(self):
        result = score_expression(
            "rank(winsorize(ts_corr(volume,returns,10),std=5))",
            settings="USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market",
        )

        self.assertTrue(result["alpha_id"].startswith("LOCAL-"))
        self.assertIn("alpha_score", result)
        self.assertIn("ic_proxy", result)
        self.assertIn("ensemble_proxy", result)
        self.assertIn("optimization_hints", result)
        self.assertIn("score_breakdown", result)
        self.assertTrue(result["style_tags"])
        self.assertIn(result["verdict"], {"PASS", "LIKELY_PASS", "BORDERLINE", "FAIL"})
        self.assertIn(result["confidence"], {"HIGH", "MEDIUM", "LOW"})
        self.assertIn(result["alpha_type"], {"mean_reversion", "momentum", "fundamental", "alternative", "hybrid"})
        self.assertEqual(result["settings"]["universe"], "TOP3000")
        self.assertIn(result["LOW_SHARPE"], {"PASS", "FAIL"})
        self.assertEqual(result["score_source"], "internal_proxy_v1")

    def test_duplicate_expression_is_penalized(self):
        history = HistoryIndex()
        expression = "rank(ts_zscore(abs(close-vwap),21))"

        first = score_expression(expression, history_index=history)
        history.observe_expression(expression, first)
        second = score_expression(expression, history_index=history)

        self.assertLess(second["uniqueness_proxy"], first["uniqueness_proxy"])
        self.assertEqual(second["MATCHES_COMPETITION"], "FAIL")
        self.assertEqual(second["SELF_CORRELATION"], "FAIL")

    def test_strong_candidates_do_not_share_the_same_fitness_ceiling(self):
        first = score_expression(
            "rank(winsorize(ts_corr(ts_rank(volume,10),ts_rank(close,10),10),std=5))",
            settings="USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Industry",
        )
        second = score_expression(
            "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
            settings="USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Industry",
        )

        self.assertNotEqual(first["fitness"], second["fitness"])
        self.assertLessEqual(first["fitness"], 3.15)
        self.assertLessEqual(second["fitness"], 3.15)

    def test_verdict_thresholds_separate_strong_and_duplicate_candidates(self):
        history = HistoryIndex()
        strong = score_expression(
            "rank(winsorize(ts_corr(ts_rank(volume,10),ts_rank(close,10),10),std=5))",
            history_index=history,
            settings="USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Industry",
        )
        weak_expr = "rank(ts_zscore(abs(close-vwap),21))"
        weak_first = score_expression(
            weak_expr,
            history_index=history,
            settings="USA, TOP200, Decay 3, Delay 1, Truncation 0.01, Neutralization Subindustry",
        )
        history.observe_expression(weak_expr, weak_first)
        weak_duplicate = score_expression(
            weak_expr,
            history_index=history,
            settings="USA, TOP200, Decay 3, Delay 1, Truncation 0.01, Neutralization Subindustry",
        )

        self.assertIn(strong["verdict"], {"PASS", "LIKELY_PASS"})
        self.assertEqual(weak_duplicate["verdict"], "FAIL")

    def test_metric_writes_local_csv(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                metric = build_internal_metric()
                result = metric("rank(ts_zscore(abs(close-vwap),21))")

                csv_path = Path("simulation_results.csv")
                self.assertTrue(csv_path.exists())
                csv_text = csv_path.read_text(encoding="utf-8")
                self.assertIn("alpha_id", csv_text.splitlines()[0])
                self.assertIn(result["alpha_id"], csv_text)
            finally:
                os.chdir(original_cwd)

    def test_reactive_price_volume_blend_gets_out_of_sample_risk_penalty(self):
        expression = (
            "rank(add(rank(multiply((1 - close / ts_delay(close, 5)), inverse(ts_std_dev(returns, 21)))), "
            "rank(multiply(ts_zscore(close, 21), divide(volume, ts_mean(volume, 63))))))"
        )

        result = score_expression(
            expression,
            settings="USA, TOP1000, Decay 4, Delay 1, Truncation 0.02, Neutralization Subindustry",
        )

        self.assertEqual(result["OUT_OF_SAMPLE_ALIGNMENT"], "FAIL")
        self.assertGreater(result["out_of_sample_risk"], 0.30)
        self.assertLess(result["alpha_score"], 70.0)
        self.assertIn(result["verdict"], {"BORDERLINE", "FAIL"})

    def test_surrogate_shadow_is_ready_with_enough_real_brain_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "simulation_results.csv"
            _write_surrogate_training_csv(csv_path, row_count=64)

            result = score_expression(
                "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                settings="USA, TOP1000, Decay 4, Delay 1, Truncation 0.02, Neutralization Industry",
                surrogate_csv_path=csv_path,
            )

            shadow = result["surrogate_shadow"]
            self.assertEqual(shadow["status"], "ready")
            self.assertEqual(shadow["training_rows"], 64)
            self.assertIsNotNone(shadow["predicted_fitness"])
            self.assertIsNotNone(shadow["predicted_sharpe"])
            self.assertIn(shadow["preview_verdict"], {"PASS", "LIKELY_PASS", "BORDERLINE", "FAIL"})
            self.assertIn(shadow["alignment"], {"aligned", "mixed", "more_cautious", "more_optimistic"})

    def test_surrogate_shadow_reports_insufficient_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "simulation_results.csv"
            _write_surrogate_training_csv(csv_path, row_count=12)

            result = score_expression(
                "rank(ts_mean(returns,21))",
                surrogate_csv_path=csv_path,
            )

            shadow = result["surrogate_shadow"]
            self.assertEqual(shadow["status"], "insufficient_rows")
            self.assertEqual(shadow["training_rows"], 12)
            self.assertEqual(shadow["minimum_rows"], 40)

    def test_surrogate_shadow_ignores_local_proxy_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "simulation_results.csv"
            _write_surrogate_training_csv(csv_path, row_count=0, local_rows=45)

            result = score_expression(
                "rank(ts_mean(returns,21))",
                surrogate_csv_path=csv_path,
            )

            shadow = result["surrogate_shadow"]
            self.assertEqual(shadow["status"], "insufficient_rows")
            self.assertEqual(shadow["training_rows"], 0)
            self.assertEqual(shadow["skipped_local_rows"], 45)

    def test_score_expressions_batch_preserves_order_and_returns_profile(self):
        expressions = [
            "rank(ts_mean(returns,21))",
            "rank(ts_zscore(abs(close-vwap),21))",
        ]

        results, profile = score_expressions_batch(
            expressions,
            settings_list=[None, None],
            max_workers=1,
            return_profile=True,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["expression"], expressions[0])
        self.assertEqual(results[1]["expression"], expressions[1])
        self.assertEqual(profile["mode"], "sequential")
        self.assertEqual(profile["task_count"], 2)
        self.assertEqual(profile["worker_count"], 1)
        self.assertIn(profile["bottleneck_hint"], {"io_and_model_load", "single_process_cpu"})


if __name__ == "__main__":
    unittest.main()
