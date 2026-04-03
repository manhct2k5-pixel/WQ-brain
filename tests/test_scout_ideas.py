import base64
import json
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.scout_ideas import (
    ScoutRequestThrottle,
    aggregate_scout_memory,
    assess_brain_feedback_health,
    assess_report_publish_status,
    apply_source_crowding_adjustments,
    build_brain_feedback_context,
    build_book_ideas,
    build_candidates,
    build_report_archive_paths,
    build_expression_parameter_variants,
    build_fallback_ideas,
    build_payload,
    build_query_profiles,
    build_submitted_alpha_context,
    compute_brain_feedback_penalty,
    compute_submitted_contrast,
    dedupe_ideas,
    fetch_github_readme_ideas,
    fetch_github_ideas,
    fetch_zip_learned_ideas,
    filter_relevant_ideas,
    infer_idea_profile,
    load_brain_feedback_rows,
    load_submitted_alphas,
    load_history_records,
    recommend_settings_profiles,
    render_feedback_health_failure_markdown,
    render_markdown,
    source_key_from_idea,
    summarize_surrogate_shadow,
    summarize_settings_robustness,
)
from src.internal_scoring import HistoryIndex


class TestScoutIdeas(unittest.TestCase):
    def test_recommend_settings_profiles_avoids_subindustry_outputs(self):
        profiles = recommend_settings_profiles(["vwap_dislocation"], "medium", ["liquidity"])

        self.assertTrue(profiles)
        self.assertTrue(all("Neutralization Subindustry" not in item for item in profiles))
        self.assertTrue(any("Neutralization Industry" in item for item in profiles))

    def test_build_query_profiles_wide_expands_search_space(self):
        memory = {"style_leaders": [{"tag": "volume"}, {"tag": "residual"}, {"tag": "correlation"}]}
        scout_memory = {"preferred_family_horizons": {"residual_beta": "long", "pv_divergence": "medium"}}

        focused = build_query_profiles(memory, scout_memory, search_breadth="focused")
        wide = build_query_profiles(memory, scout_memory, search_breadth="wide")
        explore = build_query_profiles(memory, scout_memory, search_breadth="explore")

        self.assertGreater(len(wide), len(focused))
        self.assertGreater(len(explore), len(wide))
        self.assertEqual(len({item["query"] for item in wide}), len(wide))
        self.assertEqual(len({item["query"] for item in explore}), len(explore))
        self.assertTrue(any("market neutral residual stock selection factor" == item["query"] for item in wide))
        self.assertTrue(any("order flow liquidity imbalance stock returns" == item["query"] for item in wide))
        self.assertTrue(any("closing location value stock reversal anomaly" == item["query"] for item in explore))

    def test_build_report_archive_paths_groups_by_day_and_time(self):
        paths = build_report_archive_paths(datetime(2026, 3, 26, 8, 45), archive_root="artifacts/bao_cao_ngay")
        self.assertEqual(paths["root"], Path("artifacts/bao_cao_ngay/2026-03-26/08h"))
        self.assertEqual(paths["markdown"], Path("artifacts/bao_cao_ngay/2026-03-26/08h/trinh_sat_hang_ngay.md"))

    def test_build_report_archive_paths_can_still_rotate_per_run(self):
        paths = build_report_archive_paths(
            datetime(2026, 3, 26, 8, 45),
            archive_root="artifacts/bao_cao_ngay",
            archive_frequency="run",
        )
        self.assertEqual(paths["root"], Path("artifacts/bao_cao_ngay/2026-03-26/08h_45p"))

    def test_assess_report_publish_status_blocks_second_report_within_interval(self):
        status = assess_report_publish_status(
            datetime(2026, 3, 26, 10, 45),
            reportable_count=2,
            report_interval_minutes=60,
            report_state={"last_published_at": "2026-03-26T10:00:00"},
        )
        self.assertFalse(status["published"])
        self.assertEqual(status["reason"], "report_interval_not_elapsed")
        self.assertEqual(status["next_publish_after"], "2026-03-26T11:00:00")
        self.assertEqual(status["minutes_until_next"], 15)

    def test_assess_report_publish_status_allows_first_report_after_interval(self):
        status = assess_report_publish_status(
            datetime(2026, 3, 26, 11, 2),
            reportable_count=1,
            report_interval_minutes=60,
            report_state={"last_published_at": "2026-03-26T10:00:00"},
        )
        self.assertTrue(status["published"])
        self.assertEqual(status["reason"], "reportable_pick_found")

    def test_assess_report_publish_status_skips_when_no_reportable_pick(self):
        status = assess_report_publish_status(
            datetime(2026, 3, 26, 11, 2),
            reportable_count=0,
            report_interval_minutes=60,
            report_state={"last_published_at": "2026-03-26T10:00:00"},
        )
        self.assertFalse(status["published"])
        self.assertEqual(status["reason"], "no_reportable_pick")

    def test_infer_idea_profile_maps_keywords_to_supported_families(self):
        idea = {
            "query": "price volume anomaly equities",
            "title": "Price-Volume Momentum and Volatility Shock in Equities",
            "summary": "A momentum signal conditioned on liquidity, volume, and volatility shock.",
            "bias_families": ["pv_divergence"],
            "bias_style_tags": ["volume"],
            "bias_horizon": "medium",
            "source_score": 0.82,
        }

        profile = infer_idea_profile(idea)
        self.assertIn("pv_divergence", profile["families"])
        self.assertIn("shock_response", profile["families"])
        self.assertIn("volume", profile["style_tags"])
        self.assertTrue(profile["generation_ok"])

    def test_infer_idea_profile_maps_book_style_simple_hypothesis(self):
        idea = {
            "query": "simple price hypothesis",
            "title": "Simple Price-Delay Ratio and Inverse Price Ideas",
            "summary": "Use price-delay, inverse price, ranking, decay, neutralization, and ratio-like predictors.",
            "bias_families": ["simple_price_patterns"],
            "bias_style_tags": ["simple", "ratio_like"],
            "bias_horizon": "short",
            "source_score": 0.86,
        }

        profile = infer_idea_profile(idea)
        self.assertIn("simple_price_patterns", profile["families"])
        self.assertIn("ratio_like", profile["style_tags"])
        self.assertTrue(profile["generation_ok"])

    def test_filter_relevant_ideas_rejects_broad_non_specific_sources(self):
        ideas = [
            {
                "source": "openalex",
                "query": "strategic asset allocation",
                "title": "Strategic Asset Allocation: Portfolio Choice for Long-Term Investors",
                "summary": "Portfolio choice and strategic allocation for investors.",
                "year": 2003,
                "citations": 10,
                "relevance": 900.0,
                "bias_families": ["technical_indicator"],
                "bias_style_tags": ["momentum"],
                "bias_horizon": "long",
                "source_score": 0.78,
            }
        ]

        filtered = filter_relevant_ideas(ideas, scout_memory={})
        self.assertEqual(filtered, [])

    def test_infer_idea_profile_relaxes_gate_for_finance_specific_github_repo(self):
        idea = {
            "source": "github",
            "query": "equity factor momentum anomaly",
            "title": "quant/systematic-trading-screens",
            "summary": "Momentum screens and technical indicators for systematic trading.",
            "github_full_name": "quant/systematic-trading-screens",
            "url": "https://github.com/quant/systematic-trading-screens",
            "bias_families": ["technical_indicator"],
            "bias_style_tags": ["momentum"],
            "bias_horizon": "medium",
            "source_score": 0.74,
        }

        profile = infer_idea_profile(idea)
        self.assertLess(profile["equity_factor_relevance"], 0.36)
        self.assertTrue(profile["generation_ok"])
        self.assertIn("technical_indicator", profile["families"])

    def test_filter_relevant_ideas_still_rejects_generic_github_repo(self):
        ideas = [
            {
                "source": "github",
                "query": "equity factor momentum anomaly",
                "title": "user/openreviewdata",
                "summary": "Crawl and visualize conference papers and reviews.",
                "github_full_name": "user/openreviewdata",
                "url": "https://github.com/user/openreviewdata",
                "bias_families": ["technical_indicator"],
                "bias_style_tags": ["momentum"],
                "bias_horizon": "medium",
                "source_score": 0.74,
            }
        ]

        filtered = filter_relevant_ideas(ideas, scout_memory={})
        self.assertEqual(filtered, [])

    def test_fetch_zip_learned_ideas_imports_seed_expressions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "worldquant-miner-test.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "worldquant-miner-master/generation_one/naive-ollama/mined_expressions.json",
                    json.dumps(
                        [
                            {
                                "expression": "rank(ts_mean(close, 20))",
                                "result": {"settings": {"region": "USA", "universe": "TOP3000", "neutralization": "INDUSTRY"}},
                            }
                        ]
                    ),
                )
                archive.writestr(
                    "worldquant-miner-master/generation_one/consultant-templates-api/operatorRAW.json",
                    json.dumps([{"name": "rank"}, {"name": "ts_mean"}, {"name": "subtract"}]),
                )

            ideas, status = fetch_zip_learned_ideas(zip_path, max_seed_ideas=2)

        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["seed_expression_count"], 1)
        self.assertEqual(status["operator_count"], 3)
        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0]["source"], "zip_knowledge")
        self.assertGreaterEqual(len(ideas[0]["seed_expressions"]), 1)

    @patch("scripts.scout_ideas.requests.get")
    def test_fetch_github_readme_ideas_learns_from_readme(self, mock_get):
        readme_text = """# Quant Repo

Momentum alpha examples.

```text
rank(ts_mean(close, 20))
```
"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            "content": base64.b64encode(readme_text.encode("utf-8")).decode("ascii"),
            "html_url": "https://github.com/quant/research-alpha/blob/main/README.md",
        }

        ideas, status = fetch_github_readme_ideas(
            [
                {
                    "source": "github",
                    "query": "equity factor momentum anomaly",
                    "title": "quant/research-alpha",
                    "summary": "Momentum research repo",
                    "github_full_name": "quant/research-alpha",
                    "url": "https://github.com/quant/research-alpha",
                    "year": 2025,
                    "citations": 14,
                    "relevance": 800.0,
                    "bias_families": ["technical_indicator"],
                    "bias_style_tags": ["momentum"],
                    "bias_horizon": "medium",
                    "source_score": 0.72,
                }
            ],
            limit=1,
        )

        self.assertEqual(status["attempted"], 1)
        self.assertEqual(status["succeeded"], 1)
        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0]["source"], "github_readme")
        self.assertIn("rank(ts_mean(close, 20))", ideas[0]["seed_expressions"])

    @patch("scripts.scout_ideas.time.sleep", return_value=None)
    @patch("scripts.scout_ideas.requests.get")
    def test_fetch_github_readme_ideas_retries_after_429(self, mock_get, _sleep):
        limited = Mock(status_code=429, headers={"Retry-After": "0"}, text="rate limited")
        success = Mock(status_code=200, headers={}, text="")
        success.json.return_value = {
            "content": base64.b64encode(b"rank(ts_mean(close, 20))").decode("ascii"),
            "html_url": "https://github.com/quant/research-alpha/blob/main/README.md",
        }
        mock_get.side_effect = [limited, success]

        ideas, status = fetch_github_readme_ideas(
            [
                {
                    "source": "github",
                    "query": "equity factor momentum anomaly",
                    "title": "quant/research-alpha",
                    "summary": "Momentum research repo",
                    "github_full_name": "quant/research-alpha",
                    "url": "https://github.com/quant/research-alpha",
                    "year": 2025,
                    "citations": 14,
                    "relevance": 800.0,
                    "bias_families": ["technical_indicator"],
                    "bias_style_tags": ["momentum"],
                    "bias_horizon": "medium",
                    "source_score": 0.72,
                }
            ],
            limit=1,
            max_retries=2,
        )

        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(status["quota_limited"], 1)
        self.assertEqual(status["succeeded"], 1)
        self.assertEqual(len(ideas), 1)

    @patch("scripts.scout_ideas.requests.get")
    def test_fetch_github_ideas_respects_request_budget(self, mock_get):
        success = Mock(status_code=200, headers={}, text="")
        success.json.return_value = {"items": []}
        mock_get.return_value = success

        _, status = fetch_github_ideas(
            [
                {"query": "alpha one", "families": [], "style_tags": [], "horizon": "medium"},
                {"query": "alpha two", "families": [], "style_tags": [], "horizon": "medium"},
            ],
            per_query=1,
            max_queries=2,
            request_throttle=ScoutRequestThrottle(max_requests=1, request_delay_seconds=0.0),
        )

        self.assertEqual(mock_get.call_count, 1)
        self.assertTrue(status["budget_exhausted"])
        self.assertEqual(status["network_requests"], 1)

    def test_build_expression_parameter_variants_creates_nearby_windows(self):
        variants = build_expression_parameter_variants("ts_std_dev(anl4_fcf_flag, 50)", max_variants=3)
        self.assertIn("ts_std_dev(anl4_fcf_flag, 50)", variants)
        self.assertTrue(any(item.endswith("40)") or item.endswith("60)") or item.endswith("45)") or item.endswith("55)") for item in variants))

    def test_build_candidates_returns_scored_local_candidates(self):
        ideas = filter_relevant_ideas(
            [
                {
                    "source": "openalex",
                    "query": "price volume anomaly equities",
                    "title": "Price-Volume Momentum and Volatility Shock in Equities",
                    "summary": "A momentum signal conditioned on liquidity, volume, and volatility shock.",
                    "url": "https://example.com/paper",
                    "year": 2024,
                    "citations": 12,
                    "relevance": 900.0,
                    "bias_families": ["pv_divergence", "shock_response"],
                    "bias_style_tags": ["volume", "volatility"],
                    "bias_horizon": "medium",
                    "source_score": 0.82,
                }
            ],
            scout_memory={},
        )
        memory = {"style_leaders": [], "failure_counts": {}}

        candidates = build_candidates(ideas, memory=memory, scout_memory={}, history_index=HistoryIndex())

        self.assertGreater(len(candidates), 0)
        candidate = candidates[0]
        self.assertIn("expression", candidate)
        self.assertIn("settings", candidate)
        self.assertIn("local_metrics", candidate)
        self.assertIn("robustness_score", candidate)
        self.assertIn("settings_robustness", candidate)
        self.assertGreater(len(candidate["settings_evaluated"]), 1)
        best_alpha_score = candidate["local_metrics"]["alpha_score"]
        self.assertEqual(
            best_alpha_score,
            max(item["alpha_score"] for item in candidate["settings_evaluated"]),
        )

    def test_build_candidates_injects_contrast_family_for_reactive_idea(self):
        ideas = filter_relevant_ideas(
            [
                {
                    "source": "openalex",
                    "query": "equity factor momentum anomaly",
                    "title": "Momentum and Trend Equity Anomaly",
                    "summary": "A cross-sectional stock momentum signal with trend persistence in equities.",
                    "url": "https://example.com/momentum-paper",
                    "year": 2024,
                    "citations": 10,
                    "relevance": 880.0,
                    "bias_families": ["technical_indicator"],
                    "bias_style_tags": ["momentum", "technical"],
                    "bias_horizon": "long",
                    "source_score": 0.84,
                }
            ],
            scout_memory={},
        )

        candidates = build_candidates(
            ideas,
            memory={"style_leaders": [], "failure_counts": {}},
            scout_memory={},
            history_index=HistoryIndex(),
        )

        self.assertTrue(any(item["thesis_id"] == "residual_beta" for item in candidates))

    def test_submitted_alpha_context_summarizes_alpha_types(self):
        context = build_submitted_alpha_context(
            [
                {"expression": "rank(-ts_delta(close, 5)) * rank(volume / ts_mean(volume, 20))"},
                {"expression": "rank(-ts_delta(close, 2))"},
                {"expression": "ts_rank(operating_income / cap, 252)"},
            ]
        )

        self.assertEqual(context["count"], 3)
        self.assertIn("fundamental", context["alpha_type_counts"])
        self.assertIn("rank", context["style_tag_counts"])

    def test_load_submitted_alphas_accepts_json_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submitted.json"
            path.write_text(
                '[{"label":"alpha_1","expression":"rank(-ts_delta(close, 5))"}, "ts_rank(operating_income / cap, 252)"]',
                encoding="utf-8",
            )

            loaded = load_submitted_alphas(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["label"], "alpha_1")
            self.assertEqual(loaded[1]["label"], "submitted_2")

    def test_source_key_prefers_stable_ids(self):
        openalex = {
            "source": "openalex",
            "title": "Paper A",
            "openalex_id": "https://openalex.org/W1234567890",
            "doi": "https://doi.org/10.1000/test-doi",
        }
        arxiv = {
            "source": "arxiv",
            "title": "Paper B",
            "url": "https://arxiv.org/abs/2401.12345",
        }
        github = {
            "source": "github",
            "title": "Repo C",
            "github_full_name": "quant/research-alpha",
            "url": "https://github.com/quant/research-alpha",
        }

        self.assertEqual(source_key_from_idea(openalex), "doi:10.1000/test-doi")
        self.assertEqual(source_key_from_idea(arxiv), "arxiv:2401.12345")
        self.assertEqual(source_key_from_idea(github), "gh:quant/research-alpha")

    def test_dedupe_ideas_uses_stable_source_ids(self):
        ideas = [
            {
                "source": "openalex",
                "title": "First title variant",
                "doi": "https://doi.org/10.1000/test-doi",
                "openalex_id": "https://openalex.org/W111",
            },
            {
                "source": "openalex",
                "title": "Second title variant",
                "doi": "10.1000/test-doi",
                "openalex_id": "https://openalex.org/W222",
            },
            {
                "source": "github",
                "title": "Repo one",
                "github_full_name": "quant/research-alpha",
                "url": "https://github.com/quant/research-alpha",
            },
            {
                "source": "github",
                "title": "Repo one renamed text",
                "url": "https://github.com/quant/research-alpha",
            },
        ]

        deduped = dedupe_ideas(ideas)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(source_key_from_idea(deduped[0]), "doi:10.1000/test-doi")
        self.assertEqual(source_key_from_idea(deduped[1]), "gh:quant/research-alpha")

    def test_load_brain_feedback_rows_reports_schema_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "simulation_results.csv"
            path.write_text("alpha_id,fitness\nalpha_1,1.2\n", encoding="utf-8")

            rows, status = load_brain_feedback_rows(path)

            self.assertEqual(rows, [])
            self.assertEqual(status["status"], "schema_error")
            self.assertIn("regular_code", status["message"])

    def test_load_brain_feedback_rows_auto_backfills_legacy_context_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "simulation_results.csv"
            path.write_text(
                "alpha_id,regular_code,turnover,returns,drawdown,margin,fitness,sharpe,"
                "LOW_SHARPE,LOW_FITNESS,LOW_TURNOVER,HIGH_TURNOVER,CONCENTRATED_WEIGHT,"
                "LOW_SUB_UNIVERSE_SHARPE,SELF_CORRELATION,MATCHES_COMPETITION,date\n"
                "A1,rank(close),0.2,0.1,0.05,0.03,1.4,1.8,"
                "PASS,PASS,PASS,PASS,PASS,PASS,PASS,PASS,2026-03-25 10:00:00\n",
                encoding="utf-8",
            )

            rows, status = load_brain_feedback_rows(path)

            self.assertEqual(status["status"], "ok")
            self.assertEqual(status["missing_context_columns"], [])
            self.assertIn("Backfilled legacy context columns", status["message"])
            self.assertEqual(rows[0]["region"], "USA")
            self.assertEqual(rows[0]["universe"], "TOP3000")

    def test_assess_brain_feedback_health_marks_missing_context_columns_as_degraded(self):
        rows = [
            {
                "regular_code": "rank(close)",
                "sharpe": "1.1",
                "fitness": "0.9",
                "returns": "0.03",
            }
        ]
        status = {
            "status": "ok",
            "message": "Loaded 1 simulation rows.",
            "context_columns_present": [],
            "missing_context_columns": ["region", "universe"],
        }

        assessed = assess_brain_feedback_health(rows, status, brain_feedback_context={})

        self.assertEqual(assessed["health"], "degraded")
        self.assertTrue(assessed["hard_block"])
        self.assertEqual(assessed["reason"], "missing_context_columns")
        self.assertEqual(assessed["recommended_action"], "upgrade_feedback_schema")

    def test_assess_brain_feedback_health_marks_contextual_feedback_as_healthy(self):
        rows = [
            {
                "regular_code": "rank(close)",
                "sharpe": "1.4",
                "fitness": "1.1",
                "returns": "0.05",
                "region": "USA",
                "universe": "TOP3000",
                "delay": "1",
                "decay": "6",
                "neutralization": "Subindustry",
                "truncation": "0.05",
            }
        ]
        status = {
            "status": "ok",
            "message": "Loaded 1 simulation rows.",
            "context_columns_present": ["decay", "delay", "neutralization", "region", "truncation", "universe"],
            "missing_context_columns": [],
        }
        context = build_brain_feedback_context(rows)

        assessed = assess_brain_feedback_health(rows, status, brain_feedback_context=context)

        self.assertEqual(assessed["health"], "healthy")
        self.assertFalse(assessed["hard_block"])
        self.assertEqual(assessed["reason"], "ok")
        self.assertEqual(assessed["recommended_action"], "continue")
        self.assertGreaterEqual(assessed["context_row_count"], 1)
        self.assertGreaterEqual(assessed["distinct_context_count"], 1)

    def test_render_feedback_health_failure_markdown_surfaces_action(self):
        markdown = render_feedback_health_failure_markdown(
            run_timestamp="2026-03-26 23h 59'",
            brain_feedback_status={
                "status": "ok",
                "health": "degraded",
                "message": "Loaded 128 simulation rows. Missing context columns: region, universe.",
                "missing_context_columns": ["region", "universe"],
                "recommended_action": "upgrade_feedback_schema",
            },
        )

        self.assertIn("Feedback Health Check", markdown)
        self.assertIn("Brain feedback health: degraded", markdown)
        self.assertIn("Missing context columns: region, universe", markdown)
        self.assertIn("Recommended action: upgrade_feedback_schema", markdown)

    def test_settings_robustness_rewards_consistency(self):
        robust = summarize_settings_robustness(
            [
                {"verdict": "PASS", "alpha_score": 82.0},
                {"verdict": "PASS", "alpha_score": 80.0},
                {"verdict": "LIKELY_PASS", "alpha_score": 78.0},
            ]
        )
        fragile = summarize_settings_robustness(
            [
                {"verdict": "PASS", "alpha_score": 88.0},
                {"verdict": "FAIL", "alpha_score": 54.0},
                {"verdict": "FAIL", "alpha_score": 49.0},
            ]
        )

        self.assertGreater(robust["robustness_score"], fragile["robustness_score"])
        self.assertGreater(robust["pass_rate"], fragile["pass_rate"])

    def test_submitted_contrast_rewards_less_saturated_styles(self):
        submitted_context = {
            "count": 4,
            "alpha_type_counts": {"momentum": 3, "hybrid": 1},
            "style_tag_counts": {"rank": 4, "liquidity": 3, "residual": 1},
            "saturated_alpha_types": ["momentum"],
            "dominant_style_tags": ["rank", "liquidity"],
        }

        crowded = compute_submitted_contrast(
            {"alpha_type": "momentum", "style_tags": ["rank", "liquidity", "momentum"]},
            submitted_context,
        )
        fresh = compute_submitted_contrast(
            {"alpha_type": "hybrid", "style_tags": ["residual", "beta", "correlation"]},
            submitted_context,
        )

        self.assertGreater(fresh["contrast_score"], crowded["contrast_score"])
        self.assertTrue(any("fresh_style_tags=" in reason for reason in fresh["reasons"]))

    def test_source_crowding_penalty_hits_lower_ranked_same_source_family_variants(self):
        candidates = [
            {
                "expression": "rank(ts_mean(close, 10))",
                "source_key": "paper-a",
                "thesis_id": "technical_indicator",
                "candidate_score": 0.80,
                "confidence_score": 0.78,
                "submitted_diversity_score": 0.90,
                "submitted_contrast_score": 0.55,
                "robustness_score": 0.82,
                "novelty_score": 0.70,
                "local_metrics": {"verdict": "PASS", "alpha_score": 81.0},
                "risk_tags": [],
            },
            {
                "expression": "rank(ts_mean(close, 20))",
                "source_key": "paper-a",
                "thesis_id": "technical_indicator",
                "candidate_score": 0.77,
                "confidence_score": 0.75,
                "submitted_diversity_score": 0.89,
                "submitted_contrast_score": 0.55,
                "robustness_score": 0.81,
                "novelty_score": 0.68,
                "local_metrics": {"verdict": "PASS", "alpha_score": 79.0},
                "risk_tags": [],
            },
            {
                "expression": "rank(ts_mean(close, 30))",
                "source_key": "paper-a",
                "thesis_id": "technical_indicator",
                "candidate_score": 0.73,
                "confidence_score": 0.72,
                "submitted_diversity_score": 0.88,
                "submitted_contrast_score": 0.55,
                "robustness_score": 0.79,
                "novelty_score": 0.66,
                "local_metrics": {"verdict": "PASS", "alpha_score": 77.0},
                "risk_tags": [],
            },
        ]

        adjusted = apply_source_crowding_adjustments(candidates)

        self.assertEqual(adjusted[0]["source_crowding_penalty"], 0.0)
        self.assertEqual(adjusted[1]["source_crowding_penalty"], 0.0)
        self.assertGreater(adjusted[2]["source_crowding_penalty"], 0.0)
        self.assertIn("source_crowding_risk", adjusted[2]["risk_tags"])

    def test_selection_prefers_more_robust_candidate_when_quality_is_close(self):
        candidates = [
            {
                "expression": "rank(ts_zscore(close, 21))",
                "source_key": "paper-robust",
                "thesis_id": "technical_indicator",
                "thesis_family_ids": ["technical_indicator"],
                "risk_tags": [],
                "robustness_score": 0.84,
                "settings_robustness": {
                    "settings_count": 3,
                    "pass_rate": 1.0,
                    "avg_alpha_score": 81.0,
                    "min_alpha_score": 79.0,
                    "max_alpha_score": 83.0,
                    "alpha_spread": 4.0,
                    "robustness_score": 0.84,
                },
                "confidence_score": 0.82,
                "candidate_score": 0.83,
                "source_quality_score": 0.72,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 82.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.72,
                },
            },
            {
                "expression": "rank(ts_zscore(volume, 21))",
                "source_key": "paper-fragile",
                "thesis_id": "technical_indicator",
                "thesis_family_ids": ["technical_indicator"],
                "risk_tags": [],
                "robustness_score": 0.39,
                "settings_robustness": {
                    "settings_count": 3,
                    "pass_rate": 0.33,
                    "avg_alpha_score": 75.0,
                    "min_alpha_score": 52.0,
                    "max_alpha_score": 84.0,
                    "alpha_spread": 32.0,
                    "robustness_score": 0.39,
                },
                "confidence_score": 0.83,
                "candidate_score": 0.84,
                "source_quality_score": 0.73,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 84.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.74,
                },
            },
        ]

        payload, selected, _, _ = build_payload(
            ideas=[],
            candidates=candidates,
            memory={"style_leaders": [], "failure_counts": {}},
            count=1,
            seed_store={},
            include_watchlist=False,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["expression"], "rank(ts_zscore(close, 21))")
        self.assertEqual(payload["batch"]["qualified_count"], 1)

    def test_selection_prefers_lower_submitted_overlap(self):
        candidates = [
            {
                "expression": "rank(-ts_delta(close, 5)) * rank(volume / ts_mean(volume, 20))",
                "source_key": "paper-overlap",
                "thesis_id": "pv_divergence",
                "thesis_family_ids": ["pv_divergence"],
                "risk_tags": [],
                "robustness_score": 0.88,
                "submitted_diversity_score": 0.18,
                "submitted_overlap_score": 0.82,
                "submitted_overlap_reasons": ["submitted_alpha_type=hybridx3"],
                "settings_robustness": {
                    "settings_count": 3,
                    "pass_rate": 1.0,
                    "avg_alpha_score": 84.0,
                    "min_alpha_score": 82.0,
                    "max_alpha_score": 86.0,
                    "alpha_spread": 4.0,
                    "robustness_score": 0.88,
                },
                "confidence_score": 0.84,
                "candidate_score": 0.85,
                "source_quality_score": 0.74,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 85.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.73,
                },
            },
            {
                "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "source_key": "paper-diverse",
                "thesis_id": "residual_beta",
                "thesis_family_ids": ["residual_beta"],
                "risk_tags": [],
                "robustness_score": 0.81,
                "submitted_diversity_score": 0.92,
                "submitted_overlap_score": 0.08,
                "submitted_overlap_reasons": [],
                "settings_robustness": {
                    "settings_count": 3,
                    "pass_rate": 0.67,
                    "avg_alpha_score": 80.0,
                    "min_alpha_score": 76.0,
                    "max_alpha_score": 83.0,
                    "alpha_spread": 7.0,
                    "robustness_score": 0.81,
                },
                "confidence_score": 0.8,
                "candidate_score": 0.81,
                "source_quality_score": 0.72,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 82.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.76,
                },
            },
        ]

        payload, selected, _, _ = build_payload(
            ideas=[],
            candidates=candidates,
            memory={"style_leaders": [], "failure_counts": {}},
            count=1,
            seed_store={},
            submitted_context={
                "count": 3,
                "entries": [],
                "skeletons": [],
                "alpha_type_counts": {"hybrid": 3},
                "style_tag_counts": {"volume": 2, "rank": 2},
                "saturated_alpha_types": ["hybrid"],
                "dominant_style_tags": ["volume", "rank"],
            },
            include_watchlist=False,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["expression"], "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))")
        self.assertEqual(payload["batch"]["qualified_count"], 1)

    def test_selection_prefers_family_diversity_when_scores_are_close(self):
        candidates = [
            {
                "expression": "rank(ts_zscore(close, 21))",
                "source_key": "paper-crowded",
                "thesis_id": "technical_indicator",
                "thesis_family_ids": ["technical_indicator"],
                "risk_tags": [],
                "family_diversity_score": 0.18,
                "robustness_score": 0.84,
                "submitted_diversity_score": 0.90,
                "confidence_score": 0.84,
                "candidate_score": 0.83,
                "source_quality_score": 0.74,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 84.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.74,
                    "style_tags": ["trend", "technical", "normalization", "rank"],
                },
            },
            {
                "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "source_key": "paper-contrast",
                "thesis_id": "residual_beta",
                "thesis_family_ids": ["residual_beta"],
                "risk_tags": [],
                "family_diversity_score": 0.95,
                "robustness_score": 0.81,
                "submitted_diversity_score": 0.89,
                "confidence_score": 0.81,
                "candidate_score": 0.79,
                "source_quality_score": 0.73,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 81.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.78,
                    "style_tags": ["residual", "beta", "correlation"],
                },
            },
        ]

        _, selected, _, _ = build_payload(
            ideas=[],
            candidates=candidates,
            memory={"style_leaders": [], "failure_counts": {}},
            count=1,
            seed_store={},
            include_watchlist=False,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["expression"], "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))")

    def test_selection_uses_diversity_aware_second_pick(self):
        candidates = [
            {
                "expression": "rank(ts_zscore(close, 21))",
                "source_key": "paper-a",
                "thesis_id": "technical_indicator",
                "thesis_family_ids": ["technical_indicator"],
                "risk_tags": [],
                "submitted_diversity_score": 0.88,
                "robustness_score": 0.84,
                "confidence_score": 0.86,
                "candidate_score": 0.86,
                "source_quality_score": 0.74,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 84.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.75,
                    "style_tags": ["trend", "technical", "normalization", "rank"],
                },
            },
            {
                "expression": "rank(ts_zscore(close, 20))",
                "source_key": "paper-b",
                "thesis_id": "technical_indicator",
                "thesis_family_ids": ["technical_indicator"],
                "risk_tags": [],
                "submitted_diversity_score": 0.89,
                "robustness_score": 0.83,
                "confidence_score": 0.85,
                "candidate_score": 0.85,
                "source_quality_score": 0.74,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 83.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.74,
                    "style_tags": ["trend", "technical", "normalization", "rank"],
                },
            },
            {
                "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "source_key": "paper-c",
                "thesis_id": "residual_beta",
                "thesis_family_ids": ["residual_beta"],
                "risk_tags": [],
                "submitted_diversity_score": 0.93,
                "robustness_score": 0.81,
                "confidence_score": 0.82,
                "candidate_score": 0.8,
                "source_quality_score": 0.72,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 81.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.78,
                    "style_tags": ["residual", "beta", "correlation"],
                },
            },
        ]

        _, selected, _, _ = build_payload(
            ideas=[],
            candidates=candidates,
            memory={"style_leaders": [], "failure_counts": {}},
            count=2,
            seed_store={},
            include_watchlist=False,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["expression"], "rank(ts_zscore(close, 21))")
        self.assertEqual(selected[1]["expression"], "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))")

    def test_reportable_candidates_can_exceed_selection_count_when_scores_stay_high(self):
        candidates = [
            {
                "expression": "rank(ts_zscore(close, 21))",
                "source_key": "paper-a",
                "thesis": "Technical indicator A",
                "thesis_id": "technical_indicator",
                "thesis_family_ids": ["technical_indicator"],
                "risk_tags": [],
                "family_diversity_score": 0.82,
                "submitted_diversity_score": 0.92,
                "robustness_score": 0.86,
                "confidence_score": 0.86,
                "candidate_score": 0.87,
                "source_quality_score": 0.76,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 84.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.76,
                    "style_tags": ["trend", "technical", "rank"],
                },
            },
            {
                "expression": "rank(ts_zscore(close, 20))",
                "source_key": "paper-b",
                "thesis": "Technical indicator B",
                "thesis_id": "technical_indicator",
                "thesis_family_ids": ["technical_indicator"],
                "risk_tags": [],
                "family_diversity_score": 0.81,
                "submitted_diversity_score": 0.91,
                "robustness_score": 0.855,
                "confidence_score": 0.855,
                "candidate_score": 0.865,
                "source_quality_score": 0.75,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 83.5,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.75,
                    "style_tags": ["trend", "technical", "rank"],
                },
            },
        ]

        payload, selected, _, _ = build_payload(
            ideas=[],
            candidates=candidates,
            memory={"style_leaders": [], "failure_counts": {}},
            count=1,
            seed_store={},
            include_watchlist=False,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(payload["batch"]["reportable_count"], 2)
        self.assertEqual(len(payload["reportable_selected"]), 2)

    def test_selection_relaxed_fill_promotes_safe_near_threshold_candidate(self):
        candidates = [
            {
                "expression": "rank(ts_zscore(close, 21))",
                "source_key": "paper-a",
                "thesis_id": "technical_indicator",
                "thesis_family_ids": ["technical_indicator"],
                "risk_tags": [],
                "brain_feedback_penalty": 0.04,
                "submitted_diversity_score": 0.9,
                "robustness_score": 0.85,
                "confidence_score": 0.84,
                "candidate_score": 0.85,
                "source_quality_score": 0.75,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 84.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.75,
                    "style_tags": ["trend", "technical", "normalization", "rank"],
                },
            },
            {
                "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "source_key": "paper-b",
                "thesis_id": "residual_beta",
                "thesis_family_ids": ["residual_beta"],
                "risk_tags": [],
                "brain_feedback_penalty": 0.26,
                "submitted_diversity_score": 0.94,
                "robustness_score": 0.79,
                "confidence_score": 0.55,
                "candidate_score": 0.77,
                "source_quality_score": 0.71,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 79.0,
                    "confidence": "MEDIUM",
                    "uniqueness_proxy": 0.77,
                    "style_tags": ["residual", "beta", "correlation"],
                },
            },
        ]

        payload, selected, _, _ = build_payload(
            ideas=[],
            candidates=candidates,
            memory={"style_leaders": [], "failure_counts": {}},
            count=2,
            seed_store={},
            include_watchlist=False,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(payload["batch"]["relaxed_fill_count"], 1)
        self.assertEqual(selected[1]["selection_mode"], "relaxed")
        self.assertIn("relaxed_fill_after_strict_shortfall", selected[1]["selection_reason"])

    def test_summarize_surrogate_shadow_penalizes_large_heuristic_gap(self):
        result = {
            "verdict": "PASS",
            "fitness": 1.64,
            "sharpe": 1.80,
            "surrogate_shadow": {
                "status": "ready",
                "preview_verdict": "FAIL",
                "alignment": "more_cautious",
                "predicted_fitness": 0.16,
                "predicted_sharpe": -0.15,
            },
        }

        summary = summarize_surrogate_shadow(result)

        self.assertGreater(summary["penalty_score"], 0.15)
        self.assertGreater(summary["confidence_drag"], 0.10)
        self.assertEqual(summary["hard_signal"], "severe_mismatch")
        self.assertIn("surrogate_preview_fail", summary["reasons"])

    def test_selection_rejects_severe_surrogate_shadow_mismatch(self):
        candidates = [
            {
                "expression": "rank(ts_mean(returns, 21))",
                "source_key": "paper-shadow-risk",
                "thesis_id": "technical_indicator",
                "thesis_family_ids": ["technical_indicator"],
                "risk_tags": ["surrogate_shadow_risk"],
                "brain_feedback_penalty": 0.02,
                "submitted_diversity_score": 0.92,
                "submitted_overlap_score": 0.08,
                "submitted_overlap_reasons": [],
                "surrogate_shadow_penalty": 0.18,
                "surrogate_shadow_confidence_drag": 0.14,
                "surrogate_shadow_reasons": ["surrogate_preview_fail", "surrogate_more_cautious"],
                "surrogate_shadow_preview_verdict": "FAIL",
                "surrogate_shadow_alignment": "more_cautious",
                "surrogate_shadow_hard_signal": "severe_mismatch",
                "robustness_score": 0.84,
                "confidence_score": 0.79,
                "candidate_score": 0.83,
                "source_quality_score": 0.76,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 84.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.74,
                    "style_tags": ["trend", "technical", "rank"],
                    "surrogate_shadow": {
                        "status": "ready",
                        "preview_verdict": "FAIL",
                        "alignment": "more_cautious",
                        "predicted_fitness": 0.18,
                        "predicted_sharpe": 0.12,
                        "training_rows": 128,
                    },
                },
            }
        ]

        payload, selected, _, rejected = build_payload(
            ideas=[],
            candidates=candidates,
            memory={"style_leaders": [], "failure_counts": {}},
            count=1,
            seed_store={},
            include_watchlist=False,
        )

        self.assertEqual(len(selected), 0)
        self.assertEqual(payload["batch"]["qualified_count"], 0)
        self.assertTrue(any("surrogate_shadow_risk" in item["reason"] for item in rejected))

    def test_selection_watchlists_soft_surrogate_shadow_mismatch(self):
        candidates = [
            {
                "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                "source_key": "paper-shadow-soft",
                "thesis": "Residual beta",
                "thesis_id": "residual_beta",
                "thesis_family_ids": ["residual_beta"],
                "risk_tags": [],
                "brain_feedback_penalty": 0.06,
                "submitted_diversity_score": 0.94,
                "submitted_overlap_score": 0.08,
                "submitted_overlap_reasons": [],
                "surrogate_shadow_penalty": 0.11,
                "surrogate_shadow_confidence_drag": 0.08,
                "surrogate_shadow_reasons": ["surrogate_preview_fail", "surrogate_more_cautious"],
                "surrogate_shadow_preview_verdict": "FAIL",
                "surrogate_shadow_alignment": "more_cautious",
                "surrogate_shadow_hard_signal": "soft_mismatch",
                "robustness_score": 0.82,
                "confidence_score": 0.49,
                "candidate_score": 0.73,
                "source_quality_score": 0.74,
                "source_specificity_score": 0.78,
                "style_alignment_score": 0.52,
                "source_ideas": ["paper-shadow-soft"],
                "why": "why",
                "thinking": "thinking",
                "settings": "USA, TOP3000, Decay 7, Delay 1, Truncation 0.06, Neutralization Industry",
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 74.0,
                    "confidence": "MEDIUM",
                    "sharpe": 1.52,
                    "fitness": 1.18,
                    "uniqueness_proxy": 0.77,
                    "style_tags": ["residual", "beta", "correlation"],
                    "surrogate_shadow": {
                        "status": "ready",
                        "preview_verdict": "FAIL",
                        "alignment": "more_cautious",
                        "predicted_fitness": 0.58,
                        "predicted_sharpe": 0.74,
                        "training_rows": 128,
                    },
                },
            }
        ]

        payload, selected, watchlist, rejected = build_payload(
            ideas=[],
            candidates=candidates,
            memory={"style_leaders": [], "failure_counts": {}},
            count=1,
            seed_store={},
            include_watchlist=True,
        )

        self.assertEqual(len(selected), 0)
        self.assertEqual(len(watchlist), 1)
        self.assertEqual(len(rejected), 0)
        self.assertEqual(watchlist[0]["selection_mode"], "watchlist")
        self.assertIn("confidence_score<", watchlist[0]["selection_reason"])
        self.assertEqual(payload["batch"]["watchlist_count"], 1)

    def test_brain_feedback_penalizes_exact_weak_skeleton(self):
        expression = (
            "rank(add(rank(multiply((1-close/ts_delay(close,5)),inverse(ts_std_dev(returns,21)))),"
            "rank(multiply(ts_zscore(close,21),divide(volume,ts_mean(volume,63))))))"
        )
        rows = [
            {
                "alpha_id": "weak_1",
                "regular_code": expression,
                "turnover": "0.4615",
                "returns": "0.0284",
                "fitness": "0.15",
                "sharpe": "0.61",
                "LOW_SHARPE": "FAIL",
                "LOW_FITNESS": "FAIL",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "FAIL",
                "SELF_CORRELATION": "PENDING",
                "MATCHES_COMPETITION": "PASS",
            },
            {
                "alpha_id": "weak_2",
                "regular_code": "rank(multiply((1-close/ts_delay(close,5)),inverse(ts_std_dev(returns,21))))",
                "turnover": "0.6043",
                "returns": "0.0810",
                "fitness": "0.89",
                "sharpe": "1.67",
                "LOW_SHARPE": "PASS",
                "LOW_FITNESS": "FAIL",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "SELF_CORRELATION": "PENDING",
                "MATCHES_COMPETITION": "PASS",
            },
        ]

        context = build_brain_feedback_context(rows)
        penalty = compute_brain_feedback_penalty(
            expression,
            {
                "alpha_type": "hybrid",
                "style_tags": ["rank", "liquidity", "normalization", "reversal", "volume", "volatility"],
            },
            ["reversal_conditioned", "shock_response"],
            context,
        )

        self.assertTrue(penalty["exact_skeleton_match"])
        self.assertGreater(penalty["penalty_score"], 0.20)
        self.assertIn("brain_exact_skeleton", " ".join(penalty["reasons"]))

    def test_brain_feedback_penalty_prefers_matching_context(self):
        rows = [
            {
                "alpha_id": "ctx_1",
                "regular_code": "ts_zscore(-ts_zscore(close, 21),63)",
                "turnover": "0.42",
                "returns": "-0.03",
                "fitness": "0.12",
                "sharpe": "0.41",
                "LOW_SHARPE": "FAIL",
                "LOW_FITNESS": "FAIL",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "FAIL",
                "SELF_CORRELATION": "PASS",
                "MATCHES_COMPETITION": "PASS",
                "region": "USA",
                "universe": "TOP200",
                "delay": "1",
                "decay": "3",
                "neutralization": "Subindustry",
                "truncation": "0.02",
            },
            {
                "alpha_id": "ctx_2",
                "regular_code": "ts_sum(close,21)",
                "turnover": "0.39",
                "returns": "-0.02",
                "fitness": "0.20",
                "sharpe": "0.58",
                "LOW_SHARPE": "FAIL",
                "LOW_FITNESS": "FAIL",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "FAIL",
                "SELF_CORRELATION": "PASS",
                "MATCHES_COMPETITION": "PASS",
                "region": "USA",
                "universe": "TOP200",
                "delay": "1",
                "decay": "3",
                "neutralization": "Subindustry",
                "truncation": "0.02",
            },
            {
                "alpha_id": "ctx_3",
                "regular_code": "ts_sum(close,10)",
                "turnover": "0.33",
                "returns": "-0.01",
                "fitness": "0.24",
                "sharpe": "0.61",
                "LOW_SHARPE": "FAIL",
                "LOW_FITNESS": "FAIL",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "FAIL",
                "SELF_CORRELATION": "PASS",
                "MATCHES_COMPETITION": "PASS",
                "region": "USA",
                "universe": "TOP200",
                "delay": "1",
                "decay": "3",
                "neutralization": "Subindustry",
                "truncation": "0.02",
            },
        ]

        context = build_brain_feedback_context(rows)
        matching = compute_brain_feedback_penalty(
            "rank(ts_mean(close,10))",
            {
                "alpha_type": "momentum",
                "style_tags": ["trend", "technical"],
                "settings": {
                    "region": "USA",
                    "universe": "TOP200",
                    "delay": 1,
                    "decay": 3,
                    "neutralization": "Subindustry",
                    "truncation": 0.02,
                },
            },
            ["technical_indicator"],
            context,
        )
        mismatch = compute_brain_feedback_penalty(
            "rank(ts_mean(close,10))",
            {
                "alpha_type": "momentum",
                "style_tags": ["trend", "technical"],
                "settings": {
                    "region": "USA",
                    "universe": "TOP3000",
                    "delay": 1,
                    "decay": 6,
                    "neutralization": "Industry",
                    "truncation": 0.05,
                },
            },
            ["technical_indicator"],
            context,
        )

        self.assertGreater(matching["penalty_score"], mismatch["penalty_score"])
        self.assertEqual(matching["context_mode"], "exact_context")
        self.assertEqual(mismatch["context_mode"], "context_mismatch")

    def test_selection_rejects_out_of_sample_risk(self):
        candidates = [
            {
                "expression": "rank(add(rank(multiply((1-close/ts_delay(close,5)),inverse(ts_std_dev(returns,21)))),rank(multiply(ts_zscore(close,21),divide(volume,ts_mean(volume,63))))))",
                "source_key": "paper-risky",
                "thesis_id": "blend__reversal_conditioned__shock_response",
                "thesis_family_ids": ["reversal_conditioned", "shock_response"],
                "risk_tags": ["out_of_sample_risk"],
                "robustness_score": 0.82,
                "submitted_diversity_score": 0.91,
                "submitted_overlap_score": 0.09,
                "submitted_overlap_reasons": [],
                "settings_robustness": {
                    "settings_count": 3,
                    "pass_rate": 1.0,
                    "avg_alpha_score": 80.0,
                    "min_alpha_score": 78.0,
                    "max_alpha_score": 83.0,
                    "alpha_spread": 5.0,
                    "robustness_score": 0.82,
                },
                "confidence_score": 0.78,
                "candidate_score": 0.81,
                "source_quality_score": 0.73,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 82.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.72,
                },
            }
        ]

        payload, selected, _, rejected = build_payload(
            ideas=[],
            candidates=candidates,
            memory={"style_leaders": [], "failure_counts": {}},
            count=1,
            seed_store={},
            include_watchlist=False,
        )

        self.assertEqual(len(selected), 0)
        self.assertEqual(payload["batch"]["qualified_count"], 0)
        self.assertTrue(any("out_of_sample_risk" in item["reason"] for item in rejected))

    def test_selection_rejects_blocked_scout_skeleton(self):
        candidates = [
            {
                "expression": "rank(ts_mean(close, 21))",
                "source_key": "paper-blocked",
                "thesis_id": "technical_indicator",
                "thesis_family_ids": ["technical_indicator"],
                "risk_tags": ["scout_blocked_skeleton_risk"],
                "scout_blocked_skeleton": True,
                "family_diversity_score": 0.24,
                "robustness_score": 0.83,
                "submitted_diversity_score": 0.92,
                "confidence_score": 0.82,
                "candidate_score": 0.84,
                "source_quality_score": 0.76,
                "local_metrics": {
                    "verdict": "PASS",
                    "alpha_score": 84.0,
                    "confidence": "HIGH",
                    "uniqueness_proxy": 0.74,
                    "style_tags": ["trend", "technical", "rank"],
                },
            }
        ]

        payload, selected, _, rejected = build_payload(
            ideas=[],
            candidates=candidates,
            memory={"style_leaders": [], "failure_counts": {}},
            count=1,
            seed_store={},
            include_watchlist=False,
        )

        self.assertEqual(len(selected), 0)
        self.assertEqual(payload["batch"]["qualified_count"], 0)
        self.assertTrue(any("scout_blocked_skeleton_risk" in item["reason"] for item in rejected))

    def test_quality_first_selection_can_return_fewer_than_requested(self):
        ideas = filter_relevant_ideas(
            [
                {
                    "source": "openalex",
                    "query": "price volume anomaly equities",
                    "title": "Price-Volume Momentum and Volatility Shock in Equities",
                    "summary": "A momentum signal conditioned on liquidity, volume, and volatility shock.",
                    "url": "https://example.com/paper",
                    "year": 2024,
                    "citations": 12,
                    "relevance": 900.0,
                    "bias_families": ["pv_divergence", "shock_response"],
                    "bias_style_tags": ["volume", "volatility"],
                    "bias_horizon": "medium",
                    "source_score": 0.82,
                }
            ],
            scout_memory={},
        )
        memory = {"style_leaders": [], "failure_counts": {}}
        candidates = build_candidates(ideas, memory=memory, scout_memory={}, history_index=HistoryIndex())

        payload, selected, watchlist, rejected = build_payload(
            ideas=ideas,
            candidates=candidates,
            memory=memory,
            count=6,
            seed_store={},
            min_alpha_score=90.0,
            min_confidence_score=0.95,
            include_watchlist=False,
        )

        self.assertEqual(len(selected), 0)
        self.assertGreaterEqual(len(rejected), 1)
        self.assertEqual(payload["batch"]["qualified_count"], 0)

    def test_selection_enforces_source_paper_diversity(self):
        ideas = filter_relevant_ideas(
            [
                {
                    "source": "openalex",
                    "query": "residual beta equity anomaly",
                    "title": "Residual Beta and Equity Anomalies",
                    "summary": "Residual beta and market-neutral stock selection with equity anomalies.",
                    "url": "https://example.com/beta-paper",
                    "year": 2023,
                    "citations": 8,
                    "relevance": 850.0,
                    "bias_families": ["residual_beta", "technical_indicator"],
                    "bias_style_tags": ["residual", "beta", "momentum"],
                    "bias_horizon": "long",
                    "source_score": 0.78,
                }
            ],
            scout_memory={},
        )
        memory = {"style_leaders": [], "failure_counts": {}}
        candidates = build_candidates(ideas, memory=memory, scout_memory={}, history_index=HistoryIndex())

        payload, selected, watchlist, rejected = build_payload(
            ideas=ideas,
            candidates=candidates,
            memory=memory,
            count=6,
            seed_store={},
            submitted_context={
                "count": 2,
                "entries": [],
                "skeletons": [],
                "alpha_type_counts": {"hybrid": 2},
                "style_tag_counts": {"rank": 2},
                "saturated_alpha_types": ["hybrid"],
                "dominant_style_tags": ["rank"],
            },
            include_watchlist=True,
        )

        self.assertLessEqual(len(selected), 1)
        self.assertTrue(
            any(
                item["reason"] == "source_paper_cap"
                or "source_paper_cap" in item["reason"]
                or "surrogate_shadow_risk" in item["reason"]
                or "category_overload_risk" in item["reason"]
                or "out_of_sample_risk" in item["reason"]
                for item in rejected
            )
        )
        markdown = render_markdown(payload, selected, rejected, top=6, watchlist=watchlist)
        self.assertIn("Ready To Submit", markdown)
        self.assertIn("Submitted Library Pressure", markdown)
        self.assertTrue("Settings robustness:" in markdown or "Held Out" in markdown)
        self.assertTrue("```text" in markdown or "Held Out" in markdown)
        self.assertTrue("- Universe:" in markdown or "Held Out" in markdown)

    def test_render_markdown_surfaces_brain_feedback_status_and_relaxed_bucket(self):
        payload, selected, _, _ = build_payload(
            ideas=[],
            candidates=[
                {
                    "expression": "rank(ts_zscore(close, 21))",
                    "source_key": "paper-a",
                    "thesis": "Technical indicator",
                    "thesis_id": "technical_indicator",
                    "thesis_family_ids": ["technical_indicator"],
                    "risk_tags": [],
                    "brain_feedback_penalty": 0.04,
                    "submitted_diversity_score": 0.9,
                    "submitted_overlap_score": 0.1,
                    "submitted_overlap_reasons": [],
                    "brain_feedback_reasons": [],
                    "robustness_score": 0.85,
                    "novelty_score": 0.8,
                    "confidence_score": 0.84,
                    "candidate_score": 0.85,
                    "source_quality_score": 0.75,
                    "source_specificity_score": 0.8,
                    "style_alignment_score": 0.5,
                    "source_ideas": ["paper-a"],
                    "why": "why",
                    "thinking": "thinking",
                    "settings": "USA, TOP3000, Decay 6, Delay 1, Truncation 0.05, Neutralization Industry",
                    "settings_robustness": {
                        "settings_count": 3,
                        "pass_rate": 1.0,
                        "min_alpha_score": 82.0,
                        "max_alpha_score": 84.0,
                        "robustness_score": 0.85,
                    },
                    "local_metrics": {
                        "verdict": "PASS",
                        "alpha_score": 84.0,
                        "confidence": "HIGH",
                        "sharpe": 2.0,
                        "fitness": 1.8,
                        "uniqueness_proxy": 0.75,
                        "style_tags": ["trend", "technical", "rank"],
                    },
                },
                {
                    "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                    "source_key": "paper-b",
                    "thesis": "Residual beta",
                    "thesis_id": "residual_beta",
                    "thesis_family_ids": ["residual_beta"],
                    "risk_tags": [],
                    "brain_feedback_penalty": 0.24,
                    "submitted_diversity_score": 0.95,
                    "submitted_overlap_score": 0.05,
                    "submitted_overlap_reasons": [],
                    "brain_feedback_reasons": [],
                    "robustness_score": 0.81,
                    "novelty_score": 0.82,
                    "confidence_score": 0.55,
                    "candidate_score": 0.79,
                    "source_quality_score": 0.73,
                    "source_specificity_score": 0.78,
                    "style_alignment_score": 0.52,
                    "source_ideas": ["paper-b"],
                    "why": "why",
                    "thinking": "thinking",
                    "settings": "USA, TOP3000, Decay 7, Delay 1, Truncation 0.06, Neutralization Industry",
                    "settings_robustness": {
                        "settings_count": 3,
                        "pass_rate": 1.0,
                        "min_alpha_score": 78.0,
                        "max_alpha_score": 80.0,
                        "robustness_score": 0.81,
                    },
                    "local_metrics": {
                        "verdict": "PASS",
                        "alpha_score": 79.0,
                        "confidence": "MEDIUM",
                        "sharpe": 1.7,
                        "fitness": 1.2,
                        "uniqueness_proxy": 0.77,
                        "style_tags": ["residual", "beta", "correlation"],
                    },
                },
            ],
            memory={"style_leaders": [], "failure_counts": {}},
            count=2,
            seed_store={},
            include_watchlist=False,
            brain_feedback_status={"status": "schema_error", "path": "simulation_results.csv", "message": "Missing required columns."},
        )

        markdown = render_markdown(payload, selected, [], top=2, watchlist=[])
        self.assertIn("Brain feedback status: schema_error", markdown)
        self.assertIn("Ready To Submit", markdown)
        self.assertIn("Reportable picks: 1", markdown)
        self.assertNotIn("Explore / Verify First", markdown)
        self.assertNotIn("relaxed exploration", markdown)

    def test_render_markdown_holds_back_strict_pick_below_report_gate(self):
        payload, selected, _, _ = build_payload(
            ideas=[],
            candidates=[
                {
                    "expression": "rank(ts_zscore(close, 21))",
                    "source_key": "paper-a",
                    "thesis": "Technical indicator",
                    "thesis_id": "technical_indicator",
                    "thesis_family_ids": ["technical_indicator"],
                    "risk_tags": [],
                    "brain_feedback_penalty": 0.04,
                    "submitted_diversity_score": 0.9,
                    "submitted_overlap_score": 0.1,
                    "submitted_overlap_reasons": [],
                    "brain_feedback_reasons": [],
                    "robustness_score": 0.61,
                    "novelty_score": 0.8,
                    "confidence_score": 0.61,
                    "candidate_score": 0.82,
                    "source_quality_score": 0.75,
                    "source_specificity_score": 0.8,
                    "style_alignment_score": 0.5,
                    "source_ideas": ["paper-a"],
                    "why": "why",
                    "thinking": "thinking",
                    "settings": "USA, TOP3000, Decay 6, Delay 1, Truncation 0.05, Neutralization Industry",
                    "settings_robustness": {
                        "settings_count": 3,
                        "pass_rate": 1.0,
                        "min_alpha_score": 73.0,
                        "max_alpha_score": 75.0,
                        "robustness_score": 0.61,
                    },
                    "local_metrics": {
                        "verdict": "PASS",
                        "alpha_score": 75.0,
                        "confidence": "MEDIUM",
                        "sharpe": 1.6,
                        "fitness": 1.2,
                        "uniqueness_proxy": 0.75,
                        "style_tags": ["trend", "technical", "rank"],
                    },
                }
            ],
            memory={"style_leaders": [], "failure_counts": {}},
            count=1,
            seed_store={},
            include_watchlist=False,
        )

        markdown = render_markdown(payload, selected, [], top=1, watchlist=[])
        self.assertIn("No alpha cleared the report gate this round.", markdown)
        self.assertIn("Strict picks held back from reporting: 1", markdown)

    def test_render_markdown_shows_report_overflow_candidates(self):
        payload = {
            "batch": {
                "source_inputs": ["openalex"],
                "candidates": [],
                "strict_count": 1,
                "reportable_count": 2,
                "relaxed_fill_count": 0,
                "fallback_mode": False,
            },
            "settings": {
                "search_breadth": "focused",
                "query_profile_count": 4,
                "diversity_weight": 0.35,
                "report_min_alpha_score": 76.0,
                "report_min_confidence_score": 0.64,
                "report_min_robustness_score": 0.62,
                "archive_frequency": "hour",
            },
            "ideas": [],
            "brain_feedback_status": {"status": "ok"},
            "report_status": {"published": True, "reason": "reportable_pick_found", "interval_minutes": 60},
            "archive_status": {"written": True, "reason": "reportable_pick_found", "frequency": "hour"},
            "archived_paths": {"root": "artifacts/bao_cao_ngay/2026-03-26/08h"},
            "reportable_selected": [
                {
                    "expression": "rank(ts_zscore(close, 21))",
                    "thesis": "Technical indicator A",
                    "candidate_score": 0.87,
                    "confidence_score": 0.86,
                    "robustness_score": 0.86,
                    "selection_reason": "passed_quality_gate",
                    "local_metrics": {
                        "verdict": "PASS",
                        "alpha_score": 84.0,
                        "confidence": "HIGH",
                        "uniqueness_proxy": 0.76,
                        "style_tags": ["trend", "technical", "rank"],
                        "surrogate_shadow": {"status": "unavailable"},
                    },
                },
                {
                    "expression": "rank(ts_zscore(close, 20))",
                    "thesis": "Technical indicator B",
                    "candidate_score": 0.865,
                    "confidence_score": 0.855,
                    "robustness_score": 0.855,
                    "selection_reason": "top_scored_overflow",
                    "local_metrics": {
                        "verdict": "PASS",
                        "alpha_score": 83.5,
                        "confidence": "HIGH",
                        "uniqueness_proxy": 0.75,
                        "style_tags": ["trend", "technical", "rank"],
                        "surrogate_shadow": {"status": "unavailable"},
                    },
                },
            ],
        }

        markdown = render_markdown(payload, selected=[], rejected=[], top=1, watchlist=[])
        self.assertIn("Report overflow picks: 1", markdown)
        self.assertIn("Technical indicator A", markdown)
        self.assertIn("Technical indicator B", markdown)

    def test_render_markdown_surfaces_learned_seed_bucket(self):
        payload = {
            "batch": {
                "source_inputs": ["zip_knowledge"],
                "candidates": [
                    {
                        "expression": "rank(ts_mean(close, 20))",
                        "source_key": "title:zip seed",
                        "source_kind": "zip_knowledge",
                        "learned_seed": True,
                        "thesis": "Learned Seed: Technical indicator blend",
                        "thesis_id": "seed__technical_indicator",
                        "selection_status": "rejected",
                        "selection_mode": "rejected",
                        "selection_reason": "confidence_score<0.58",
                        "selection_rank_score": 0.71,
                        "candidate_score": 0.69,
                        "confidence_score": 0.48,
                        "novelty_score": 0.77,
                        "robustness_score": 0.84,
                        "submitted_overlap_score": 0.12,
                        "submitted_diversity_score": 0.88,
                        "brain_feedback_penalty": 0.08,
                        "brain_feedback_reasons": [],
                        "brain_feedback_context_mode": "global_legacy",
                        "brain_feedback_candidate_context": "region=USA",
                        "submitted_overlap_reasons": [],
                        "surrogate_shadow_penalty": 0.0,
                        "surrogate_shadow_reasons": [],
                        "source_quality_score": 0.74,
                        "source_specificity_score": 0.7,
                        "style_alignment_score": 0.45,
                        "source_ideas": ["ZIP Seed"],
                        "why": "why",
                        "thinking": "thinking",
                        "risk_tags": [],
                        "settings": "USA, TOP3000, Decay 6, Delay 1, Truncation 0.05, Neutralization Industry",
                        "settings_robustness": {
                            "settings_count": 3,
                            "pass_rate": 1.0,
                            "min_alpha_score": 79.0,
                            "max_alpha_score": 81.0,
                            "robustness_score": 0.84,
                        },
                        "local_metrics": {
                            "verdict": "PASS",
                            "alpha_score": 80.0,
                            "confidence": "MEDIUM",
                            "sharpe": 1.8,
                            "fitness": 1.3,
                            "uniqueness_proxy": 0.71,
                            "style_tags": ["trend", "technical"],
                            "surrogate_shadow": {"status": "unavailable"},
                        },
                    }
                ],
                "fallback_mode": False,
                "relaxed_fill_count": 0,
            },
            "settings": {"search_breadth": "focused", "query_profile_count": 10, "diversity_weight": 0.35},
            "ideas": [],
            "brain_feedback_status": {"status": "ok"},
            "self_learning": {"zip": {"status": "ok", "seed_expression_count": 3, "operator_count": 10}},
            "run_timestamp": "2026-03-26 08h 45'",
            "archived_paths": {"root": "artifacts/bao_cao_ngay/2026-03-26/08h"},
            "archive_status": {"written": True, "reason": "reportable_pick_found", "frequency": "hour"},
        }

        markdown = render_markdown(payload, [], [], top=2, watchlist=[])
        self.assertIn("Learned Seeds / Verify First", markdown)
        self.assertIn("Selection mode: rejected", markdown)
        self.assertIn("Learned Seed: Technical indicator blend", markdown)
        self.assertIn("Run time: 2026-03-26 08h 45'", markdown)
        self.assertIn("Archived run folder: artifacts/bao_cao_ngay/2026-03-26/08h", markdown)

    def test_fallback_mode_builds_local_library_ideas(self):
        ideas = build_fallback_ideas()
        self.assertGreater(len(ideas), 0)
        self.assertTrue(all(item["source"] == "local_library" for item in ideas))

    def test_build_book_ideas_returns_local_book_sources(self):
        ideas = build_book_ideas()
        self.assertGreater(len(ideas), 0)
        self.assertTrue(all(item["source"] == "finding_alphas_book" for item in ideas))

    def test_archive_memory_tracks_weak_sources(self):
        records = [
            {
                "source_key": "weak paper",
                "skeleton": "alpha1",
                "thesis_id": "technical_indicator",
                "family_components": ["technical_indicator"],
                "horizon": "long",
                "selection_status": "rejected",
                "alpha_score": 42.0,
                "confidence_score": 0.41,
                "source_quality_score": 0.35,
                "verdict": "FAIL",
            },
            {
                "source_key": "weak paper",
                "skeleton": "alpha1",
                "thesis_id": "technical_indicator",
                "family_components": ["technical_indicator"],
                "horizon": "long",
                "selection_status": "rejected",
                "alpha_score": 44.0,
                "confidence_score": 0.43,
                "source_quality_score": 0.34,
                "verdict": "FAIL",
            },
        ]

        memory = aggregate_scout_memory(records)
        self.assertIn("weak paper", memory["blocked_source_keys"])
        self.assertIn("alpha1", memory["blocked_skeletons"])

    def test_history_records_can_be_loaded_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "scout_history.jsonl"
            history_path.write_text('{"source_key":"paper1","selection_status":"selected"}\n', encoding="utf-8")

            records = load_history_records(history_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["source_key"], "paper1")


if __name__ == "__main__":
    unittest.main()
