import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.approve_seeds import (
    DEFAULT_JSON_OUTPUT,
    DEFAULT_MARKDOWN_OUTPUT,
    load_seed_store,
    merge_into_seed_store,
    resolve_output_path,
    select_candidates,
)
from src.seed_store import (
    SeedStoreCorruptError,
    backup_seed_store_path,
    dated_backup_seed_store_path,
    load_seed_payload,
    write_seed_store,
)
from src.program_tokens import validate_token_program


class TestApproveSeeds(unittest.TestCase):
    def test_select_candidates_rejects_invalid_and_limits_top(self):
        payload = {
            "batch": {
                "candidates": [
                    {
                        "expression": "rank(ts_zscore(abs(close-vwap),21))",
                        "seed_ready": True,
                        "qualified": True,
                        "quality_label": "qualified",
                        "confidence_score": 0.72,
                        "candidate_score": 1.2,
                        "novelty_score": 0.8,
                        "style_alignment_score": 0.8,
                        "thesis_id": "vwap_dislocation",
                        "local_metrics": {
                            "verdict": "LIKELY_PASS",
                            "alpha_score": 71.0,
                            "sharpe": 1.41,
                            "fitness": 1.18,
                        },
                        "token_program": ["FACTOR_3", "TSZ_21", "RANK"],
                    },
                    {
                        "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                        "seed_ready": True,
                        "qualified": True,
                        "quality_label": "qualified",
                        "confidence_score": 0.66,
                        "candidate_score": 1.1,
                        "novelty_score": 0.7,
                        "style_alignment_score": 0.7,
                        "thesis_id": "residual_beta",
                        "local_metrics": {
                            "verdict": "PASS",
                            "alpha_score": 74.0,
                            "sharpe": 1.88,
                            "fitness": 1.94,
                        },
                        "token_program": ["CLOSE", "BETA", "REG_RESD_63", "RANK"],
                    },
                    {
                        "expression": "invalid(candidate)",
                        "seed_ready": True,
                        "qualified": True,
                        "quality_label": "qualified",
                        "confidence_score": 0.7,
                        "candidate_score": 1.0,
                        "novelty_score": 0.9,
                        "style_alignment_score": 0.1,
                        "thesis_id": "unknown",
                        "local_metrics": {
                            "verdict": "PASS",
                            "alpha_score": 68.0,
                            "sharpe": 1.5,
                            "fitness": 1.2,
                        },
                        "token_program": ["DOES_NOT_EXIST"],
                    },
                    {
                        "expression": "rank(close)",
                        "seed_ready": False,
                        "qualified": True,
                        "quality_label": "qualified",
                        "confidence_score": 0.7,
                        "candidate_score": 2.0,
                        "novelty_score": 0.5,
                        "style_alignment_score": 0.2,
                        "thesis_id": "baseline",
                        "local_metrics": {
                            "verdict": "PASS",
                            "alpha_score": 80.0,
                            "sharpe": 2.0,
                            "fitness": 1.8,
                        },
                        "token_program": ["CLOSE", "RANK"],
                    },
                ]
            }
        }

        approved, rejected = select_candidates(payload, top=2)
        self.assertEqual(len(approved), 2)
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all("compiled_expression" in item for item in approved))
        self.assertTrue(all("selection_score" in item for item in approved))

    def test_select_candidates_preserves_thesis_diversity(self):
        payload = {
            "batch": {
                "candidates": [
                    {
                        "expression": "rank(ts_zscore(abs(close-vwap),21))",
                        "seed_ready": True,
                        "qualified": True,
                        "quality_label": "qualified",
                        "confidence_score": 0.71,
                        "candidate_score": 1.5,
                        "novelty_score": 0.7,
                        "style_alignment_score": 0.8,
                        "thesis_id": "vwap_dislocation",
                        "local_metrics": {
                            "verdict": "LIKELY_PASS",
                            "alpha_score": 71.0,
                            "sharpe": 1.42,
                            "fitness": 1.2,
                        },
                        "token_program": ["FACTOR_3", "TSZ_21", "RANK"],
                        "risk_tags": [],
                    },
                    {
                        "expression": "zscore(rank(divide(subtract(close,vwap),ts_std_dev(returns,21))))",
                        "seed_ready": True,
                        "qualified": True,
                        "quality_label": "qualified",
                        "confidence_score": 0.69,
                        "candidate_score": 1.4,
                        "novelty_score": 0.75,
                        "style_alignment_score": 0.78,
                        "thesis_id": "vwap_dislocation",
                        "local_metrics": {
                            "verdict": "PASS",
                            "alpha_score": 73.5,
                            "sharpe": 1.8,
                            "fitness": 1.7,
                        },
                        "token_program": ["CLOSE", "VWAP", "SUB", "RET", "STD_21", "DIV", "RANK", "ZSCORE"],
                        "risk_tags": [],
                    },
                    {
                        "expression": "rank(ts_regression(close,beta_last_60_days_spy,63,lag=0,rettype=0))",
                        "seed_ready": True,
                        "qualified": True,
                        "quality_label": "qualified",
                        "confidence_score": 0.66,
                        "candidate_score": 1.1,
                        "novelty_score": 0.8,
                        "style_alignment_score": 0.7,
                        "thesis_id": "residual_beta",
                        "local_metrics": {
                            "verdict": "PASS",
                            "alpha_score": 74.0,
                            "sharpe": 1.88,
                            "fitness": 1.94,
                        },
                        "token_program": ["CLOSE", "BETA", "REG_RESD_63", "RANK"],
                        "risk_tags": [],
                    },
                ]
            }
        }

        approved, _ = select_candidates(payload, top=2)
        self.assertEqual(len(approved), 2)
        self.assertEqual({item["thesis_id"] for item in approved}, {"vwap_dislocation", "residual_beta"})

    def test_select_candidates_rejects_watchlist_even_if_seed_ready(self):
        payload = {
            "batch": {
                "candidates": [
                    {
                        "expression": "rank(ts_zscore(divide(ts_std_dev(returns,21),ts_mean(volume, 63)),21))",
                        "seed_ready": True,
                        "qualified": False,
                        "quality_label": "watchlist",
                        "confidence_score": 0.82,
                        "candidate_score": 1.6,
                        "novelty_score": 0.9,
                        "style_alignment_score": 0.8,
                        "thesis_id": "shock_response",
                        "local_metrics": {
                            "verdict": "PASS",
                            "alpha_score": 79.0,
                            "sharpe": 2.0,
                            "fitness": 2.2,
                        },
                        "token_program": ["RET", "STD_21", "ADV", "DIV", "TSZ_21", "RANK"],
                    }
                ]
            }
        }

        approved, rejected = select_candidates(payload, top=1)
        self.assertEqual(approved, [])
        self.assertEqual(len(rejected), 1)
        self.assertIn("submit_gate_failed", rejected[0]["reason"])
        self.assertIn("quality_label!=qualified", rejected[0]["reason"])

    def test_merge_into_seed_store_preserves_existing_entries(self):
        existing_program = validate_token_program(["CLOSE", "RANK"]).program
        seed_store = {
            "rank(close)": {
                "program": existing_program,
                "fitness": 1.0,
                "result": {"status": "PASS"},
            }
        }
        approved = [
            {
                "compiled_expression": "rank(ts_zscore(abs(close-vwap),21))",
                "token_program": ["FACTOR_3", "TSZ_21", "RANK"],
                "thesis_id": "vwap_dislocation",
                "novelty_score": 0.8,
                "risk_tags": [],
                "seed_source": "submit_ready_report",
            },
            {
                "compiled_expression": "rank(close)",
                "token_program": ["CLOSE", "RANK"],
                "thesis_id": "baseline",
                "novelty_score": 0.4,
                "risk_tags": [],
            },
        ]

        merged, inserted, skipped = merge_into_seed_store(seed_store, approved)
        self.assertIn("rank(close)", merged)
        self.assertEqual(merged["rank(close)"]["fitness"], 1.0)
        self.assertEqual(inserted, ["rank(ts_zscore(abs(close-vwap),21))"])
        self.assertEqual(skipped, ["rank(close)"])
        self.assertEqual(merged["rank(ts_zscore(abs(close-vwap),21))"]["result"]["source"], "submit_ready_report")
        self.assertTrue(merged["rank(ts_zscore(abs(close-vwap),21))"]["result"]["lineage"]["stage_results"]["seed"]["selected"])
        self.assertEqual(
            merged["rank(ts_zscore(abs(close-vwap),21))"]["result"]["lineage"]["stage_results"]["seed"]["seed_source"],
            "submit_ready_report",
        )

    def test_pickle_round_trip_works_with_new_entries(self):
        approved = [
            {
                "compiled_expression": "rank(ts_zscore(abs(close-vwap),21))",
                "token_program": ["FACTOR_3", "TSZ_21", "RANK"],
                "thesis_id": "vwap_dislocation",
                "novelty_score": 0.8,
                "risk_tags": [],
            }
        ]

        merged, _, _ = merge_into_seed_store({}, approved)
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "initial-population.pkl"
            with seed_path.open("wb") as handle:
                pickle.dump(merged, handle)
            with seed_path.open("rb") as handle:
                reloaded = pickle.load(handle)
            self.assertIn("rank(ts_zscore(abs(close-vwap),21))", reloaded)

    def test_resolve_output_path_uses_format_defaults(self):
        self.assertEqual(resolve_output_path(None, "markdown"), DEFAULT_MARKDOWN_OUTPUT)
        self.assertEqual(resolve_output_path(None, "json"), DEFAULT_JSON_OUTPUT)
        self.assertEqual(resolve_output_path("custom.md", "markdown"), Path("custom.md"))

    def test_load_seed_store_returns_empty_for_truncated_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "initial-population.pkl"
            seed_path.write_bytes(b"not-a-valid-pickle")

            loaded = load_seed_store(seed_path)

            self.assertEqual(loaded, {})

    def test_load_seed_store_raises_for_truncated_pickle_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "initial-population.pkl"
            seed_path.write_bytes(b"not-a-valid-pickle")

            with self.assertRaises(SeedStoreCorruptError):
                load_seed_store(seed_path, on_corrupt="raise")

    def test_write_seed_store_round_trip(self):
        payload = {
            "rank(close)": {
                "program": ["CLOSE", "RANK"],
                "fitness": 1.2,
                "result": {"status": "PASS"},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "initial-population.pkl"
            write_seed_store(seed_path, payload)

            with seed_path.open("rb") as handle:
                reloaded = pickle.load(handle)

            self.assertEqual(reloaded, payload)

    def test_write_seed_store_also_creates_backup_files(self):
        payload = {
            "rank(close)": {
                "program": ["CLOSE", "RANK"],
                "fitness": 1.2,
                "result": {"status": "PASS"},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "initial-population.pkl"
            write_seed_store(seed_path, payload)

            backup_path = backup_seed_store_path(seed_path)
            dated_backup_path = dated_backup_seed_store_path(seed_path)

            self.assertTrue(backup_path.exists())
            self.assertTrue(dated_backup_path.exists())
            with backup_path.open("rb") as handle:
                self.assertEqual(pickle.load(handle), payload)

    def test_load_seed_store_falls_back_to_backup_when_primary_is_corrupt(self):
        payload = {
            "rank(close)": {
                "program": ["CLOSE", "RANK"],
                "fitness": 1.2,
                "result": {"status": "PASS"},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "initial-population.pkl"
            write_seed_store(seed_path, payload)
            seed_path.write_bytes(b"not-a-valid-pickle")

            loaded = load_seed_store(seed_path, on_corrupt="raise")

            self.assertEqual(loaded, payload)

    def test_load_seed_payload_falls_back_to_dated_backup_when_primary_and_backup_are_corrupt(self):
        payload = {
            "rank(close)": {
                "program": ["CLOSE", "RANK"],
                "fitness": 1.2,
                "result": {"status": "PASS"},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "initial-population.pkl"
            write_seed_store(seed_path, payload)
            backup_path = backup_seed_store_path(seed_path)
            dated_backup_path = dated_backup_seed_store_path(seed_path)

            seed_path.write_bytes(b"not-a-valid-pickle")
            backup_path.write_bytes(b"not-a-valid-pickle")

            loaded, info = load_seed_payload(
                seed_path,
                on_corrupt="raise",
                allowed_types=dict,
                with_info=True,
            )

        self.assertEqual(loaded, payload)
        self.assertEqual(info["source_path"], str(dated_backup_path))
        self.assertTrue(info["fallback_used"])
        self.assertEqual(len(info["errors"]), 2)


if __name__ == "__main__":
    unittest.main()
