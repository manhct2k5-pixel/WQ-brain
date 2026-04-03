import pickle
import tempfile
import unittest
from pathlib import Path

from main import load_initial_population
from src.seed_store import write_seed_store


class DummyLogger:
    def log(self, _message):
        return None

    def warning(self, _message):
        return None


class TestSeedTokens(unittest.TestCase):
    def test_load_initial_population_materializes_token_names(self):
        payload = {
            "rank(close)": {
                "program": ["CLOSE", "RANK"],
                "fitness": None,
                "result": {"status": "PLANNED"},
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "initial-population.pkl"
            with seed_path.open("wb") as handle:
                pickle.dump(payload, handle)

            programs = load_initial_population(DummyLogger(), path=seed_path)
            self.assertEqual(len(programs), 1)
            self.assertEqual(str(programs[0][0]), "close")
            self.assertEqual(str(programs[0][1]), "rank")

    def test_load_initial_population_respects_profile_cap_and_priority(self):
        payload = {
            "rank(close)": {
                "program": ["CLOSE", "RANK"],
                "fitness": 0.2,
                "result": {"status": "FAIL"},
            },
            "rank(open)": {
                "program": ["OPEN", "RANK"],
                "fitness": 1.3,
                "result": {"status": "PASS"},
            },
            "rank(high)": {
                "program": ["HIGH", "RANK"],
                "fitness": 0.8,
                "result": {"status": "PLANNED", "novelty_score": 0.9},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "initial-population.pkl"
            with seed_path.open("wb") as handle:
                pickle.dump(payload, handle)

            programs = load_initial_population(DummyLogger(), path=seed_path, max_programs=2)
            self.assertEqual(len(programs), 2)
            self.assertEqual(str(programs[0][0]), "open")
            self.assertEqual(str(programs[1][0]), "high")

    def test_load_initial_population_prioritizes_completed_results_over_planned(self):
        payload = {
            "rank(high)": {
                "program": ["HIGH", "RANK"],
                "fitness": 0.7,
                "result": {"status": "PLANNED", "novelty_score": 0.9},
            },
            "rank(low)": {
                "program": ["LOW", "RANK"],
                "fitness": 0.5,
                "result": {"status": "COMPLETED", "novelty_score": 0.1},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "initial-population.pkl"
            with seed_path.open("wb") as handle:
                pickle.dump(payload, handle)

            programs = load_initial_population(DummyLogger(), path=seed_path, max_programs=1)
            self.assertEqual(len(programs), 1)
            self.assertEqual(str(programs[0][0]), "low")

    def test_load_initial_population_falls_back_to_backup_when_primary_is_corrupt(self):
        payload = {
            "rank(close)": {
                "program": ["CLOSE", "RANK"],
                "fitness": 0.4,
                "result": {"status": "PASS"},
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "initial-population.pkl"
            write_seed_store(seed_path, payload)
            seed_path.write_bytes(b"not-a-valid-pickle")

            programs = load_initial_population(DummyLogger(), path=seed_path, max_programs=1)

            self.assertEqual(len(programs), 1)
            self.assertEqual(str(programs[0][0]), "close")


if __name__ == "__main__":
    unittest.main()
