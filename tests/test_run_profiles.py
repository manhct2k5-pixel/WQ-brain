import json
import tempfile
import unittest
from pathlib import Path

from src.run_profiles import (
    RUN_PROFILES,
    detect_machine_resources,
    recommend_parallel_workers,
    resolve_profile,
    save_run_summary,
)


class DummySimulator:
    def __init__(self):
        self.best_fitness = 1.23
        self.best_program = "rank(close)"
        self.fitness_evaluations = 5

    def get_hall_of_fame(self):
        return [(1.23, "rank(close)", {"sharpe": 1.5, "fitness": 1.23})]

    def get_all_history(self):
        return [{"generation": 1, "best_fitness": 1.23}]


class TestRunProfiles(unittest.TestCase):
    def test_resolve_profile_override(self):
        profile = resolve_profile("light", population_size=12, generations=4)
        self.assertEqual(profile.population_size, 12)
        self.assertEqual(profile.generations, 4)
        self.assertEqual(profile.max_depth, RUN_PROFILES["light"].max_depth)
        self.assertEqual(profile.max_seed_programs, RUN_PROFILES["light"].max_seed_programs)

    def test_smart_profile_is_registered(self):
        profile = RUN_PROFILES["smart"]
        self.assertEqual(profile.n_parallel, 2)
        self.assertLess(profile.generations, RUN_PROFILES["careful"].generations)
        self.assertGreater(profile.max_seed_programs, 0)

    def test_save_run_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_run_summary(DummySimulator(), RUN_PROFILES["test"], output_dir=tmpdir)
            self.assertTrue(path.exists())
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["profile"]["name"], "test")
            self.assertEqual(payload["best_fitness"], 1.23)

    def test_detect_machine_resources_returns_cpu_count(self):
        resources = detect_machine_resources()
        self.assertGreaterEqual(int(resources["cpu_count"]), 1)

    def test_recommend_parallel_workers_respects_headroom_for_internal(self):
        workers = recommend_parallel_workers(
            scoring="internal",
            cpu_headroom=2,
            memory_headroom_gb=2.0,
            machine_resources={"cpu_count": 16, "memory_gb": 32.0},
        )
        self.assertEqual(workers, 14)

    def test_recommend_parallel_workers_caps_worldquant_parallelism(self):
        workers = recommend_parallel_workers(
            scoring="worldquant",
            cpu_headroom=1,
            memory_headroom_gb=1.0,
            machine_resources={"cpu_count": 16, "memory_gb": 32.0},
        )
        self.assertEqual(workers, 6)


if __name__ == "__main__":
    unittest.main()
