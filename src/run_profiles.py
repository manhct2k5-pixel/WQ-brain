"""Run profiles and artifact helpers for brain-learn."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from scripts.flow_utils import atomic_write_json


@dataclass(frozen=True)
class RunProfile:
    name: str
    description: str
    population_size: int
    generations: int
    tournament_size: int
    p_crossover: float
    p_mutation: float
    p_subtree_mutation: float
    parsimony_coefficient: float
    max_depth: int
    max_operators: int
    n_parallel: int
    max_seed_programs: int


RUN_PROFILES = {
    "test": RunProfile(
        name="test",
        description="Fast smoke-test profile with minimal quota usage.",
        population_size=8,
        generations=2,
        tournament_size=3,
        p_crossover=0.6,
        p_mutation=0.2,
        p_subtree_mutation=0.1,
        parsimony_coefficient=0.03,
        max_depth=4,
        max_operators=4,
        n_parallel=1,
        max_seed_programs=4,
    ),
    "careful": RunProfile(
        name="careful",
        description="Conservative profile with serialized evaluation and smaller daily submission pressure.",
        population_size=14,
        generations=4,
        tournament_size=4,
        p_crossover=0.55,
        p_mutation=0.2,
        p_subtree_mutation=0.08,
        parsimony_coefficient=0.03,
        max_depth=5,
        max_operators=5,
        n_parallel=1,
        max_seed_programs=8,
    ),
    "smart": RunProfile(
        name="smart",
        description="Balanced daily profile that keeps the run responsive while staying selective.",
        population_size=16,
        generations=3,
        tournament_size=4,
        p_crossover=0.6,
        p_mutation=0.2,
        p_subtree_mutation=0.08,
        parsimony_coefficient=0.03,
        max_depth=5,
        max_operators=5,
        n_parallel=2,
        max_seed_programs=8,
    ),
    "light": RunProfile(
        name="light",
        description="Safer research profile for everyday iteration.",
        population_size=20,
        generations=5,
        tournament_size=4,
        p_crossover=0.6,
        p_mutation=0.2,
        p_subtree_mutation=0.1,
        parsimony_coefficient=0.025,
        max_depth=5,
        max_operators=5,
        n_parallel=2,
        max_seed_programs=12,
    ),
    "full": RunProfile(
        name="full",
        description="Heavy search profile for long runs and larger quota budgets.",
        population_size=100,
        generations=50,
        tournament_size=5,
        p_crossover=0.6,
        p_mutation=0.15,
        p_subtree_mutation=0.1,
        parsimony_coefficient=0.02,
        max_depth=5,
        max_operators=6,
        n_parallel=3,
        max_seed_programs=20,
    ),
}


def resolve_profile(
    mode: str,
    *,
    population_size: int | None = None,
    generations: int | None = None,
    tournament_size: int | None = None,
    max_depth: int | None = None,
    max_operators: int | None = None,
    n_parallel: int | None = None,
) -> RunProfile:
    profile = RUN_PROFILES[mode]
    values = asdict(profile)
    values.update(
        {
            "population_size": population_size or profile.population_size,
            "generations": generations or profile.generations,
            "tournament_size": tournament_size or profile.tournament_size,
            "max_depth": max_depth or profile.max_depth,
            "max_operators": max_operators or profile.max_operators,
            "n_parallel": n_parallel or profile.n_parallel,
        }
    )
    return RunProfile(**values)


def detect_machine_resources() -> dict[str, float | int | None]:
    cpu_count = max(1, int(os.cpu_count() or 1))
    memory_gb = None
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    memory_kib = int(line.split()[1])
                    memory_gb = round(memory_kib / (1024 * 1024), 2)
                    break
        except (OSError, ValueError):
            memory_gb = None
    return {"cpu_count": cpu_count, "memory_gb": memory_gb}


def recommend_parallel_workers(
    *,
    scoring: str,
    cpu_headroom: int = 2,
    memory_headroom_gb: float = 2.0,
    parallel_cap: int = 0,
    machine_resources: dict[str, float | int | None] | None = None,
) -> int:
    resources = machine_resources or detect_machine_resources()
    cpu_count = max(1, int(resources.get("cpu_count") or 1))
    available_cpu_workers = max(1, cpu_count - max(0, int(cpu_headroom)))

    if scoring == "worldquant":
        backend_cap = 6
        memory_per_worker_gb = 0.5
    else:
        backend_cap = cpu_count
        memory_per_worker_gb = 1.5

    effective_cap = backend_cap
    if parallel_cap and parallel_cap > 0:
        effective_cap = min(effective_cap, int(parallel_cap))

    memory_gb = resources.get("memory_gb")
    memory_limited_workers = available_cpu_workers
    if isinstance(memory_gb, (int, float)) and memory_gb > 0:
        usable_memory_gb = max(1.0, float(memory_gb) - max(0.0, float(memory_headroom_gb)))
        memory_limited_workers = max(1, int(usable_memory_gb // memory_per_worker_gb))

    return max(1, min(available_cpu_workers, memory_limited_workers, effective_cap))


def ensure_artifacts_dir(root: Path | str = "artifacts") -> Path:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run_summary(
    simulator,
    profile: RunProfile,
    *,
    output_dir: Path | str = "artifacts",
) -> Path:
    artifacts_dir = ensure_artifacts_dir(output_dir)
    hall_of_fame = []
    for fitness, program_string, result_details in simulator.get_hall_of_fame():
        hall_of_fame.append(
            {
                "fitness": fitness,
                "program_string": program_string,
                "result_details": result_details,
            }
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "profile": asdict(profile),
        "best_fitness": simulator.best_fitness,
        "best_program": str(simulator.best_program) if simulator.best_program is not None else None,
        "fitness_evaluations": simulator.fitness_evaluations,
        "history": simulator.get_all_history(),
        "hall_of_fame": hall_of_fame,
    }
    output_path = artifacts_dir / "latest_run.json"
    atomic_write_json(output_path, payload)
    return output_path
