import os
import argparse
import sys
from dataclasses import replace
from pathlib import Path
from time import time_ns

from dotenv import load_dotenv

from src.function import *
from src.logger import Logger
from src.program_tokens import materialize_token_program
from src.submit_gate import local_metrics_pass_submit_gate
from src.seed_store import load_seed_payload
from src.utils import create_authenticated_session
from src.run_profiles import (
    detect_machine_resources,
    recommend_parallel_workers,
    resolve_profile,
    save_run_summary,
)

INITIAL_POPULATION_PATH = Path("initial-population.pkl")


def _coerce_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _seed_entry_priority(item):
    expression, entry = item
    if not isinstance(entry, dict):
        return (-10.0, expression)

    result = entry.get("result", {}) if isinstance(entry.get("result"), dict) else {}
    status = str(result.get("status", "")).upper()
    verdict = str(result.get("verdict", "")).upper()
    status_score = {
        "PASS": 5.0,
        "COMPLETE": 4.5,
        "COMPLETED": 4.5,
        "PLANNED": 3.0,
        "PENDING": 1.5,
        "FAIL": 0.5,
    }.get(status, 1.0)
    verdict_score = {
        "PASS": 4.5,
        "LIKELY_PASS": 3.8,
        "BORDERLINE": 2.0,
        "FAIL": 0.5,
    }.get(verdict, 0.0)
    readiness_bonus = 1.0 if local_metrics_pass_submit_gate(result) else 0.0

    fitness_score = _coerce_float(entry.get("fitness"))
    if fitness_score is None:
        fitness_score = _coerce_float(result.get("fitness"))
    fitness_score = fitness_score if fitness_score is not None else -1.0
    novelty_score = _coerce_float(result.get("novelty_score"))
    novelty_score = novelty_score if novelty_score is not None else 0.0
    return (max(status_score, verdict_score) + readiness_bonus, fitness_score, novelty_score, expression)


def _build_run_seed() -> int:
    return int((time_ns() ^ (os.getpid() << 16)) & 0xFFFFFFFF)

def create_session(logger):
    """Create and authenticate a new session."""
    # Load credentials from .env file
    load_dotenv()
    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    
    if not username or not password:
        logger.error("USERNAME or PASSWORD environment variables not set.")
        logger.error("Please check your .env file.")
        sys.exit(1)

    s, response = create_authenticated_session(
        username,
        password,
        logger=logger,
        context="Authentication",
    )

    if s is not None:
        logger.log("Authentication successful.")
        return s

    logger.error("Failed to authenticate.")
    if response is not None:
        logger.error(f"Status Code: {response.status_code}")
        logger.error(f"Response: {response.text}")
    return None


def load_initial_population(logger, path=INITIAL_POPULATION_PATH, max_programs=None):
    """Load optional seed programs for the next run."""
    if not path.exists():
        logger.log("No initial-population.pkl found. Starting from a fresh population.")
        return []

    payload, payload_info = load_seed_payload(
        path,
        allowed_types=(dict, list),
        with_info=True,
    )
    if payload_info.get("fallback_used"):
        logger.warning(
            f"Loaded seed data from backup {Path(payload_info['source_path']).name} because {path.name} could not be read safely."
        )
    elif payload_info.get("errors"):
        first_error = payload_info["errors"][0]["reason"]
        logger.warning(f"Could not load {path.name}: {first_error}. Starting fresh.")
        return []

    total_candidates = 0
    if isinstance(payload, dict):
        ranked_items = sorted(payload.items(), key=_seed_entry_priority, reverse=True)
        if max_programs:
            ranked_items = ranked_items[:max_programs]
        total_candidates = len(payload)
        programs = [
            entry.get("program")
            for _, entry in ranked_items
            if isinstance(entry, dict) and entry.get("program")
        ]
    elif isinstance(payload, list):
        total_candidates = len(payload)
        programs = []
        iterable = payload[:max_programs] if max_programs else payload
        for entry in iterable:
            if isinstance(entry, dict) and entry.get("program"):
                programs.append(entry["program"])
            elif entry:
                programs.append(entry)
    else:
        logger.warning(
            f"Unsupported {path.name} payload type: {type(payload).__name__}. Starting fresh."
        )
        return []

    normalized = []
    for program in programs:
        if isinstance(program, list) and program and all(isinstance(item, str) for item in program):
            try:
                normalized.append(materialize_token_program(program))
            except Exception as exc:
                logger.warning(f"Skipping serialized seed program that could not be materialized: {exc}")
            continue
        normalized.append(program)

    if max_programs and total_candidates > len(normalized):
        logger.log(
            f"Loaded {len(normalized)} seed programs from {path.name} "
            f"(filtered from {total_candidates} using the current profile cap)."
        )
    else:
        logger.log(f"Loaded {len(normalized)} seed programs from {path.name}.")
    return normalized

def parse_args():
    parser = argparse.ArgumentParser(description="Run the brain-learn alpha engine.")
    parser.add_argument(
        "--mode",
        choices=("test", "careful", "smart", "light", "full"),
        default="light",
        help="Run profile. Defaults to the safer light profile.",
    )
    parser.add_argument("--population-size", type=int, help="Override profile population size.")
    parser.add_argument("--generations", type=int, help="Override profile generations.")
    parser.add_argument("--tournament-size", type=int, help="Override tournament size.")
    parser.add_argument("--max-depth", type=int, help="Override max expression depth.")
    parser.add_argument("--max-operators", type=int, help="Override max operators per expression.")
    parser.add_argument("--n-parallel", type=int, help="Override number of parallel workers.")
    parser.add_argument(
        "--auto-parallel",
        action="store_true",
        help="Auto-tune parallel workers to use most of the machine while leaving configurable headroom.",
    )
    parser.add_argument(
        "--cpu-headroom",
        type=int,
        default=2,
        help="When auto-parallel is enabled, keep at least this many CPU threads free.",
    )
    parser.add_argument(
        "--memory-headroom-gb",
        type=float,
        default=2.0,
        help="When auto-parallel is enabled, leave roughly this much RAM free.",
    )
    parser.add_argument(
        "--parallel-cap",
        type=int,
        default=0,
        help="Optional hard cap for auto-parallel workers. 0 means backend default.",
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Skip writing the latest run summary artifact.",
    )
    parser.add_argument(
        "--scoring",
        choices=("worldquant", "internal"),
        default="worldquant",
        help="Evaluation backend. 'internal' keeps scoring fully local and never calls WorldQuant.",
    )
    parser.add_argument(
        "--internal-settings",
        help=(
            "Optional settings string for internal scoring, for example "
            "'USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market'."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize logger
    logger = Logger(
        job_name="brain-learn",
        console_log=True,
        file_log=True,
        logs_directory="logs",
        incremental_run_number=True
    )
    
    metric = None
    s = None
    if args.scoring == "worldquant":
        s = create_session(logger)
        if not s:
            logger.error("Exiting due to authentication failure.")
            return 1
    else:
        from src.internal_scoring import build_internal_metric

        logger.log(
            "Using internal proxy scoring. No WorldQuant authentication or remote submission will be performed."
        )
        metric = build_internal_metric(logger=logger, settings=args.internal_settings)

    profile = resolve_profile(
        args.mode,
        population_size=args.population_size,
        generations=args.generations,
        tournament_size=args.tournament_size,
        max_depth=args.max_depth,
        max_operators=args.max_operators,
        n_parallel=args.n_parallel,
    )
    machine_resources = detect_machine_resources()
    if args.auto_parallel and args.n_parallel is None:
        auto_workers = recommend_parallel_workers(
            scoring=args.scoring,
            cpu_headroom=args.cpu_headroom,
            memory_headroom_gb=args.memory_headroom_gb,
            parallel_cap=args.parallel_cap,
            machine_resources=machine_resources,
        )
        profile = replace(profile, n_parallel=max(profile.n_parallel, auto_workers))
    elif args.auto_parallel and args.n_parallel is not None:
        logger.log(
            f"Auto-parallel ignored because --n-parallel={args.n_parallel} was provided explicitly."
        )
    init_population = load_initial_population(logger, max_programs=profile.max_seed_programs)
    logger.log(
        "Machine resource snapshot: "
        f"cpu_count={machine_resources.get('cpu_count')}, "
        f"memory_gb={machine_resources.get('memory_gb')}"
    )
    logger.log(
        f"Using run profile '{profile.name}' with scoring='{args.scoring}' "
        f"(population={profile.population_size}, generations={profile.generations}, "
        f"max_depth={profile.max_depth}, max_operators={profile.max_operators}, "
        f"n_parallel={profile.n_parallel}, max_seed_programs={profile.max_seed_programs})"
    )
    run_seed = _build_run_seed()
    logger.log(f"Using run seed {run_seed}")
    
    # Run the GPLearn simulator
    from src.genetic import GPLearnSimulator
    simulator = GPLearnSimulator(
        session=s,
        population_size = profile.population_size,
        generations = profile.generations,
        tournament_size = profile.tournament_size,
        p_crossover = profile.p_crossover,
        p_mutation = profile.p_mutation,
        p_subtree_mutation = profile.p_subtree_mutation,
        parsimony_coefficient = profile.parsimony_coefficient,
        random_state = run_seed,
        init_population = init_population,
        metric=metric,
        max_depth = profile.max_depth,
        max_operators = profile.max_operators,
        n_parallel = profile.n_parallel,
        logger=logger
        )
    simulator.evolve()

    if not args.skip_artifacts:
        artifact_path = save_run_summary(simulator, profile)
        logger.log(f"Saved run summary to {artifact_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
