import requests
import numpy as np
import time
import os
from src.utils import create_authenticated_session, evaluate_fitness
from src.program_tokens import serialize_token_program
from datetime import timedelta
from src.program import Program
from itertools import batched
import concurrent.futures
import threading
import warnings
import heapq  # Import heapq for efficient HOF management

from src.seed_store import SeedStoreCorruptError, load_seed_store, write_seed_store
from src.submit_gate import local_metrics_pass_submit_gate

# Maximum number of reconnection attempts
MAX_RECONNECTION_ATTEMPTS = 5
HARD_MAX_PARALLEL_WORKERS = max(1, min(int(os.cpu_count() or 1), 32))
PARALLEL_EVAL_BATCH_SIZE = 50
PARALLEL_EVAL_TIMEOUT_SECONDS = 10.0
IS_TESTS = ['LOW_SHARPE', 'LOW_FITNESS', 'LOW_TURNOVER', 'HIGH_TURNOVER', 
         'CONCENTRATED_WEIGHT', 'LOW_SUB_UNIVERSE_SHARPE', 'SELF_CORRELATION',
         'MATCHES_COMPETITION']


def _normalize_parallel_workers(n_parallel):
    try:
        requested = int(n_parallel)
    except (TypeError, ValueError):
        requested = 1
    return max(1, min(requested, HARD_MAX_PARALLEL_WORKERS))


def _normalize_probability_pair(first, second):
    total = first + second
    if total <= 0:
        return 0.5, 0.5
    return first / total, second / total


def _normalize_probability_triplet(first, second, third):
    total = first + second + third
    if total <= 0:
        return (1.0 / 3.0,) * 3
    return first / total, second / total, third / total

class GPLearnSimulator:
    def __init__(self, 
                 population_size=30,
                 generations=20,
                 tournament_size=5,
                 p_crossover=0.7,
                 p_mutation=0.1,
                 p_subtree_mutation=0.05,
                 p_hoist_mutation=0.05,
                 p_point_mutation=0.1,
                 max_depth=5,
                 max_operators=10,
                 session = None,
                 random_state=None,
                 parsimony_coefficient=0.15,
                 n_parallel=3,
                 init_population=None,
                 metric=None,
                 logger=None,
                 hof_size=50):
        """
        A Genetic Programming simulator optimized for simulation-based fitness evaluation.
        
        Parameters
        ----------
        population_size : int
            Size of the population
        generations : int
            Number of generations to evolve
        tournament_size : int
            Size of tournament for selection
        p_crossover : float
            Probability of crossover
        p_mutation : float
            Probability of mutation
        p_subtree_mutation : float
            Probability of subtree mutation
        p_hoist_mutation : float
            Probability of hoist mutation
        p_point_mutation : float
            Probability of point mutation
        max_depth : int
            Maximum tree depth
        max_operators : int
            Maximum number of operators
        random_state : int or RandomState, optional
            Random number generator
        metric : callable, optional
            Function to evaluate fitness
        parsimony_coefficient : float
            Coefficient for parsimony pressure (penalty for size)
        n_parallel : int, optional
            Number of parallel workers for fitness evaluation. Capped to the local hardware limit.
        init_population : list, optional
            List of Program objects to initialize the population with.
        logger : Logger, optional
            Logger instance for logging messages.
        hof_size : int, optional
            Number of best individuals to keep track of in the Hall-of-Fame (default is 50).
        """
        # Basic parameters
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.max_operators = max_operators
        self.parsimony_coefficient = parsimony_coefficient
        self.n_parallel = _normalize_parallel_workers(n_parallel)
        self.logger = logger
        self.hof_size = hof_size
        
        # Preserve the configured probabilities for callers and normalize only
        # at selection time so the public attributes stay backward compatible.
        self.p_crossover = float(p_crossover)
        self.p_mutation = float(p_mutation)
        self._operation_probabilities = _normalize_probability_pair(
            self.p_crossover,
            self.p_mutation,
        )
        
        self.p_subtree_mutation = float(p_subtree_mutation)
        self.p_hoist_mutation = float(p_hoist_mutation)
        self.p_point_mutation = float(p_point_mutation)
        self._mutation_probabilities = _normalize_probability_triplet(
            self.p_subtree_mutation,
            self.p_hoist_mutation,
            self.p_point_mutation,
        )
        
        # Setup session and metric
        self.session = session
        self.metric = metric or evaluate_fitness(session, logger=logger)
        
        # Setup random state
        if isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state or np.random.RandomState()
            
        # Initialize tracking variables
        self.population = []
        self.seen_expressions = set()
        self.evaluated_expressions = set()
        if init_population:
            self._create_initial_programs(init_population)
            
        self.history = []
        self.best_program = None
        self.best_fitness = float('-inf')
        self.generation = 0
        self.hall_of_fame = []
        self._seed_store_lock = threading.Lock()
        
        # Statistics
        self.fitness_evaluations = 0
        self.start_time = None
        self._session_lock = threading.Lock()
        
    def _create_initial_programs(self, programs):
        """Create Program objects from the provided initial programs."""
        normalized_programs = []

        if isinstance(programs, dict):
            iterable = programs.values()
        else:
            iterable = programs

        for item in iterable:
            candidate = item
            if isinstance(item, dict):
                candidate = item.get("program")
            elif hasattr(item, "program"):
                candidate = item.program

            if candidate:
                normalized_programs.append(candidate)

        self.population = []
        for program in normalized_programs:
            try:
                compiled = Program(
                    max_depth=self.max_depth,
                    max_operators=self.max_operators,
                    random_state=self.random_state,
                    metric=self.metric,
                    parimony_coefficient=self.parsimony_coefficient,
                    program=program,
                )
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"Skipping invalid seed program: {exc}")
                continue

            self.population.append(compiled)
            self.seen_expressions.add(str(compiled))
        
    def _initialize_population(self):
        """Initialize population with random programs, avoiding duplicates."""
        # Create random programs until we reach the desired population size
        while len(self.population) < self.population_size:
            program = Program(
                max_depth=self.max_depth,
                max_operators=self.max_operators,
                random_state=self.random_state,
                metric=self.metric,
                parimony_coefficient=self.parsimony_coefficient
            )
            program_str = str(program)
            
            # Only add if not seen before
            if program_str not in self.seen_expressions:
                self.population.append(program)
                self.seen_expressions.add(program_str)
        
        if self.logger:
            self.logger.log(f"Initialized population with {len(self.population)} programs")

    def _clone_program(self, program):
        """Create a detached copy of a Program, preserving cached fitness."""
        cloned = Program(
            max_depth=self.max_depth,
            max_operators=self.max_operators,
            random_state=self.random_state,
            metric=self.metric,
            parimony_coefficient=self.parsimony_coefficient,
            program=program.reproduce(),
        )
        cloned.raw_fitness = program.raw_fitness
        cloned.fitness = program.fitness
        return cloned
    
    def _recreate_session(self):
        """Recreate the session if authentication fails."""
        from dotenv import load_dotenv
        
        # Load credentials from .env file
        load_dotenv()
        username = os.getenv("USERNAME")
        password = os.getenv("PASSWORD")

        self.session, response = create_authenticated_session(
            username,
            password,
            logger=self.logger,
            context="Session reconnection",
        )

        if self.session is not None:
            if self.logger:
                self.logger.log("Session reconnected successfully.")
            else:
                print("Session reconnected successfully.")
            self.metric = evaluate_fitness(self.session, logger=self.logger)
            return True

        if self.logger and response is not None:
            self.logger.error(f"Failed to reconnect session. Status Code: {response.status_code}")
            self.logger.error(f"Response: {response.text}")
        elif response is not None:
            print(f"Failed to reconnect session. Status Code: {response.status_code}")
            print(f"Response: {response.text}")
        return False
    
    def _meets_hof_threshold(self, result):
        """Checks if a simulation result meets the criteria for Hall of Fame."""
        if not result:
            return False

        # Check if result passes all specified tests
        passes_tests = all(result.get(k) == 'PASS' for k in IS_TESTS if k in result)
        
        # Check threshold values
        sharpe = result.get('sharpe')
        fitness = result.get('fitness')
        high_sharpe = (sharpe if sharpe is not None else -float('inf')) > 2.0
        high_fitness = (fitness if fitness is not None else -float('inf')) >= 1.5

        return passes_tests or high_sharpe or high_fitness

    def _should_autosave_to_seed_store(self, result):
        """Persist only results that already look close to submit-ready."""
        if not isinstance(result, dict) or not result:
            return False

        passes_tests = all(result.get(k) == 'PASS' for k in IS_TESTS if k in result)
        if not passes_tests:
            return False

        sharpe = result.get('sharpe')
        fitness = result.get('fitness')
        try:
            sharpe_value = float(sharpe) if sharpe is not None else float('-inf')
        except (TypeError, ValueError):
            sharpe_value = float('-inf')
        try:
            fitness_value = float(fitness) if fitness is not None else float('-inf')
        except (TypeError, ValueError):
            fitness_value = float('-inf')

        if {'verdict', 'alpha_score'}.issubset(result):
            return local_metrics_pass_submit_gate(result)

        return sharpe_value >= 1.4 and fitness_value >= 1.0

    def _evaluate_program_expression(self, program_str):
        """Evaluate a program string and return the raw metric payload."""
        result = None
        for attempt in range(MAX_RECONNECTION_ATTEMPTS):
            try:
                with self._session_lock:
                    current_session = self.session
                    metric_func = self.metric
                    requires_session = getattr(metric_func, "requires_session", False)

                if current_session is None and requires_session:
                    if self.logger:
                        self.logger.warning("Session is None, attempting to recreate.")
                    with self._session_lock:
                        if not self._recreate_session():
                            if self.logger:
                                self.logger.error("Failed to recreate session, cannot evaluate.")
                            break
                        metric_func = evaluate_fitness(self.session, logger=self.logger)

                result = metric_func(program_str)

                if result is None:
                    if not requires_session or self.session is None:
                        if self.logger:
                            self.logger.error(f"Metric returned None. Skipping program: {program_str[:50]}...")
                        break

                    with self._session_lock:
                        try:
                            test_response = self.session.get('https://api.worldquantbrain.com/authentication', timeout=10)
                            if test_response.status_code != 200:
                                if self.logger:
                                    self.logger.warning(f"Session check failed ({test_response.status_code}), recreating.")
                                if self._recreate_session():
                                    self.metric = evaluate_fitness(self.session, logger=self.logger)
                                    continue
                            else:
                                if self.logger:
                                    self.logger.log("Session check OK, but metric returned None. Retrying.")
                        except requests.exceptions.RequestException as e:
                            if self.logger:
                                self.logger.warning(f"Session check error: {e}. Attempting recreation.")
                            if self._recreate_session():
                                self.metric = evaluate_fitness(self.session, logger=self.logger)
                                continue
                    if self.logger:
                        self.logger.error(f"Metric returned None. Skipping program: {program_str[:50]}...")
                    break

                if result is not None:
                    break

            except requests.exceptions.Timeout:
                if self.logger:
                    self.logger.warning(f"Timeout: {program_str[:50]}... Attempt {attempt+1}/{MAX_RECONNECTION_ATTEMPTS}")
                if attempt < MAX_RECONNECTION_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
                else:
                    result = {"error": "timeout"}

            except requests.exceptions.RequestException as e:
                if self.logger:
                    self.logger.error(f"Request error: {program_str[:50]}... {e}. Attempt {attempt+1}/{MAX_RECONNECTION_ATTEMPTS}")
                if attempt < MAX_RECONNECTION_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
                else:
                    result = {"error": f"request error: {e}"}

            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Evaluation error: {program_str[:50]}... {e}. "
                        f"Attempt {attempt+1}/{MAX_RECONNECTION_ATTEMPTS}"
                    )
                if attempt < MAX_RECONNECTION_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
                else:
                    result = {"error": f"evaluation error: {e}"}

        return result

    def _apply_program_evaluation(self, program, program_str, result):
        """Apply a metric payload to a Program and update shared tracking."""
        self.evaluated_expressions.add(program_str)
        self.seen_expressions.add(program_str)

        if result is not None and "error" not in result:
            raw_score = result.get('fitness')
            if raw_score is None:
                raw_score = result.get('sharpe')
            if raw_score is None:
                program.raw_fitness = float('-inf')
                program.fitness = float('-inf')
                if self.logger:
                    self.logger.warning(
                        f"Evaluation returned no usable fitness for: {program_str[:50]}..."
                    )
                return program_str, result

            self.fitness_evaluations += 1

            program.raw_fitness = raw_score
            penalty = self.parsimony_coefficient * program.length()
            program.fitness = program.raw_fitness - penalty
            result['final_fitness'] = program.fitness

            if self.logger:
                self.logger.log(f"Program fitness: {program_str[:50]}... = {program.fitness:.4f}")

            if self._meets_hof_threshold(result):
                fitness_for_hof = result.get('fitness')
                if fitness_for_hof is None:
                    fitness_for_hof = result.get('sharpe', -float('inf'))
                entry = (fitness_for_hof, program_str, result)

                if len(self.hall_of_fame) < self.hof_size:
                    heapq.heappush(self.hall_of_fame, entry)
                else:
                    heapq.heappushpop(self.hall_of_fame, entry)

                if self._should_autosave_to_seed_store(result):
                    self._save_to_initial_population(program, result, fitness_for_hof)
        else:
            program.raw_fitness = float('-inf')
            program.fitness = float('-inf')
            if self.logger:
                self.logger.warning(f"Evaluation failed for: {program_str[:50]}...")

        return program_str, result

    def _evaluate_single_program(self, program):
        """
        Evaluate the fitness of a single program, handling retries and session recreation.
        
        Returns:
        -------
        tuple: (program_str, result_dict or None if skipped/failed)
        """
        program_str = str(program)

        # Skip if already evaluated
        if program_str in self.evaluated_expressions:
            return program_str, None

        result = self._evaluate_program_expression(program_str)
        return self._apply_program_evaluation(program, program_str, result)
        
    def _save_to_initial_population(self, program, result, fitness):
        """Save a program to the initial population file."""
        try:
            with self._seed_store_lock:
                seed_path = 'initial-population.pkl'
                existing_data = load_seed_store(seed_path, on_corrupt="raise")
                program_str = str(program)
                stored_result = dict(result)
                stored_result.setdefault('status', 'PASS')
                stored_result.setdefault('source', 'auto_hof')
                existing_data[program_str] = {
                    'program': serialize_token_program(program.program),
                    'fitness': fitness,
                    'result': stored_result,
                }
                write_seed_store(seed_path, existing_data)
        except SeedStoreCorruptError as e:
            if self.logger:
                self.logger.error(f"Error saving to init-population.pkl: {e}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving to init-population.pkl: {e}")

    def _evaluate_fitness(self, program):
        """Deprecated: Use parallel_evaluate_fitness."""
        warnings.warn("_evaluate_fitness is deprecated. Use parallel_evaluate_fitness instead.", DeprecationWarning)
        # Call _evaluate_single_program for consistency
        _, result = self._evaluate_single_program(program)
        return program.raw_fitness, program.fitness


    def parallel_evaluate_fitness(self, programs_to_evaluate, n_parallel=None):
        """
        Evaluate fitness for a list of programs in parallel.
        Manages evaluated_expressions and HOF updates.
        """
        if not programs_to_evaluate:
            return
            
        if n_parallel is None:
            n_parallel = self.n_parallel
        n_parallel = _normalize_parallel_workers(n_parallel)

        # Filter out already evaluated programs
        to_evaluate = []
        for program in programs_to_evaluate:
            program_str = str(program)
            if program_str not in self.evaluated_expressions:
                to_evaluate.append(program)
            # If already evaluated, fitness should be set elsewhere

        if not to_evaluate:
            return

        if self.logger:
            self.logger.log(f"Evaluating {len(to_evaluate)} programs (pool size {n_parallel})...")

        start_time = time.time()

        # Run evaluations in bounded batches so a single slow metric does not
        # stall the whole population forever.
        for batch in batched(to_evaluate, PARALLEL_EVAL_BATCH_SIZE):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel)
            futures = {}
            try:
                for program in batch:
                    program_str = str(program)
                    futures[executor.submit(self._evaluate_program_expression, program_str)] = (
                        program,
                        program_str,
                    )

                done, not_done = concurrent.futures.wait(
                    futures,
                    timeout=PARALLEL_EVAL_TIMEOUT_SECONDS,
                )

                for future in done:
                    program, program_str = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        if self.logger:
                            self.logger.error(f'Program {program_str[:50]}... generated an exception: {exc}')
                        result = {"error": f"evaluation error: {exc}"}
                    self._apply_program_evaluation(program, program_str, result)

                for future in not_done:
                    program, program_str = futures[future]
                    future.cancel()
                    if self.logger:
                        self.logger.warning(
                            f"Parallel evaluation timed out after {PARALLEL_EVAL_TIMEOUT_SECONDS:.1f}s: {program_str[:50]}..."
                        )
                    self._apply_program_evaluation(
                        program,
                        program_str,
                        {"error": f"timeout after {PARALLEL_EVAL_TIMEOUT_SECONDS:.1f}s"},
                    )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        # Ensure all programs have fitness values
        for program in programs_to_evaluate:
            if program.fitness is None:
                program.raw_fitness = float('-inf')
                program.fitness = float('-inf')

        duration = time.time() - start_time
        if self.logger:
            self.logger.log(f"Batch evaluation completed in {duration:.2f}s")
            if self.hall_of_fame and len(self.hall_of_fame) > 0:
                best_hof = max(f[0] for f in self.hall_of_fame)
                self.logger.log(f"HOF: {len(self.hall_of_fame)} entries, best={best_hof:.4f}")


    def _tournament_selection(self):
        """Select a program using tournament selection."""
        indices = self.random_state.randint(0, len(self.population), self.tournament_size)
        tournament = [self.population[i] for i in indices]
        
        # Return the best individual in the tournament based on pre-calculated fitness
        # Handle potential None fitness values gracefully if evaluation failed.
        return max(tournament, key=lambda program: program.fitness if program.fitness is not None else float('-inf'))
    
    def _update_best(self):
        """Update the best program found so far based on the main population."""
        # Find the best program in the current population
        if not self.population:
            return
            
        current_best = max(self.population, 
                          key=lambda program: program.fitness if program.fitness is not None else float('-inf'))
        
        # Handle case where fitness might be None
        if current_best.fitness is None:
            return
            
        # Update best program if the current best is better
        if current_best.fitness > self.best_fitness:
            self.best_program = self._clone_program(current_best)
            self.best_fitness = current_best.fitness

    def evolve(self, verbose=True, log_interval=1):
        """Evolve the population over generations."""
        if self.start_time is None:
            self.start_time = time.time()

        # Initialize population if needed
        if not self.population:
            self._initialize_population()

        # Initial evaluation of the starting population
        if self.logger:
            self.logger.log(f"Initial evaluation of {len(self.population)} programs...")
        # Ensure all initial programs are evaluated and HOF is populated
        needs_evaluation = [p for p in self.population if p.fitness is None]
        self.parallel_evaluate_fitness(needs_evaluation)
        self._update_best() # Update best based on initial population

        if verbose and self.logger:
            self.logger.log(f"Generation {self.generation}: Best Fitness={self.best_fitness:.4f}")

        for gen in range(1, self.generations + 1):
            self.generation = gen
            start_gen_time = time.time()

            next_population = []
            next_gen_strings = set()

            # Add elitism: Keep the best program from the previous generation
            if self.best_program is not None:
                elite_program = self._clone_program(self.best_program)
                best_str = str(elite_program)
                next_population.append(elite_program)
                next_gen_strings.add(best_str)
                if self.logger and gen % log_interval == 0:
                    self.logger.log(f"Elite: {best_str[:50]}... (Fitness: {self.best_fitness:.4f})")

            # Generate new population through crossover and mutation
            while len(next_population) < self.population_size:
                operation = self.random_state.choice(
                    ['crossover', 'mutation'],
                    p=self._operation_probabilities,
                )
                
                if operation == 'crossover':
                    parent1 = self._tournament_selection()
                    parent2 = self._tournament_selection()
                    
                    # Ensure parents are different
                    attempts = 0
                    while parent1 is parent2 and attempts < 10:
                        parent2 = self._tournament_selection()
                        attempts += 1
                    
                    # Get offspring programs
                    offspring1_program, _, _ = parent1.crossover(parent2.program, self.random_state)
                    offspring2_program, _, _ = parent2.crossover(parent1.program, self.random_state)
                    
                    # Create new Program instances
                    offspring1 = Program(
                        max_depth=self.max_depth,
                        max_operators=self.max_operators,
                        random_state=self.random_state,
                        metric=self.metric,
                        parimony_coefficient=self.parsimony_coefficient,
                        program=offspring1_program
                    )
                    
                    offspring2 = Program(
                        max_depth=self.max_depth,
                        max_operators=self.max_operators,
                        random_state=self.random_state,
                        metric=self.metric,
                        parimony_coefficient=self.parsimony_coefficient,
                        program=offspring2_program
                    )
                    
                    offspring = [offspring1, offspring2]
                else: # Mutation
                    parent = self._tournament_selection()
                    mutation_op = self.random_state.uniform()
                    
                    subtree_prob, hoist_prob, _ = self._mutation_probabilities
                    if mutation_op < subtree_prob:
                        # subtree_mutation returns (program, removed, donor_removed)
                        mutation_result = parent.subtree_mutation(self.random_state)
                        offspring_program = mutation_result[0]
                    elif mutation_op < subtree_prob + hoist_prob:
                        # hoist_mutation returns (program, removed)
                        offspring_program, _ = parent.hoist_mutation(self.random_state)
                    else:
                        # point_mutation returns (program, mutated_indices)
                        offspring_program, _ = parent.point_mutation(self.random_state)
                    
                    offspring = [Program(
                        max_depth=self.max_depth,
                        max_operators=self.max_operators,
                        random_state=self.random_state,
                        metric=self.metric,
                        parimony_coefficient=self.parsimony_coefficient,
                        program=offspring_program
                    )]

                # Add unique offspring to next generation
                for child in offspring:
                    child_str = str(child)
                    if child_str not in next_gen_strings:
                        next_population.append(child)
                        next_gen_strings.add(child_str)
                        if len(next_population) >= self.population_size:
                            break # Exit loop when population is full
                            
                # Exit outer loop if population is full
                if len(next_population) >= self.population_size:
                    break

            # Evaluate the new generation
            if self.logger:
                self.logger.log(f"\n--- Generation {gen} ---")
            self.parallel_evaluate_fitness(next_population)

            # Update population and best program
            self.population = next_population
            self._update_best()

            # Store history with consistent metrics
            avg_fitness = np.mean([p.fitness for p in self.population 
                                  if p.fitness is not None and p.fitness > -np.inf])
            best_hof_fitness = max((f[0] for f in self.hall_of_fame), default=-float('inf'))
            
            self.history.append({
                'generation': gen,
                'best_fitness': self.best_fitness,
                'avg_fitness': avg_fitness,
                'population_size': len(self.population),
                'hof_size': len(self.hall_of_fame),
                'best_hof_fitness': best_hof_fitness
            })

            # Log progress if necessary
            end_gen_time = time.time()
            gen_duration = end_gen_time - start_gen_time

            if verbose and self.logger and (gen % log_interval == 0 or gen == self.generations):
                elapsed_time = timedelta(seconds=int(end_gen_time - self.start_time))
                self.logger.log(
                    f"Gen {gen:>{len(str(self.generations))}}: "
                    f"Best={self.best_fitness:.4f}, "
                    f"Avg={avg_fitness:.4f}, "
                    f"HOF Best={best_hof_fitness:.4f} (Size:{len(self.hall_of_fame)}), "
                    f"Evals={self.fitness_evaluations}, "
                    f"Time={gen_duration:.2f}s, "
                    f"Total={elapsed_time}"
                )
        
        return self.best_program

    def get_best_individual(self):
        """Return the best Program instance found during the evolution."""
        return self.best_program


    def get_hall_of_fame(self):
        """Returns the entire Hall of Fame, sorted by fitness descending."""
        # Sort by fitness (descending) before returning
        return sorted(self.hall_of_fame, key=lambda item: item[0], reverse=True)

    def get_fitness_history(self):
        """Return the history of best fitness values."""
        return [stats['best_fitness'] for stats in self.history]
    
    def get_all_history(self):
        """Return all tracked statistics."""
        return self.history
