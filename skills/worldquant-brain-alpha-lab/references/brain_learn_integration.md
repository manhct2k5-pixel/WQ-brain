# brain-learn Integration Notes

Use this file when the user's files look like the provided `brain-learn-main.zip`.
These notes are inferred from the local repo snapshot, not official WorldQuant documentation.

## Relevant files

- `main.py`: creates an authenticated session, loads `initial-population.pkl`, and launches `GPLearnSimulator`
- `src/brain.py`: posts simulations and fetches alpha performance
- `src/function.py`: defines the search space of terminals and operators
- `src/program.py`: builds and validates postfix expression trees
- `src/genetic.py`: runs the evolutionary loop

## Simulation defaults found in `src/brain.py`

The repo currently sends simulations with these defaults:

- `instrumentType`: `EQUITY`
- `region`: `USA`
- `universe`: `TOP3000`
- `delay`: `1`
- `decay`: `1`
- `neutralization`: `INDUSTRY`
- `truncation`: `0.1`
- `pasteurization`: `ON`
- `unitHandling`: `VERIFY`
- `nanHandling`: `OFF`
- `language`: `FASTEXPR`

Treat these as editable assumptions, not sacred defaults.

## Search-space inventory in `src/function.py`

The provided repo already contains a useful but opinionated set of building blocks.

Terminal families:

- price: `open`, `close`, `high`, `low`, `vwap`
- liquidity: `volume`, `ts_mean(volume, 63)`
- return and volatility: `returns`, `ts_std_dev(returns, 63)`
- regime or risk: beta and correlation to SPY, systematic and unsystematic risk
- handcrafted features: reversal, momentum, VWAP spread, Bollinger-like zscore, Williams-like range term

Operator families:

- arithmetic: add, subtract, multiply, divide, min, max
- unary transforms: reverse, inverse, abs, sign, rank, zscore, winsorize
- time-series transforms: `ts_rank`, `ts_sum`, `ts_zscore`, `ts_std_dev`
- relationships: `ts_corr`, ranked correlation, `ts_regression`

This means the fastest wins usually come from better search discipline and analysis, not from adding dozens of extra operators on day one.

## Repo caveats worth checking before editing

1. `main.py` unconditionally loads `initial-population.pkl`

If that file is missing, the run will fail before evolution starts. Guard it or make it optional when productizing the repo.

2. CSV write and read paths disagree

`save_alpha_to_csv()` writes a headered `simulation_results.csv`, but `read_simulations_csv()` assumes a headerless `simulations.csv` with a fixed column order.
Parse result files defensively instead of trusting one function.

3. Unit handling around division is suspicious

The `DIV` operator's unit rule currently multiplies units instead of dividing them.
If you depend on unit checks, verify this before extending the operator set.

4. Tests look stale relative to the operator inventory

The test suite references `SQRT` even though it is commented out in `src/function.py`.
Do not assume tests fully describe the current search space.

5. Parallelism is capped

`GPLearnSimulator` limits `n_parallel` to 3 and already includes retry logic for simulation rate limits.
Respect that when scaling runs.

## Practical improvement order

When improving this repo, usually do this first:

1. add result digestion and near-duplicate detection
2. clean up startup and CSV handling
3. tune weights and operator families for diversity
4. only then increase population or generations

## How to use these notes

If the user wants code changes, tie your proposal to one of these concrete leverage points.
If the user wants alpha ideas only, use the existing terminals and operators as the default vocabulary.
