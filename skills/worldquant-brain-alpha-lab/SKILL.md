---
name: worldquant-brain-alpha-lab
description: Research, audit, and iterate WorldQuant BRAIN alpha ideas, FastExpr formulas, simulation settings, and brain-learn style search repos. Use this skill whenever the user mentions WorldQuant Brain, Brain competition, alpha generation, FastExpr, Sharpe, Fitness, turnover, simulation results, or wants to improve a WorldQuant alpha-search agent. Prefer this skill even when the user only says "make better alphas", "analyze my Brain repo", or "fix my alpha generator."
compatibility: Works best with local repos, zip archives, CSV exports, and logs that contain Brain expressions, simulation results, or automation code.
---

# WorldQuant Brain Alpha Lab

## Mission

Build a disciplined research assistant for WorldQuant Brain style alpha work.
Optimize for repeatable research quality, expression diversity, and clear iteration loops.
Do not promise profits, a contest win, or guaranteed approval.

Mirror the user's language.
If the user asks for "perfect" or guaranteed outcomes, reset expectations gently and continue with the strongest realistic system.

## When to Read Extra Files

- Read [references/brain_learn_integration.md](references/brain_learn_integration.md) when the repo looks like the provided `brain-learn-main.zip` or has `src/function.py`, `src/genetic.py`, `simulation_results.csv`, or WorldQuant API calls.
- Read [references/alpha_patterns.md](references/alpha_patterns.md) when you need new alpha theses, variation ideas, or a way to diversify a batch.
- Use [scripts/expression_lint.py](scripts/expression_lint.py) before finalizing an alpha batch or when comparing many candidate formulas.
- Use [scripts/results_digest.py](scripts/results_digest.py) whenever a `simulation_results.csv`, `simulations.csv`, or similar export exists.

## Default Workflow

### 1. Classify the task

Pick the main mode early:

- `repo-audit`: inspect an existing codebase or zip and find leverage points
- `result-analysis`: digest prior simulations and explain what to keep or kill
- `alpha-generation`: create a fresh batch of candidate expressions
- `search-space-tuning`: edit operators, terminals, weights, or GP parameters
- `automation-repair`: fix API/auth/rate-limit/logging issues

The user may ask for several at once. Keep the critical path small and sequence the rest.

### 2. Gather the minimum working context

Collect these before making bold claims:

- repo path or zip path
- latest result CSV or log
- current simulation defaults
- operator and terminal inventory
- examples of expressions that already failed or worked

If the user only gives a zip, inspect it with `python3` and `zipfile` and extract only the files you need first.

Never print credentials from `.env`.

### 3. Profile the current system

When a `brain-learn` style repo is present:

- inspect `main.py` for run settings
- inspect `src/brain.py` for simulation payload defaults
- inspect `src/function.py` for operator and terminal search space
- inspect `src/genetic.py` for mutation, crossover, hall-of-fame, and parallelism
- inspect any result CSV before proposing code changes

Call out bottlenecks that change alpha quality materially:

- narrow or low-diversity operator space
- too many near-clone expressions
- missing post-simulation analysis
- search parameters that encourage bloat
- CSV or logging mismatches
- unsafe assumptions about rate limits or authentication

### 4. Build research theses before formulas

Do not generate a batch of expressions that differ only by window numbers.
Start from 2 to 4 distinct theses, then map each thesis to 1 to 3 formulas.

Good thesis buckets:

- price-volume disagreement
- short-horizon reversal with liquidity or volatility conditioning
- residual or de-beta structure
- VWAP or open-close dislocation
- surprise or volatility normalization
- momentum or reversal regime switch

For each thesis, explain:

- what inefficiency it targets
- why the chosen terminals fit
- what makes it different from the rest of the batch
- which checks are most likely to fail

### 5. Build the batch with diversity rules

Unless the user asks otherwise, aim for 6 to 12 expressions in one batch.

Batch rules:

- no more than 2 expressions from the same thesis
- at least 3 distinct operator skeletons
- at least 3 distinct terminal combinations
- avoid tiny variants that only tweak one lookback
- prefer expressions that can be normalized cross-sectionally
- keep expressions interpretable enough to debug after the run

Before presenting the final batch, lint it with `scripts/expression_lint.py`.

### 6. Analyze results like a research loop, not a leaderboard

Use `scripts/results_digest.py` when possible.

Interpret failures as signals:

- `LOW_SHARPE`: thesis is weak or normalization is poor
- `LOW_FITNESS`: alpha may be noisy, fragile, or too expensive in turnover
- `HIGH_TURNOVER`: slow the signal, smooth inputs, or move to broader horizons
- `LOW_TURNOVER`: speed up the signal or add a more reactive component
- `CONCENTRATED_WEIGHT`: rank, zscore, winsorize, or stabilize denominators
- `SELF_CORRELATION`: diversify thesis, not just parameters
- `LOW_SUB_UNIVERSE_SHARPE`: reduce niche behavior and prefer broader structure

Always recommend the next experiment based on the dominant failure mode.

## Output Format

When doing alpha research, default to this shape:

### Context

- repo or files inspected
- current settings that matter
- strongest constraints or unknowns

### Research theses

List 2 to 4 thesis statements in plain language.

### Candidate batch

For each candidate include:

- `id`
- `expression`
- `thesis`
- `why it might work`
- `why it is meaningfully different`
- `likely failure modes`

### Next iteration

- what to simulate first
- what metric or check decides whether to keep the thesis
- what code or settings to change next if results disappoint

## Working With the Provided brain-learn Repo

Use the repo as a starting engine, not as ground truth.

When editing it:

- make small search-space changes before broad refactors
- preserve authentication and rate-limit safety
- explain how any weight or operator change alters the search distribution
- prefer deterministic analysis scripts over hand-wavy summaries
- treat CSV parsing defensively because this repo has format inconsistencies

When asked to improve the generator itself, prioritize:

1. better result analysis and de-duplication
2. more deliberate operator and terminal diversity
3. smaller, more interpretable experiments
4. only then larger evolutionary runs

## Current-Info Safety

If you need current WorldQuant Brain platform details and they might have changed, verify them against official sources before stating them as fact.
When you are inferring from the user's local repo instead of official documentation, say so.

## Anti-Patterns

Avoid these:

- promising a guaranteed winning alpha
- flooding the user with dozens of near-identical formulas
- changing windows without changing thesis
- proposing giant expression trees that are impossible to debug
- ignoring turnover, self-correlation, or concentration checks
- recommending code edits before reading the existing search space
- leaking `.env` or auth details into logs or output
