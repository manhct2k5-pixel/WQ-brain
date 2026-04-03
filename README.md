# brain-learn

A genetic programming framework for the [WorldQuant BRAIN platform](https://platform.worldquantbrain.com/), inspired by [gplearn](https://github.com/trevorstephens/gplearn).

## Setup

1.  **Environment:** Configure your Python environment using [`uv`](https://docs.astral.sh/uv/):
    ```bash
    uv sync
    ```
    This command creates a python virtual environment in `.venv` and installs the dependencies listed in `pyproject.toml`.
    Before using `bash run_wsl.sh ...`, activate that environment in the current WSL shell:
    ```bash
    source .venv/bin/activate
    ```

2.  **Configuration:**
    *   Create a `.env` file in the project root directory. The file should contain necessary credentials, particularly `USERNAME` and `PASSWORD`.
    *   You can copy `.env.example` first, then replace the placeholder values with your real WorldQuant Brain credentials.
    *   You can keep the defaults in `main.py` and choose the run profile from the command line (`--mode test|careful|smart|light|full`).

3.  **Initial Population (Optional):** `main.py` now loads `initial-population.pkl` only when it exists. Good seed entries are written automatically when strong programs reach the hall of fame, so later runs can start from better expressions without crashing when the file is missing.

## Running the Framework

### Recommended quick start on WSL

This repo is documented for WSL usage. If your repo path looks like `/mnt/d/...`, open WSL bash and run:

```bash
cd /mnt/d/skill-creator/brain-learn-main
source .venv/bin/activate
bash run_wsl.sh doctor
bash run_wsl.sh auth
bash run_wsl.sh run
```

`run_wsl.sh doctor` checks your Python version, dependencies, repo path, and `.env`.
`run_wsl.sh auth` performs a real WorldQuant Brain login check before you spend time on a run.
`run_wsl.sh run` is now a guarded one-round `light` cycle that checks auth, runs one safe research round, refreshes artifacts, and writes `artifacts/trang_thai_chay.json`.

If you only want local proxy scoring and do not want formulas or credentials sent to WorldQuant, use:

```bash
bash run_wsl.sh offline
```

This runs `main.py` with `--scoring internal`, skips authentication entirely, and scores alphas with a local heuristic model based on syntax validity, proxy IC, proxy Sharpe, turnover, capacity, stability, and uniqueness.

To score one expression directly without starting a full run:

```bash
python3 scripts/internal_score.py --expression "rank(ts_zscore(abs(close-vwap),21))"
```

To score with explicit WorldQuant-style settings:

```bash
python3 scripts/internal_score.py \
  --expression "rank(ts_zscore(abs(close-vwap),21))" \
  --settings "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market"
```

When `simulation_results.csv` has enough real Brain rows, local scoring now also shows a `surrogate_shadow` preview trained from those historical outcomes. This stays in shadow mode for now, so it does not replace the heuristic verdict or alpha score.

To score a local batch file:

```bash
python3 scripts/internal_score.py --file artifacts/bieu_thuc_ung_vien.txt
```

For a dedicated WSL-only walkthrough, see [WSL_RUN_GUIDE.md](/mnt/d/skill-creator/brain-learn-main/docs/WSL_RUN_GUIDE.md).

## Run Modes

`main.py` now supports explicit run profiles:

- `test`: very small smoke-test run for pipeline checks
- `careful`: conservative daily profile with `n_parallel=1` and lower submission pressure
- `smart`: faster balanced daily profile with capped seed loading and smarter seed selection
- `light`: recommended daily profile and the default when no mode is passed
- `full`: heavy search profile for longer runs and bigger quota budgets

If you want `full` to automatically push parallelism close to your machine limit while still leaving some headroom, add:

```bash
python3 main.py --mode full --auto-parallel --cpu-headroom 2 --memory-headroom-gb 2
```

Examples:

```bash
python3 main.py --mode test
python3 main.py --mode careful
python3 main.py --mode smart
python3 main.py --mode light
python3 main.py --mode full
```

The helper scripts mirror those profiles:

```bash
bash run_wsl.sh test
bash run_wsl.sh careful
bash run_wsl.sh smart
bash run_wsl.sh light
bash run_wsl.sh full
bash run_wsl.sh max
bash run_wsl.sh turbo
```

`max` and `turbo` are helper aliases for `full --auto-parallel`, so they try to use most of your CPU/RAM without pushing the machine into obvious overload. On a machine with more cores, this can raise `n_parallel` well above the old fixed cap.

For a local-only run that never contacts WorldQuant:

```bash
python3 main.py --mode light --scoring internal
```

For a near-max local run that auto-tunes worker count:

```bash
python3 main.py --mode full --scoring internal --auto-parallel --cpu-headroom 2 --memory-headroom-gb 2
```

For the simplest orchestrated workflow, use:

```bash
bash run_wsl.sh light
bash run_wsl.sh daily
bash run_wsl.sh feed
```

To keep that orchestrated workflow running continuously until you stop it, use:

```bash
bash run_wsl.sh loop --scoring internal --interval-minutes 30
```

That command runs one orchestrated round every 30 minutes, refreshes `artifacts/latest/...` after each round, and keeps going until you stop it with `Ctrl+C` or:

```bash
bash run_wsl.sh loop-stop
```

To inspect the current loop state:

```bash
bash run_wsl.sh loop-status
```

Operational guardrails in the current orchestrated pipeline:

- `artifacts/recent_runs/<run_id>/...` is the canonical per-run workspace. `artifacts/runs/...` is now legacy and only migrated for compatibility.
- `artifacts/latest/...` is only considered trustworthy when `artifacts/latest/latest_metadata.json` says the publish is complete.
- JSON artifacts now carry `schema_version`, are validated on read, and can be migrated from simple legacy payloads.
- Broken JSON files, bad JSONL rows, and malformed candidates are isolated into `artifacts/quarantine/...` instead of flowing deeper into the pipeline.
- Candidate merge now applies lineage-aware dedupe, recent-failure penalties, and soft source quotas across planner, fix, and scout inputs.
- Loop mode now emits graded health alerts (`warning`, `error`, `critical`) for memory size, repeated API failures, stale latest metadata, and long submit-ready droughts.
- `careful`, `smart`, `light`, `run`, `cycle`, `full`, `max`, and `turbo` all run through `scripts/orchestrator.py`, which writes a full run-scoped pipeline into `artifacts/recent_runs/<run_id>/...` and refreshes `artifacts/latest/...`.
- `feed`: prints the full markdown feed in `artifacts/latest/bang_tin_alpha.md`.
- `daily`: prints up to 3 best submit-ready candidates in `artifacts/latest/alpha_tot_nhat_hom_nay.md`. This is the canonical short list to review.
- `cleanup`: applies retention policy to old run folders, migrates legacy `artifacts/runs/...` into `artifacts/recent_runs/...`, gzips old `.jsonl/.csv`, and deletes stale archives.
- `scout`: uses public APIs to find factor ideas, turns them into local alpha candidates, scores them locally, and returns up to your requested number of daily picks without contacting WorldQuant.
- `fix`: diagnoses one alpha and explains how to repair it from `alpha_id` or explicit failing checks. Add `--auto-rewrite` to locally rescore rewrite candidates and surface the closest submit-ready repairs.
- `score`: scores one expression or a batch locally and returns optimization hints without contacting WorldQuant.

For a wider review-first workflow, use:

```bash
bash run_wsl.sh review
bash run_wsl.sh plan
```

- `review`: refreshes digest, planning artifacts, and `artifacts/duyet_tay.md` for manual inspection.
- `plan`: rebuilds the next-batch artifacts without starting a new run.
- `cycle`: one guarded `light` round with stop conditions for auth failures, rate limits, pending backlog, or no improvement.
- `seed`: approves the top planner-ready candidates into `initial-population.pkl`.
- `seed-submit-ready`: seeds exactly the up-to-3 submit-ready candidates currently shown in `artifacts/latest/alpha_tot_nhat_hom_nay.md`, and also writes a run snapshot to `artifacts/recent_runs/<run_id>/seed_population.pkl`.

Run cleanup manually or from cron if you want a dedicated retention step:

```bash
bash run_wsl.sh cleanup
```

To diagnose a failing alpha:

```bash
bash run_wsl.sh fix --alpha-id YOUR_ALPHA_ID
```

or

```bash
bash run_wsl.sh fix --expression "rank(...)" --errors LOW_SHARPE LOW_FITNESS
```

To score an expression locally and get optimization hints:

```bash
bash run_wsl.sh score --expression "rank(ts_zscore(abs(close-vwap),21))"
```

If you need to intervene manually during a bad streak, the orchestrator and loop both support:

```bash
bash run_wsl.sh light --manual-only-fix --manual-freeze-memory-update
```

```bash
bash run_wsl.sh loop \
  --scoring internal \
  --manual-disable-scout \
  --manual-increase-explore \
  --interval-minutes 30
```

You can also temporarily ignore planner hard/soft block lists:

```bash
bash run_wsl.sh light --manual-ignore-block-list
```

To let the repo scout public thesis ideas and produce up to 6 local alpha picks for the day:

```bash
bash run_wsl.sh scout --count 6
```

This uses public APIs such as OpenAlex, arXiv, and GitHub for idea discovery, can optionally learn from a local `worldquant-miner` ZIP plus public GitHub READMEs, mixes in distilled local idea templates from `Finding Alphas`, converts those ideas into supported token-program candidates, scores them locally, writes `artifacts/trinh_sat_hang_ngay.md`, and never sends the generated alpha expressions to WorldQuant. The selector is `quality-first`, so `--count` is a maximum, not a promise to fill every slot.
The final selector is now diversity-aware, and if strict picks are too sparse it may promote a small number of clearly labeled relaxed exploration picks.
The markdown report now surfaces Brain feedback load status explicitly, but only expands `Ready To Submit` for strict picks that also clear a higher report gate.
By default, scout keeps running normally but only publishes one report every 60 minutes, and it skips creating a new archive copy when no pick clears that report gate.
That published report is now score-first: it aims for roughly your requested `--count` best alphas, but it can include extra overflow names when the next-ranked candidates are still close enough in score to be genuinely worth reviewing.
Brain feedback penalties are now context-aware when `simulation_results.csv` includes settings columns such as `region`, `universe`, `delay`, `decay`, `neutralization`, and `truncation`; older CSV files still work and fall back to softer global penalties.
When enough historical Brain rows are available, each candidate block also shows a `Surrogate shadow` line with the gradient-boosting preview trained from your own result history.
That surrogate shadow now also downranks candidates when it strongly disagrees with the heuristic scorer, and can block severe heuristic-vs-history mismatches from the strict daily picks.
Softer surrogate disagreements are not dropped completely; when they stay structurally safe, scout can keep them in `Watchlist` so you still see them as verify-first ideas.

Useful scout flags:

```bash
bash run_wsl.sh scout \
  --count 6 \
  --search-breadth wide \
  --min-alpha-score 68 \
  --min-confidence-score 0.58 \
  --report-min-alpha-score 76 \
  --report-min-confidence-score 0.64 \
  --report-min-robustness-score 0.62 \
  --report-interval-minutes 60 \
  --archive-frequency hour \
  --github-per-query 1 \
  --github-query-limit 6 \
  --github-readme-limit 3 \
  --learn-zip-path /mnt/c/Users/OS/Downloads/worldquant-miner-master.zip \
  --zip-seed-limit 6 \
  --diversity-weight 0.35 \
  --include-watchlist
```

For a broader overnight search that deliberately explores more papers/repos and more query families, use:

```bash
bash run_wsl.sh scout \
  --count 6 \
  --search-breadth explore \
  --github-per-query 2 \
  --github-query-limit 10 \
  --github-readme-limit 6 \
  --include-watchlist
```

If you want scout to keep running until it finds at least one `Ready To Submit` strict pick, use:

```bash
bash run_wsl.sh scout-loop \
  --poll-seconds 900 \
  --clear-stop-file \
  --count 4 \
  --search-breadth focused \
  --github-per-query 2 \
  --github-query-limit 6 \
  --github-readme-limit 3 \
  --learn-zip-path /mnt/c/Users/OS/Downloads/worldquant-miner-master.zip \
  --zip-seed-limit 6 \
  --include-watchlist
```

That loop still runs every `--poll-seconds`, but the published markdown report and archived report snapshot are rate-limited separately by `--report-interval-minutes` in `scout`.

`scout-loop` now forces a Brain feedback health gate by default. If `simulation_results.csv` is missing, malformed, or still missing settings context columns such as `region/universe/delay/decay`, the loop halts early with exit code `2` instead of continuing with degraded learning. Only bypass this intentionally with:

```bash
bash run_wsl.sh scout-loop --allow-degraded-feedback ...
```

If you do not want the loop to auto-stop even when strict picks appear, use:

```bash
bash run_wsl.sh scout-loop --manual-stop-only ...
```

To stop that long-running loop gracefully, either press `Ctrl+C` or run:

```bash
bash run_wsl.sh scout-stop
```

Scout artifacts now include:

- `artifacts/trinh_sat_hang_ngay.md`: top picks, optional watchlist, and explicit held-out reasons.
- `artifacts/bao_cao_ngay/YYYY-MM-DD/HHh/`: hourly archived copy of reportable scout runs. Use `--archive-frequency run` if you explicitly want one folder per run.
- `artifacts/_trinh_sat/du_lieu.json`: machine-readable daily payload.
- `artifacts/_trinh_sat/bieu_thuc_da_chon.txt`: flat list of the reportable expressions that cleared the report gate.
- `artifacts/_trinh_sat/lich_su.jsonl`: per-candidate archive across runs.
- `artifacts/_trinh_sat/bo_nho.json`: aggregated scout-specific learning used to bias future runs.
- `artifacts/_trinh_sat/phan_hoi_brain.json`: summarized real Brain feedback learned from `simulation_results.csv`, used to penalize weak families and exact bad skeletons.
- `artifacts/_trinh_sat/kien_thuc_tu_hoc.json`: summary of what scout learned from local ZIP seeds and GitHub READMEs.
- `artifacts/_trinh_sat/cache_tim_kiem.json`: cached public-source fetches that make repeated scout runs much faster.
- `artifacts/alpha_da_gui.json`: your submitted-alpha library used to reduce overlap with new scout picks.

You can also pass settings:

```bash
bash run_wsl.sh score \
  --expression "rank(ts_zscore(abs(close-vwap),21))" \
  --settings "USA, TOP200, Decay 3, Delay 1, Truncation 0.01, Neutralization Subindustry"
```

Execute the main script using `uv`:

```bash
uv run main.py
```

or activate the environment first then run with the corresponding interpreter

```bash
source .venv/bin/activate
python3 main.py
```

This will start the genetic programming process based on the configurations in `main.py`.

## Research Workflow

Use the repo in a tight research loop instead of treating one long run as the whole process:

1. Run the generator with the safer default:
   ```bash
   bash run_wsl.sh smart
   ```
2. Refresh the report and manual review queue:
   ```bash
   bash run_wsl.sh feed
   ```
3. Inspect:
   ```bash
   cat artifacts/latest/bang_tin_alpha.md
   ```
4. Only if you explicitly want to update the local seed pool:
   ```bash
   bash run_wsl.sh seed
   ```
5. Lint a batch of candidate expressions before submitting or seeding:
   ```bash
   python3 scripts/expression_lint.py --file artifacts/bieu_thuc_ung_vien.txt --format markdown
   ```

The repo now keeps `simulation_results.csv` in a standard headered format and can still read the older `simulations.csv` layout.

You can also digest results directly through the helper scripts:

```bash
bash run_wsl.sh digest
```

For orchestrated runs (`light`, `run`, `cycle`, `careful`, `smart`, `full`, `max`, `turbo`, `loop`), the canonical publish targets are:

- `artifacts/latest/latest_metadata.json`
- `artifacts/latest/alpha_tot_nhat_hom_nay.md`
- `artifacts/latest/bang_tin_alpha.md`
- `artifacts/latest/evaluated_candidates.json`
- `artifacts/latest/results_summary.json`
- `artifacts/latest/orchestrator_summary.json`
- `artifacts/state/global_research_memory.json`

The root-level files in `artifacts/` are still refreshed as compatibility mirrors for quick review, but `latest/` and `recent_runs/` are the primary sources.

For the least cluttered review flow, open `artifacts/latest/alpha_tot_nhat_hom_nay.md` first. If a run looks suspicious or incomplete, check `artifacts/latest/latest_metadata.json` and then inspect the matching folder under `artifacts/recent_runs/<run_id>/`.

## Included Research Tools

- `scripts/results_digest.py`: ranks recent simulation rows, summarizes the dominant failure checks, and highlights near-duplicate expression skeletons.
- `scripts/expression_lint.py`: detects exact duplicates, near-duplicates, and low-diversity expression batches before you waste simulation budget.
- `scripts/doctor.py`: checks whether your current shell, Python version, dependencies, and `.env` are ready for a real run.
- `scripts/plan_next_batch.py`: turns prior simulation results into a structured next batch with novelty scores, risk tags, `seed_ready`, and `token_program`.
- `scripts/approve_seeds.py`: validates planner-ready candidates, merges them into `initial-population.pkl`, and writes a latest summary to `artifacts/seed_approval.md` by default.
- `scripts/seed_submit_ready.py`: seeds the current submit-ready report shortlist and writes a latest summary to `artifacts/seed_submit_ready.md`.
- `scripts/research_cycle.py`: runs guarded one-round or two-round research cycles and writes `artifacts/trang_thai_chay.json`.
- `scripts/render_cycle_report.py`: builds a compact markdown report focused on the top submit-ready daily alphas.
- `scripts/runtime_control.py`: writes stop files safely and clears stale `*.lock` files under runtime folders.

## Customization

You can customize the building blocks of the genetic programming process:

*   **Operators and Terminals:** Add or modify operators (functions like `add`, `ts_rank`) and terminals (input features like `close`, `volume`) in `src/function.py`.
*   **Weights:** Adjust the `weight` parameter for `Operator` and `Terminal` instances in `src/function.py` to influence their probability of being selected during the evolutionary process. Higher weights mean higher probability.

## Notes

- Duplicate avoidance is now separate from fitness evaluation tracking, so newly generated expressions are not skipped before they are scored.
- Division now uses a correct unit rule during validation.
- Offline or test metrics that do not require a WorldQuant session can run without forcing a reconnect attempt.
- Authentication now retries on transient network errors and `429` rate limits instead of failing immediately.
- `main.py` defaults to the safer `light` profile when no `--mode` is passed.
- Planner memory now ignores `PENDING` rows for block decisions and only blocks repeated serious failures or direct competition matches.
- Seed files now store token-name programs so review-seeded entries are stable to pickle and can be materialized back into `Program` objects on load.
- Artifact JSON readers now validate `schema_version`, can migrate simple legacy payloads, and quarantine unreadable files instead of failing silently.
- Candidate merge now tracks deeper lineage, filters exact and near-duplicate candidates before evaluation, and balances queue share across planner, fix, and scout.
- Local scoring is now batched, can use multiprocessing, and supports `--local-score-limit`, `--local-score-workers`, and `--min-parallel-local-scoring`.
- Loop mode now supports manual override flags and health thresholds, and cleanup tracks artifact footprint plus recent-run retention.

## Disclaimer

Notice: This codebase is experimental and intended solely for personal use. It is provided 'AS IS', without representation or warranty of any kind. Liability for any use or reliance upon this software is expressly disclaimed.
