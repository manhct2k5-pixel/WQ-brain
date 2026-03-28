# Quy Trinh Tiep Theo

## Muc Tieu

- Chay pipeline co `run_id` ro rang.
- Xem output tu `artifacts/latest/` va debug tu `artifacts/recent_runs/<run_id>/`.
- Chi tin `latest/` khi publish da hoan tat.
- Neu loop bi "kho" candidate thi biet khi nao nen doi mode.

## File Can Mo Truoc

- `artifacts/latest/latest_metadata.json`
  marker publish moi nhat. Chi tin `latest/` khi `status=complete` va `complete=true`.

- `artifacts/latest/alpha_tot_nhat_hom_nay.md`
  shortlist submit-ready hien tai.

- `artifacts/latest/bang_tin_alpha.md`
  full feed sau evaluate.

- `artifacts/latest/evaluated_candidates.json`
  pool chuan cho `daily / feed / seed-submit-ready`.

- `artifacts/latest/orchestrator_summary.json`
  tom tat run moi nhat: `queue_candidates`, `submit_ready_candidates`, `adaptive_controls`, `manual_overrides`.

- `artifacts/state/orchestrator_loop_status.json`
  trang thai loop dang chay: `round_index`, `health`, `stagnation`, `run_id`.

- `artifacts/recent_runs/<run_id>/`
  artifact day du cua tung vong chay de debug.

## Cach Chay Dung

Luon chay tu thu muc repo:

```bash
cd /mnt/d/skill-creator/brain-learn-main
source .venv/bin/activate
```

### 1. Chay 1 vong an toan

Neu muon chay 1 vong local de xem nhanh:

```bash
bash ./run_wsl.sh internal
```

Neu muon chay 1 vong profile can than hon:

```bash
bash ./run_wsl.sh careful --scoring internal
```

### 2. Chay lien tuc, cham ma chac

Neu muon loop chay lien tuc khong ngu 30 phut:

```bash
bash ./run_wsl.sh loop \
  --profile careful \
  --scoring internal \
  --interval-minutes 0 \
  --local-score-limit 4 \
  --local-score-workers 1 \
  --min-parallel-local-scoring 4
```

Lenh nay phu hop khi uu tien:

- chay lien tuc
- local-only
- it dot CPU hon
- profile bao thu hon `light` / `turbo`

### 3. Ban an toan hon nua khi can co lap he

Neu nghi scout dang lam nhieu nhieu hon loi:

```bash
bash ./run_wsl.sh loop \
  --profile careful \
  --scoring internal \
  --interval-minutes 0 \
  --local-score-limit 4 \
  --local-score-workers 1 \
  --manual-disable-scout
```

Luu y:

- mode nay on de debug
- nhung neu chay qua lau ma `queue_candidates` van 0/1 thi he de bi stagnation vi mat mot nguon candidate

### 4. Xem trang thai va dung loop

```bash
bash ./run_wsl.sh loop-status
```

```bash
bash ./run_wsl.sh loop-stop
```

## Cach Doc Output

### Nhanh nhat

```bash
bash ./run_wsl.sh daily
bash ./run_wsl.sh feed
```

### Thu tu doc khuyen nghi

1. `artifacts/latest/latest_metadata.json`
   Neu file dang la `publishing`, doi them mot chut roi doc lai.

2. `artifacts/latest/orchestrator_summary.json`
   Xem:
   - `queue_candidates`
   - `submit_ready_candidates`
   - `adaptive_controls`
   - `manual_overrides`
   - `artifact_resource_guard`

3. `artifacts/latest/alpha_tot_nhat_hom_nay.md`
   Neu file nay `0` candidate, chuyen sang feed.

4. `artifacts/latest/bang_tin_alpha.md`
   Xem candidate dang o watchlist hay da co candidate dat gate.

5. `artifacts/recent_runs/<run_id>/`
   Khi can debug stage cu the.

## Luong Ben Trong

```text
run_wsl.sh
-> orchestrator.py
-> planned_candidates.json
-> pending_simulation_queue.json
-> simulation_results.jsonl
-> results_summary.json
-> evaluated_candidates.json
-> daily / feed / latest publish
-> cleanup resource guard
```

Moc output chinh theo run:

- `artifacts/recent_runs/<run_id>/planned_candidates.json`
- `artifacts/recent_runs/<run_id>/pending_simulation_queue.json`
- `artifacts/recent_runs/<run_id>/simulation_results.jsonl`
- `artifacts/recent_runs/<run_id>/results_summary.json`
- `artifacts/recent_runs/<run_id>/evaluated_candidates.json`
- `artifacts/recent_runs/<run_id>/orchestrator_summary.json`
- `artifacts/state/global_research_memory.json`

## Log Nao La Binh Thuong

- `Removed stale lock file ...`
  binh thuong. Loop dang don lock cua process cu da chet.

- `missing schema_version; treating as legacy JSON ...`
  binh thuong voi artifact cu. He dang migrate tam trong luc doc.

- `Skipping prior evaluated pool ... latest publish metadata is incomplete`
  co the xay ra o round dau sau mot lan dung giua chung. Neu chi xuat hien 1 lan roi mat thi khong sao.

- `adaptive_recovery: true`
  binh thuong khi nhieu round lien tiep khong co submit-ready.

## Dau Hieu He Dang Bi Stagnation

Cac dau hieu can chu y:

- `queue_candidates` thuong xuyen = 0 hoac 1
- `submit_ready_candidates` = 0 qua nhieu round
- `health.submit_ready_freshness` len `warning`, `error`, hoac `critical`
- `bang_tin_alpha.md` chi con watchlist, khong co candidate qua gate

Neu gap tinh trang nay trong nhieu round lien tuc:

1. dung mode qua co lap qua lau
2. bo `--manual-disable-scout` de mo lai nguon candidate
3. can nhac them `--manual-increase-explore`
4. xem `filtered_queue_candidates` trong `orchestrator_summary.json` de biet candidate bi rot vi ly do nao

## Khi Nao Nen Doi Mode

### Giu mode hien tai

Giu `careful + internal + interval 0` khi:

- muon chay dai hoi
- uu tien on dinh
- chap nhan toc do cham

### Mo rong hon mot chut

Neu sau 10-20 round van khong co submit-ready:

```bash
bash ./run_wsl.sh loop \
  --profile careful \
  --scoring internal \
  --interval-minutes 0 \
  --local-score-limit 6 \
  --local-score-workers 1 \
  --manual-increase-explore
```

### Mo lai scout khi pipeline qua kho

```bash
bash ./run_wsl.sh loop \
  --profile careful \
  --scoring internal \
  --interval-minutes 0 \
  --local-score-limit 4 \
  --local-score-workers 1
```

Tuc la bo han `--manual-disable-scout`.

## Khi Alpha Fail Nhung Co Trien Vong

Uu tien sua theo `expression`:

```bash
bash ./run_wsl.sh fix \
  --expression "rank(...)" \
  --errors LOW_SHARPE LOW_FITNESS \
  --auto-rewrite
```

Output thuong nam o:

- `artifacts/latest/auto_fix_candidates.json`
- `artifacts/recent_runs/<run_id>/auto_fix_candidates.json`

Neu file `artifacts/auto_fix_candidates.json` qua cu va warning `schema_version` lap lai, co the regen lai bang mot lan `fix --auto-rewrite` moi.

## Khi Nao Seed

Chi seed khi shortlist `latest` nhin that su on:

- `quality_label: qualified`
- `verdict: PASS` hoac `LIKELY_PASS`
- `alpha_score >= 65`
- `sharpe >= 1.4`
- `fitness >= 1.0`

Lenh:

```bash
bash ./run_wsl.sh seed-submit-ready
```

Lenh nay se:

- cap nhat `initial-population.pkl`
- ghi snapshot vao `artifacts/recent_runs/<run_id>/seed_population.pkl`

## Quy Tac Nho

- Khong ket luan tu `latest/` khi `latest_metadata.json` dang `publishing`.
- Khong seed candidate chi nam trong `Watchlist`.
- Khi nghi output co van de, mo `orchestrator_summary.json` truoc khi mo code.
- Khi nghi du lieu ban, vao `artifacts/quarantine/`.
- Khi log spam lock cu, co the don an toan bang:

```bash
python3 scripts/runtime_control.py clear-stale-locks artifacts
```
