# Quy Trinh Tiep Theo

## Trang Thai Hien Tai (2026-04-02)

| Thong so | Gia tri |
|---|---|
| `run_id` | `light_20260402_205642` |
| `state` | `completed` |
| `profile` | `light` |
| `scoring` | `internal` |
| `queue_candidates` | 3 |
| `queue_source_counts` | `planner=1`, `scout=2`, `auto_fix_rewrite=0` |
| `filtered_queue_candidates` | `not_seed_ready=3`, `surrogate_shadow_fail=7` |
| `exploratory_queue` | `active=true`, `used=true`, `mode=strict_queue_empty`, `count=3` |
| `exploratory_selected_reasons` | `not_seed_ready=2`, `surrogate_shadow_fail=1` |
| `evaluated_candidates` | 3 |
| `submit_ready_candidates` | 0 |
| `auto_fix_contexts_processed` | 2 |
| `manual_overrides` | `allow_exploratory_queue=true`, `exploratory_queue_limit=3` |
| `latest_publish_status` | `complete` |

> **Ket luan ngan**: He da thoat trang thai queue chet. Truoc do queue ve 0; hien tai da co 3 candidate duoc dua vao hang evaluate qua exploratory fallback, trong do co 1 candidate planner duoc day sang lane `surrogate_verify_first`.

---

## He Da Cai Thien Gi

- Planner da nghieng manh hon sang thesis `simple_price_patterns` va `residual_beta` de giam `SELF_CORRELATION` / `MATCHES_COMPETITION`.
- Orchestrator da co auto-fix synthesis loop tu near-miss contexts thay vi chi cho file fix san co.
- Submit gate da mo them lane `exploratory verify-first` cho candidate `surrogate_shadow_fail` neu metrics du kha quan.
- Manual exploratory queue hien hoat dong duoc ca voi `internal` scoring, khong chi `worldquant`.

## Muc Tieu Luc Nay

- Khong con tap trung vao viec "lam sao de queue co candidate".
- Tap trung vao viec day candidate moi vao trang thai "co the nop thu":
  - `seed_ready=true`
  - khong bi loai som vi `surrogate_shadow_fail`
  - neu chua qua strict gate thi it nhat phai vao duoc exploratory queue de verify-first

## File Can Mo Truoc

- `artifacts/latest/latest_metadata.json`
  Chi tin `artifacts/latest/` khi `status=complete` va `complete=true`.

- `artifacts/latest/orchestrator_summary.json`
  File tong quan nhanh nhat de xem queue co song hay khong, exploratory co duoc dung hay khong.

- `artifacts/latest/planned_candidates.json`
  Xem batch planner moi nhat dang nghi ve family nao.

- `artifacts/latest/alpha_tot_nhat_hom_nay.md`
  Shortlist strict submit-ready. Hien tai van la `0`.

- `artifacts/latest/bang_tin_alpha.md`
  Full feed sau evaluate.

- `artifacts/recent_runs/<run_id>/pending_simulation_queue.json`
  File quan trong nhat de xem candidate nao duoc promote vao queue va vi sao.

- `artifacts/recent_runs/<run_id>/evaluated_candidates.json`
  Xem ket qua thuc te sau evaluate.

- `artifacts/recent_runs/<run_id>/results_summary.json`
  Xem fail pattern tong hop cua round.

## Cach Doc Nhanh Round Moi Nhat

Thu tu uu tien:

1. `artifacts/latest/latest_metadata.json`
2. `artifacts/latest/orchestrator_summary.json`
3. `artifacts/recent_runs/<run_id>/pending_simulation_queue.json`
4. `artifacts/recent_runs/<run_id>/evaluated_candidates.json`
5. `artifacts/latest/planned_candidates.json`
6. `artifacts/latest/alpha_tot_nhat_hom_nay.md`
7. `artifacts/latest/bang_tin_alpha.md`

Doc `orchestrator_summary.json` truoc, roi dien giai nhanh:

- `queue_candidates > 0`
  Pipeline dang co dau ra de evaluate.

- `submit_ready_candidates = 0`
  Chua co alpha dat gate cuoi, nhung khong co nghia la round vo ich.

- `exploratory_queue_used = true`
  Strict queue da rong hoac qua thua, he da chu dong dua ung vien can nguong vao lane tham do.

- `exploratory_selected_reasons.surrogate_shadow_fail > 0`
  Da co candidate bi surrogate chan nhung van duoc verify-first thay vi bi bo di.

## Candidate Quan Trong Nhat O Run Hien Tai

Trong `artifacts/recent_runs/light_20260402_205642/pending_simulation_queue.json`, 3 candidate da vao queue:

1. `scout`, reason `not_seed_ready`
2. `scout`, reason `not_seed_ready`
3. `planner`, reason `surrogate_shadow_fail`, co `surrogate_verify_first=true`

Candidate planner verify-first hien tai:

```txt
winsorize(rank(multiply((close/ts_delay(close, 3)-1),divide(volume,ts_mean(volume, 63)))),std=5)
```

Y nghia:

- Candidate nay du kha on ve local metrics de duoc xem tiep.
- No chua qua strict gate vi surrogate shadow qua chat.
- He hien da biet day no vao exploratory lane thay vi bo di ngay.

## Chuan Bi Moi Truong

Chay tu thu muc repo:

```bash
cd /mnt/d/skill-creator/brain-learn-main
```

Neu muon kich hoat virtualenv:

```bash
source .venv/Scripts/activate
```

Neu can auth WorldQuant, su dung:

```env
WORLDQUANT_USERNAME="your_worldquant_brain_username"
WORLDQUANT_PASSWORD="your_worldquant_brain_password"
```

## Lenh Chay Khuyen Nghi

### 1. Kiem tra he thong

```bash
bash ./run_wsl.sh doctor
```

Neu can check auth:

```bash
bash ./run_wsl.sh auth
```

### 2. Chay local co exploratory queue

Day la lenh da cho ra run moi nhat:

```bash
python3 scripts/orchestrator.py \
  --profile light \
  --scoring internal \
  --count 8 \
  --queue-limit 10 \
  --top 10 \
  --history-window 120 \
  --manual-allow-exploratory-queue \
  --manual-exploratory-queue-limit 3
```

### 3. Chay local nhanh qua wrapper

```bash
bash ./run_wsl.sh internal
```

Lenh nay hop de xem nhanh he co tiep tuc sinh batch duoc khong, nhung khong ep exploratory verify-first nhu lenh tren.

### 4. Chay WorldQuant de nop thu can nguong

Chi bat exploratory limit nho, uu tien `2` hoac `3`:

```bash
bash ./run_wsl.sh light \
  --scoring worldquant \
  --manual-allow-exploratory-queue \
  --manual-exploratory-queue-limit 2
```

### 5. Xem va dung loop

```bash
bash ./run_wsl.sh loop-status
```

```bash
bash ./run_wsl.sh loop-stop
```

## Cach Doc Ket Qua Sau Khi Chay

### Neu `queue_candidates = 0`

- Mo `filtered_queue_candidates` trong `orchestrator_summary.json`.
- Neu thay chu yeu la `not_seed_ready`, uu tien sua seedability va confidence.
- Neu thay chu yeu la `surrogate_shadow_fail`, uu tien chay exploratory verify-first thay vi tiep tuc siet gate.

### Neu `queue_candidates > 0` nhung `submit_ready_candidates = 0`

Day la trang thai hien tai, va van la tien trien tot hon truoc.

Can xem:

- `pending_simulation_queue.json`
  Candidate nao duoc promote.

- `results_summary.json`
  Failure nao dang lap lai sau evaluate.

- `evaluated_candidates.json`
  Candidate nao gan nguong nhat de rewrite them.

### Neu `exploratory_queue_used = true`

Khong xem do la fail. Day la co che "cau noi" de tranh lam mat alpha tiem nang.

## Tieu Chi Phan Loai Candidate

### Strict submit-ready

Chi duoc xem la san sang seed khi:

- `quality_label = qualified`
- `seed_ready = true`
- `confidence_score >= 0.45`
- `verdict = PASS` hoac `LIKELY_PASS`
- `alpha_score >= 65`
- `sharpe >= 1.4`
- `fitness >= 1.0`

### Exploratory verify-first

Dung cho candidate chua qua strict gate nhung van dang de thu tiep:

- co local metrics kha on
- surrogate shadow fail nhung khong qua te
- hoac chua seed-ready nhung sat nguong

Muc tieu cua lane nay la "dua vao hang nop thu co kiem soat", khong phai tu dong coi la dat.

## Uu Tien Xu Ly Tiep Theo

1. Doc `results_summary.json` cua run moi nhat de xem failure lap lai nhieu nhat sau exploratory evaluate.
2. Tap trung vao candidate planner verify-first da duoc day vao queue, vi day la ung vien gan strict gate nhat hien tai.
3. Giam `SELF_CORRELATION` va `MATCHES_COMPETITION` ma khong lam roi `sharpe` / `fitness`.
4. Bien 2 scout candidate `not_seed_ready` thanh `seed_ready` thay vi de chung luon nam o watchlist.
5. Giu exploratory queue limit nho (`2-3`) de tranh lam loang batch.

## Khi Nao Seed

Chi seed candidate da qua evaluate va vuot strict submit-ready gate:

```bash
bash ./run_wsl.sh seed-submit-ready
```

Khong seed candidate chi vi no da vao exploratory queue.

## Don File Khong Can Thiet

Lenh don stale lock:

```bash
python3 scripts/runtime_control.py clear-stale-locks artifacts
```

Neu can don them theo retention policy:

```bash
bash ./run_wsl.sh cleanup
```

## Quy Tac Nho

- Khong ket luan tu `artifacts/latest/` khi publish chua complete.
- `submit_ready_candidates = 0` khong co nghia la run do vo gia tri.
- Neu `exploratory_queue_used = true`, phai doc them `pending_simulation_queue.json`.
- Uu tien candidate da duoc promote vao `surrogate_verify_first` truoc khi mo rong them thesis moi.
- Khong seed candidate chi vi no "co ve hay"; chi seed khi no qua strict gate that.
