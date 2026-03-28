# WSL Run Guide

Huong dan nay tap trung vao cach chay `brain-learn-main` bang WSL khi repo nam o duong dan kieu `/mnt/d/...`.

Khong chay `bash run_wsl.sh ...` trong:

- Git Bash
- PowerShell
- Command Prompt

## Quy trinh lan dau

Mo WSL roi chay:

```bash
cd /mnt/d/skill-creator/brain-learn-main
uv sync
source .venv/bin/activate
cp -n .env.example .env
```

Sua `.env` va dien tai khoan WorldQuant BRAIN:

```env
USERNAME="your_worldquant_brain_username"
PASSWORD="your_worldquant_brain_password"
```

Moi lan mo terminal moi:

```bash
cd /mnt/d/skill-creator/brain-learn-main
source .venv/bin/activate
```

Kiem tra moi truong:

```bash
bash run_wsl.sh doctor
bash run_wsl.sh auth
```

Neu chi muon cham local va khong gui expression ra WorldQuant:

```bash
bash run_wsl.sh offline
```

## Lenh khuyen nghi

Chay mot vong orchestrator an toan:

```bash
bash run_wsl.sh light
```

Chay local-only:

```bash
bash run_wsl.sh internal
```

Xem shortlist va feed moi nhat:

```bash
bash run_wsl.sh daily
bash run_wsl.sh feed
```

Chay loop lien tuc den khi ban chu dong dung:

```bash
bash run_wsl.sh loop --scoring internal --interval-minutes 30
```

Dung loop:

```bash
bash run_wsl.sh loop-stop
```

Xem trang thai loop:

```bash
bash run_wsl.sh loop-status
```

## Manual override khi debug hoac cuu he

Chi danh gia candidate tu auto-fix va khong ghi de memory:

```bash
bash run_wsl.sh light --manual-only-fix --manual-freeze-memory-update
```

Tat scout tam thoi va tang explore cho planner:

```bash
bash run_wsl.sh loop \
  --scoring internal \
  --manual-disable-scout \
  --manual-increase-explore
```

Bo qua tam thoi block list cua planner:

```bash
bash run_wsl.sh light --manual-ignore-block-list
```

## Local scoring va CPU

Backend `internal` hien tai co batch scoring va co the dung multiprocessing.
Neu muon cap worker hoac han muc local score:

```bash
bash run_wsl.sh light \
  --scoring internal \
  --local-score-limit 12 \
  --local-score-workers 4 \
  --min-parallel-local-scoring 4
```

## Health check va canh bao

`loop` hien tai co health check dinh ky:

- shared/latest file co doc duoc khong
- memory artifact co qua to khong
- API co loi lien tiep khong
- da bao lau chua co submit-ready alpha

Canh bao duoc phan cap:

- `warning`
- `error`
- `critical`

Thong tin nay duoc ghi vao:

- `artifacts/state/orchestrator_loop_status.json`
- `logs/system.log`
- `logs/error.log`

Ban co the tinh chinh threshold:

```bash
bash run_wsl.sh loop \
  --health-memory-warning-mb 4 \
  --health-api-warning-streak 2 \
  --health-no-pass-warning-rounds 3
```

## Output nam o dau

Nguon chuan can doc truoc:

- `artifacts/latest/latest_metadata.json`: marker publish complete cua `latest/`
- `artifacts/latest/alpha_tot_nhat_hom_nay.md`: shortlist submit-ready
- `artifacts/latest/bang_tin_alpha.md`: full feed
- `artifacts/latest/evaluated_candidates.json`: evaluated pool
- `artifacts/latest/results_summary.json`: tom tat ket qua evaluate
- `artifacts/latest/orchestrator_summary.json`: summary vong moi nhat

Artifact theo tung run:

- `artifacts/recent_runs/<run_id>/`: artifact day du theo vong
- `artifacts/archive/<run_id>/`: run cu da archive

Artifact runtime:

- `artifacts/state/global_research_memory.json`
- `artifacts/state/orchestrator_loop_status.json`
- `artifacts/state/artifact_cleanup_status.json`
- `artifacts/quarantine/`: file/candidate/jsonl loi bi cach ly

Mirror legacy o goc `artifacts/` van con duoc giu cho de review nhanh, nhung `latest/` va `recent_runs/` moi la nguon chuan.

## Cac lenh hay dung

- `doctor`: kiem tra shell, Python, dependency, `.env`
- `auth`: kiem tra dang nhap that voi WorldQuant BRAIN
- `light`, `run`, `cycle`: 1 vong orchestrator profile `light`
- `careful`, `smart`, `full`, `max`, `turbo`: profile khac nhau cua orchestrator
- `internal`, `offline`: chay orchestrator voi backend local
- `loop`: lap lai orchestrator den khi stop
- `loop-stop`, `loop-status`: dung/xem loop
- `scout`: tim y tuong cong khai va cham local
- `scout-loop`, `scout-stop`: loop rieng cho scout
- `fix`: phan tich alpha fail va goi y rewrite
- `score`: cham nhanh 1 expression hay 1 file candidate
- `seed`, `seed-submit-ready`: dua candidate tot vao seed store
- `cleanup`: dọn run cu, archive, temp file, stale retention

## Loi thuong gap

- `401`, `403`, `auth_failed`: kiem tra `USERNAME`/`PASSWORD` trong `.env`
- `429`, `rate_limited`: giam tan suat, doi them, uu tien `internal` khi debug
- `artifacts/latest is not marked complete yet`: kiem tra `artifacts/latest/latest_metadata.json`
- `schema_version` hoac `invalid_json`: vao `artifacts/quarantine/` de xem file nao bi cach ly
- `Missing Python modules`: quen `source .venv/bin/activate` hoac chua `uv sync`

## Don stale lock an toan

Neu nghi repo bi sot lai file `.lock` cu:

```bash
python3 scripts/runtime_control.py clear-stale-locks artifacts
```
