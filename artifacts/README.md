# Artifacts Guide

Mo file nay truoc neu ban muon xem nhanh he thong vua sinh ra gi.

## Canonical latest outputs

- `latest/latest_metadata.json`: marker publish moi nhat. Chi tin `latest/` khi file nay co `status=complete`.
- `latest/alpha_tot_nhat_hom_nay.md`: shortlist submit-ready de doc nhanh.
- `latest/bang_tin_alpha.md`: full feed sau evaluate.
- `latest/evaluated_candidates.json`: pool chuan cho `daily`, `feed`, `seed-submit-ready`.
- `latest/results_summary.json`: tom tat queue da evaluate.
- `latest/orchestrator_summary.json`: summary mot vong orchestrator moi nhat.

## Run-scoped outputs

- `recent_runs/<run_id>/`: artifact day du cua tung vong orchestrated gan day.
- `archive/<run_id>/`: run cu da duoc dua vao kho luu tru.
- Thu muc `recent_runs/` la noi dung chinh hien tai. `runs/` cu chi con duoc migrate de tuong thich.

## State va runtime

- `state/global_research_memory.json`: bo nho nghien cuu dung chung qua nhieu vong.
- `state/orchestrator_loop_status.json`: trang thai loop moi nhat, gom `run_id`, round, health, summary.
- `state/artifact_cleanup_status.json`: ket qua cleanup, footprint, so run duoc giu/archive.
- `state/DUNG_LOOP`: file stop cho orchestrator loop.

## Scout outputs

- `trinh_sat_hang_ngay.md`: bao cao scout moi nhat de doc nhanh.
- `_trinh_sat/du_lieu.json`: payload scout day du cho may doc.
- `_trinh_sat/bieu_thuc_da_chon.txt`: expression da duoc report.
- `_trinh_sat/lich_su.jsonl`: lich su candidate scout qua nhieu lan chay.
- `_trinh_sat/bo_nho.json`: bo nho rieng cua scout.
- `_trinh_sat/phan_hoi_brain.json`: feedback rut ra tu `simulation_results.csv`.
- `_trinh_sat/DUNG`: file stop cho `scout-loop`.
- `bao_cao_ngay/YYYY-MM-DD/HHh_MMp/`: ban luu report scout theo ngay/gio.

## Quarantine va schema safety

- Artifact JSON quan trong deu co `schema_version`.
- Reader se validate schema co ban, migrate legacy payload don gian, hoac tu choi doc neu version moi hon reader.
- File JSON hong, JSONL line hong, va candidate ban se duoc dua vao `quarantine/` thay vi di tiep vao flow chinh.
- `quarantine/` chia theo nhom `files/`, `jsonl_lines/`, `candidates/` de debug de hon.

## Retention va dọn dẹp

- `scripts/cleanup_artifacts.py` giu `recent_runs` trong cua so retention ngan, archive run cu, nen file lon, va theo doi footprint.
- Mặc định repo giu toi da `24` run gan nhat trong `recent_runs/` truoc khi archive.
- `scripts/runtime_control.py clear-stale-locks artifacts` dung de xoa `.lock` cu mot cach an toan.

## Ghi chu

- Thu muc goc `artifacts/` van giu mot so file mirror legacy de review nhanh, nhung `latest/` va `recent_runs/` moi la nguon chuan.
- Neu can debug loi publish, kiem tra `latest/latest_metadata.json`, `recent_runs/<run_id>/orchestrator_summary.json`, va `quarantine/` truoc.
