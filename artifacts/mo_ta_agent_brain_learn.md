# Mô tả agent `brain-learn`

Ngày cập nhật: 2026-03-28

## 1. Mục tiêu của agent

`brain-learn` là một agent nghiên cứu alpha cho WorldQuant BRAIN.

Phiên bản hiện tại không còn là một script đơn lẻ, mà là một hệ thống nhiều lớp:

- planner sinh candidate mới từ lịch sử
- merger gộp candidate từ nhiều nguồn
- local scorer và evaluator chấm điểm
- builder dựng evaluated pool và report
- loop tự chạy nhiều vòng
- scout tìm ý tưởng công khai
- fix tạo rewrite cho alpha fail
- cleanup và health layer giữ runtime ổn định

Mục tiêu của hệ là:

- tạo candidate mới có kiểm soát
- không để dữ liệu bẩn làm nhiễu pipeline
- giảm lãng phí evaluate cho candidate trùng hoặc gần trùng
- giữ lineage đủ sâu để biết candidate đến từ đâu và fail vì sao
- cho phép chạy dài hơi nhưng vẫn có điểm can thiệp tay khi cần

## 2. Entry point chính

Người dùng thường đi qua:

```bash
bash run_wsl.sh <action>
```

Các action quan trọng:

- `light`, `run`, `cycle`, `careful`, `smart`, `full`, `max`, `turbo`
  gọi `scripts/orchestrator.py`

- `loop`
  gọi `scripts/orchestrator_loop.py`

- `scout`
  gọi `scripts/scout_ideas.py`

- `scout-loop`
  gọi `scripts/scout_loop.py`

- `fix`
  gọi `scripts/fix_alpha.py`

- `score`
  gọi `scripts/internal_score.py`

- `cleanup`
  gọi `scripts/cleanup_artifacts.py`

## 3. Hai backend chấm điểm

- `internal`
  chấm hoàn toàn local

- `worldquant`
  local-score trước, sau đó mới gọi WorldQuant cho bước evaluate chính

`internal` phù hợp cho debug, scout, loop local, hoặc khi cần tiết kiệm quota.

## 4. Kiến trúc pipeline orchestrator

Luồng chuẩn hiện tại:

```text
run_wsl.sh
-> doctor/auth guard
-> orchestrator.py
-> build memory
-> plan_next_batch
-> merge_candidate_pool
-> simulate_batch
-> build_evaluated_pool
-> update_research_memory
-> daily_best / alpha_feed
-> publish latest
-> cleanup resource guard
```

### 4.1 Runtime layout

Mỗi vòng chạy có `run_id` riêng.

Thư mục chuẩn cho mỗi run:

```text
artifacts/recent_runs/<run_id>/
```

Lưu ý:

- `artifacts/runs/<run_id>/` là layout cũ
- dữ liệu hiện tại dùng `recent_runs/`
- cleanup có thể migrate run cũ từ `runs/` sang `recent_runs/`

### 4.2 Checkpoint và publish safety

`orchestrator.py` có checkpoint theo stage để:

- resume khi crash giữa chừng
- tránh publish `latest/` nửa chừng
- xác nhận artifact đang đọc đúng `run_id` và `batch_id`

`artifacts/latest/latest_metadata.json` là marker publish chuẩn.
Chỉ nên tin `artifacts/latest/*` khi:

- `status = "complete"`
- `complete = true`

## 5. Planner và memory

`plan_next_batch.py` học từ lịch sử để build memory và batch mới.

Memory chính:

- `failure_counts`
- `blocked_skeletons`
- `blocked_families`
- `preferred_skeletons`
- `family_stats`
- `style_leaders`
- `seed_context`

Planner hiện tại không chỉ sinh expression thô, mà còn gắn metadata giúp bước sau ra quyết định tốt hơn:

- `candidate_id`
- `source`
- `family`
- `risk_tags`
- `candidate_score`
- `confidence_score`
- `novelty_score`
- `seed_ready`

## 6. Candidate lifecycle hiện tại

### 6.1 Nguồn candidate

Queue có thể nhận candidate từ:

- `planner`
- `scout`
- `fix`
- `seed`

### 6.2 Lineage sâu hơn

Mỗi candidate hiện có lineage đủ sâu để trace:

- nguồn sinh ra candidate
- `parent` candidate nếu là rewrite
- `hypothesis` hoặc `family`
- stage result theo từng chặng
- pass/fail reason ở planning và evaluation

Điều này giúp trả lời các câu hỏi kiểu:

- nguồn nào đang hiệu quả hơn
- strategy nào đáng giữ
- rewrite nào chỉ là biến thể của một candidate fail cũ

### 6.3 Dedupe và near-dedupe

Trước evaluate, hệ lọc trùng theo nhiều tầng:

- exact match
- normalized expression
- skeleton/signature gần giống

Candidate cũng được gắn signature như:

- `candidate_signature`
- `expression_signature`
- `skeleton_signature`
- `structure_signature`

Nếu candidate gần với candidate vừa fail gần đây, queue sẽ hạ ưu tiên thay vì đánh giá lại một cách mù quáng.

### 6.4 Soft quota giữa các nguồn

`merge_candidate_pool.py` không còn chỉ sort theo điểm thô.
Nó còn xét:

- novelty
- diversity
- historical pass rate theo source
- soft quota giữa `planner / scout / fix`

Khi hệ rơi vào stagnation, `orchestrator_loop.py` có thể đổi quota tạm thời để mở rộng explore hoặc tăng chỗ cho scout/fix.

## 7. Evaluate và local scoring

### 7.1 Local scoring

`simulate_batch.py` luôn local-score trước.

Phần local scoring hiện có:

- batch scoring thay vì gọi rải rác
- multiprocessing khi đủ số candidate
- `local_score_limit` để tránh đốt CPU vô ích
- `local_score_workers` để cap worker
- `min_parallel_local_scoring` để tránh overhead khi batch quá nhỏ

### 7.2 Backend behavior

- nếu `scoring=internal`
  local result là evaluation chính

- nếu `scoring=worldquant`
  local-score vẫn chạy trước, nhưng result chính đến từ WorldQuant simulation

### 7.3 Result payload

Mỗi record evaluate thường giữ:

- local metrics
- result WorldQuant nếu có
- check columns
- sharpe, fitness, turnover, returns, margin
- pass/fail flags
- lineage và signature được carry tiếp

## 8. Artifact contract và data safety

### 8.1 Schema versioning

Artifact JSON quan trọng hiện có `schema_version`.

Khi đọc:

- validate schema cơ bản
- migrate payload legacy đơn giản nếu cần
- từ chối version mới hơn reader đang hỗ trợ
- log rõ lý do

Mục tiêu là tránh kiểu:

- đổi field nhưng script cũ vẫn đọc sai ngầm
- publish artifact nửa vời mà bước sau vẫn nuốt vào

### 8.2 Quarantine

Khi phát hiện dữ liệu bẩn, hệ ưu tiên cách ly thay vì để flow chính bị contaminate.

Các thứ có thể bị đưa vào `artifacts/quarantine/`:

- file JSON hỏng
- JSON sai schema hoặc sai version
- JSONL line lỗi
- candidate thiếu field bắt buộc hoặc sai kiểu

Quarantine hiện chia thành:

- `quarantine/files/`
- `quarantine/jsonl_lines/`
- `quarantine/candidates/`

Mỗi payload quarantine có `reason` để debug.

## 9. Evaluated pool và report

Sau evaluate, `build_evaluated_pool.py` dựng pool chuẩn:

- `evaluated_submit_ready`
- summary counts
- lineage carry-over
- signature và duplicate metadata

Từ evaluated pool, hệ render:

- `alpha_tot_nhat_hom_nay.md`
- `bang_tin_alpha.md`
- `results_summary.json`
- `results_summary.md`

Điểm quan trọng:

- một candidate có thể từng `qualified` ở planning
- nhưng chỉ candidate qua gate evaluate mới thực sự được xem là submit-ready

## 10. Bộ nhớ dùng chung

Memory toàn cục hiện được ghi ở:

```text
artifacts/state/global_research_memory.json
```

Ngoài ra còn có mirror legacy:

```text
artifacts/bo_nho_nghien_cuu.json
```

Manual override có thể `freeze` bước update memory khi cần debug hoặc khi nghi ngờ memory đang bias sai hệ.

## 11. Loop orchestrator

### 11.1 Cách chạy

```bash
bash run_wsl.sh loop --scoring internal --interval-minutes 30
```

Loop chạy liên tục cho đến khi:

- người dùng stop
- đạt ngưỡng lỗi liên tiếp
- gặp điều kiện guard nghiêm trọng

Nó không còn là cơ chế "chạy 2 vòng rồi dừng".

### 11.2 Stop và status

File status:

```text
artifacts/state/orchestrator_loop_status.json
```

File stop:

```text
artifacts/state/DUNG_LOOP
```

Lệnh hỗ trợ:

```bash
bash run_wsl.sh loop-status
bash run_wsl.sh loop-stop
```

### 11.3 Health check và severity

Loop có health check định kỳ cho:

- khả năng đọc latest/shared file
- kích thước memory
- streak lỗi API
- thời gian từ lần cuối có submit-ready alpha

Severity:

- `warning`
- `error`
- `critical`

Alert được ghi ra:

- terminal
- `system.log`
- `error.log`

### 11.4 Manual override

Các cờ override quan trọng:

- `--manual-only-fix`
- `--manual-disable-scout`
- `--manual-increase-explore`
- `--manual-freeze-memory-update`
- `--manual-ignore-block-list`

Những cờ này rất hữu ích khi:

- debug hành vi lệch
- cứu hệ lúc stagnation
- muốn cô lập riêng planner hay fix branch

## 12. Resource guard nội bộ

Repo hiện theo dõi không chỉ API quota mà còn cả tài nguyên nội bộ:

- log rotation cho runtime log
- cleanup file tạm
- giới hạn số `recent_runs` giữ lại
- timeout cho subprocess ở một số loop/script
- đo footprint của `artifacts/`

Summary cleanup được ghi ở:

```text
artifacts/state/artifact_cleanup_status.json
```

## 13. Nhánh scout

`scout` là nhánh riêng, không phụ thuộc trực tiếp vào orchestrator chính.

Nó có thể lấy ý tưởng từ:

- OpenAlex
- arXiv
- GitHub search và README
- local ZIP seed
- thư viện thesis nội bộ fallback

Artifact chính của scout:

- `artifacts/trinh_sat_hang_ngay.md`
- `artifacts/_trinh_sat/du_lieu.json`
- `artifacts/_trinh_sat/lich_su.jsonl`
- `artifacts/_trinh_sat/bo_nho.json`
- `artifacts/_trinh_sat/phan_hoi_brain.json`
- `artifacts/bao_cao_ngay/...`

`scout-loop` hiện mặc định có health gate theo feedback Brain.
Nếu feedback quá bẩn hoặc thiếu context nghiêm trọng, loop có thể dừng sớm thay vì học trên dữ liệu nhiễu.

## 14. Bản đồ artifact nên đọc trước

Nếu muốn xem nhanh:

- `artifacts/latest/latest_metadata.json`
- `artifacts/latest/alpha_tot_nhat_hom_nay.md`
- `artifacts/latest/bang_tin_alpha.md`
- `artifacts/latest/evaluated_candidates.json`
- `artifacts/latest/orchestrator_summary.json`

Nếu muốn debug theo một run:

- `artifacts/recent_runs/<run_id>/orchestrator_summary.json`
- `artifacts/recent_runs/<run_id>/pending_simulation_queue.json`
- `artifacts/recent_runs/<run_id>/results_summary.json`
- `artifacts/recent_runs/<run_id>/evaluated_candidates.json`

Nếu nghi dữ liệu bẩn:

- `artifacts/quarantine/`

## 15. Lệnh hay dùng

Kiểm tra môi trường:

```bash
bash run_wsl.sh doctor
bash run_wsl.sh auth
```

Chạy một vòng chuẩn:

```bash
bash run_wsl.sh light
```

Chạy local-only:

```bash
bash run_wsl.sh internal
```

Chạy loop:

```bash
bash run_wsl.sh loop --scoring internal --interval-minutes 30
```

Scout:

```bash
bash run_wsl.sh scout --count 6
```

Fix alpha:

```bash
bash run_wsl.sh fix --alpha-id YOUR_ALPHA_ID
```

Score nhanh một expression:

```bash
bash run_wsl.sh score --expression "rank(ts_zscore(abs(close-vwap),21))"
```

Cleanup:

```bash
bash run_wsl.sh cleanup
python3 scripts/runtime_control.py clear-stale-locks artifacts
```

## 16. Tóm tắt phiên bản hiện tại

Phiên bản `brain-learn` hiện tại có các đặc điểm đáng chú ý:

- pipeline chuẩn dựa trên `evaluated_candidates.json`
- publish `latest/` có marker complete rõ ràng
- `recent_runs/<run_id>/` là layout chính
- artifact JSON có `schema_version`
- dữ liệu lỗi bị cách ly vào `quarantine/`
- candidate có lineage sâu hơn
- queue có dedupe nhiều tầng và soft quota theo source
- local scoring đã batch hóa và có multiprocessing
- loop có health check phân cấp và manual override
- cleanup có resource guard và retention rõ ràng
