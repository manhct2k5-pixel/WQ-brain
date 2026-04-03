# Agentic AI - BI MVP

MVP này cho phép bạn:
- Tải file `CSV` hoặc `XLSX`
- Xem schema và quality profile
- Nhận insight tự động
- Hỏi dữ liệu theo kiểu BI bằng tiếng Việt hoặc tiếng Anh
- Bật planner AI nếu có `OPENAI_API_KEY`

## 1. Chạy ứng dụng

### Windows

Mở thư mục `mvp_agentic_bi` rồi chạy:

```bat
start_windows.bat
```

Nếu máy báo cổng đang bận:

```bat
set PORT=8001
start_windows.bat
```

### Linux / WSL

```bash
cd /mnt/d/skill-creator/mvp_agentic_bi
python3 server.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:8000
```

Nếu cổng `8000` đang bận:

```bash
cd /mnt/d/skill-creator/mvp_agentic_bi
PORT=8001 python3 server.py
```

## 2. Bật planner AI tùy chọn

App vẫn chạy nếu không có OpenAI API key. Khi đó nó dùng heuristic cục bộ.

Nếu muốn bật planner AI:

```bash
export OPENAI_API_KEY="your_key_here"
export OPENAI_MODEL="gpt-5-mini"
python3 server.py
```

`OPENAI_MODEL` có thể đổi sang model khác mà tài khoản của bạn đang có quyền dùng.

## 3. Câu hỏi mẫu

- `Cho tôi tổng quan dữ liệu này`
- `Tổng doanh_thu là bao nhiêu?`
- `Tổng doanh_thu theo khu_vuc`
- `Top 5 san_pham theo doanh_thu`
- `Cột nào đang thiếu dữ liệu nhiều nhất?`

## 4. Giới hạn của MVP

- Chỉ hỗ trợ `CSV` và `XLSX`
- Chỉ nạp tối đa `10.000` dòng đầu để giữ app nhẹ
- Không hỗ trợ `XLS` cũ
- Phần hỏi đáp nâng cao vẫn là MVP, phù hợp cho KPI cơ bản, group by, top N, quality check

## 5. Kiến trúc ngắn gọn

- `server.py`: web server, parser dữ liệu, planner và execution engine
- `templates/index.html`: giao diện
- `static/app.js`: logic frontend
- `static/styles.css`: giao diện

## 6. Hướng nâng cấp

- Kết nối SQL thay cho upload file
- Thêm chart
- Thêm semantic layer riêng
- Thêm forecast và anomaly detection
- Thêm action workflow sau approval

## 7. Nếu bạn vẫn không chạy được

Kiểm tra 3 điểm sau:
- Máy có `Python 3` chưa: chạy `python --version` hoặc `py -3 --version`
- Bạn có đang đứng đúng thư mục `mvp_agentic_bi` không
- Cổng `8000` có đang bị ứng dụng khác dùng không

Nếu vẫn lỗi, gửi lại đúng dòng lệnh bạn chạy và toàn bộ thông báo lỗi hiện ra trong terminal.
