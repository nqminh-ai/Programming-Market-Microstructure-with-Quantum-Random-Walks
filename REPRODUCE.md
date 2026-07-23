# Tái lập kết quả

Mọi con số headline trong [báo cáo cuối](docs/final_report.md) và [tóm tắt điều
hành](docs/executive_summary.md) đều truy ngược được về một artifact JSON có
provenance trong [`reports/research/`](reports/research/). Tài liệu này cho bạn
ba mức kiểm chứng, từ nhanh nhất đến đầy đủ nhất.

## Chuỗi bằng chứng

```
lệnh (tài liệu hoá)  →  artifact JSON (có provenance)  →  prose trong docs
   reproduce.py            reproduce.py (verify)          test_report_numbers.py
   --commands
```

Ba mắt xích, mỗi mắt xích kiểm được bằng máy:

| Mắt xích | Công cụ | Kiểm gì |
|---|---|---|
| Lệnh → artifact | `python -m scripts.operations.reproduce --commands` | In lệnh tạo lại từng artifact, trỏ đúng file input thật |
| Artifact hợp lệ | `python -m scripts.operations.reproduce` | Mọi artifact tồn tại, JSON hợp lệ, có git commit + Python + SHA, gắn nhãn **exploratory** chứ không phải confirmatory |
| Artifact → prose | `python -m pytest tests/test_report_numbers.py` | Các con số trong docs đúng bằng số trong artifact |

## Mức 1 — Xác minh artifact đã ship (vài giây)

Không chạy lại nghiên cứu; kiểm tra artifact kèm theo repo là well-formed, có
provenance và gắn nhãn đúng. Thoát khác 0 nếu bất kỳ artifact nào thiếu/hỏng, nên
dùng được để gate một release.

```powershell
python -m scripts.operations.reproduce
python -m pytest tests/test_report_numbers.py tests/test_reproduce.py -q
```

## Mức 2 — Chạy lại toàn bộ pipeline Phase 2–6 (phút–giờ)

```powershell
make full        # data → simulate → test → report
```

hoặc trên Windows PowerShell (nếu không có `make`), chạy trực tiếp:

```powershell
python -m scripts.pipelines.phase2_pipeline process
python -m scripts.pipelines.phase2_pipeline features --obi-source trade_imbalance
python -m scripts.pipelines.phase4_pipeline
python -m scripts.pipelines.phase5_pipeline
python -m scripts.pipelines.phase6_pipeline
```

## Mức 3 — Tạo lại các artifact nghiên cứu §5b–5e (giờ)

Đây là các con số mà kết luận hiện tại thực sự dựa vào, và Makefile **không** phủ
chúng. In danh sách lệnh chính xác (kèm ghi chú về quy mô/bộ nhớ):

```powershell
python -m scripts.operations.reproduce --commands
```

Lưu ý quy mô: run confirmation 100M dòng là nặng nhất trong dự án; mỗi run CRPS
40-window mất ~1 giờ/tài sản. Seed cố định (2026), nên cùng input cho cùng số.

## Cố định môi trường

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --requirement requirements.lock
```

`requirements.lock` là exact pins đã resolve trên CPython 3.14 / Windows. Regenerate
có chủ đích cho nền tảng khác.

## Một chú thích trung thực về provenance §5d

Bảng marginal-CRPS 5-window (§5d) được chạy **trước khi** các store 69 ngày tồn
tại, mỗi tài sản off một file riêng (BTC `recent_subset`, ETH `2026-06-12`, BNB
`multiday`). Hai trong ba (BTC, ETH) có trước khi thêm trường `feature_sha256`,
nên artifact của chúng mang provenance yếu hơn phần còn lại — `reproduce.py` đánh
dấu rõ điều này thay vì che đi. Các run day-cluster (giới hạn #4, chạy trên full
69 ngày với cửa sổ theo ngày UTC) là bản thay thế provenance đầy đủ.
