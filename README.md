# AI Quantum — QRW cho vi cấu trúc thị trường

Dự án nghiên cứu thăm dò quantum random walk (QRW) density-matrix và các
baseline cổ điển trên dữ liệu giao dịch tần suất cao.

## Kết luận hiện tại

Ablation/so-sánh Phase 1–5 ([reports/research/](reports/research/)) cho thấy
**không có lợi thế dự báo bền vững của QRW, và cơ chế lượng tử (pha) đóng góp
bằng 0**: (1) pha `alpha_phase` = 0 trên cả ba asset; (2) ở chiều directional,
windowing thắng affine yếu nhưng **thua** baseline cổ điển mạnh (OrderFlow
AR(5), Logistic+Pairwise) trên cả ba asset — ETH xếp chót 7/7; (3) ở endpoint
chính đăng-ký-trước là marginal CRPS, QRW **cạnh tranh** với GARCH/GBM (hạng 1
trên ETH, 2 trên BNB, 3 trên BTC) — không thua rõ nhưng cũng không vượt trội.
Fold-fragility trên BTC là bug `calibrate_bias` đã sửa (Phase 2); full 32M dataset
gốc đã chạy được (Phase 4). OBI vẫn là trade-flow proxy, không phải L2 LOB. Xem
[báo cáo cuối](docs/final_report.md) §5b–5d.

**Bắt đầu từ đây:** [tóm tắt điều hành](docs/executive_summary.md) — một trang
kết luận, bằng chứng chốt và hạn chế, dành cho người đọc lần đầu.

Chi tiết đầy đủ ở [báo cáo cuối](docs/final_report.md) và
[trạng thái artifact](reports/ARTIFACT_STATUS.md).

## Cài đặt

Yêu cầu CPython 3.14 và môi trường ảo riêng.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --requirement requirements.lock
```

`requirements.txt` và `requirements.lock` đều dùng exact pins. Không dùng file
credential trong repository; cấu hình bí mật qua biến môi trường hoặc private
file nằm ngoài Git.

## Kiểm thử

```powershell
python -m pytest tests/ -v
```

## Cấu trúc dự án

```text
config/                          cấu hình theo tài sản
data/assets/<symbol>/            raw, processed và features
docs/                            báo cáo, kế hoạch và ghi chú lý thuyết
scripts/pipelines/               pipeline Phase 2–6
scripts/operations/              thu thập, rebuild và đóng băng release
scripts/audits/                  kiểm toán chống overfit/leakage
scripts/research/                thí nghiệm mở rộng
src/                             thư viện ứng dụng
tests/                           kiểm thử tự động
reports/assets/<symbol>/         metadata/chất lượng dữ liệu canonical
reports/archive/                 báo cáo đã vô hiệu hóa, chỉ để truy vết
results/archive/                 kết quả đã vô hiệu hóa, chỉ để truy vết
```

Ba symbol chuẩn là `btcusdt`, `ethusdt`, `bnbusdt`.

## Chạy pipeline

```powershell
python -m scripts.pipelines.phase3_pipeline --feature-path data/assets/btcusdt/features/features_BTCUSDT_<date>.parquet
python -m scripts.audits.phase3_overfitting_audit --feature-path data/assets/btcusdt/features/features_BTCUSDT_<date>.parquet
python -m scripts.pipelines.phase4_pipeline --feature-path data/assets/btcusdt/features/features_BTCUSDT_<date>.parquet
python -m scripts.pipelines.phase5_pipeline --feature-path data/assets/btcusdt/features/features_BTCUSDT_<date>.parquet
python -m scripts.pipelines.phase6_pipeline --feature-path data/assets/btcusdt/features/features_BTCUSDT_<date>.parquet
```

Run chính thức yêu cầu source tree sạch. Pipeline hard-fail nếu protocol,
commit, feature path hoặc SHA-256 không khớp. Phase 6 tạo
`reports/release_manifest.json` cho input/output chính thức.

## Quy tắc thống kê

- QRW chỉ được đánh giá như fixed-origin marginals.
- Primary endpoint: mean marginal CRPS.
- Tie-break: directional log loss.
- ACF/tail: chỉ model có trajectory thật.
- Diebold–Mariano: rolling-origin one-step losses căn chỉnh timestamp.
- AIC/BIC: chỉ so sánh trong cùng likelihood family.

## Nghiên cứu confirmatory tiếp theo

Pre-registration nằm tại [docs/data_collection_todo.md](docs/data_collection_todo.md):
tối thiểu 20 ngày UTC tương lai cho mỗi tài sản, trade/L2 LOB đồng bộ và không
mở nhãn holdout trước khi đóng băng protocol.
