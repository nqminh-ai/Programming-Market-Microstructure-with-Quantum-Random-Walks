# Overfitting Critical Fixes

**Ngày:** 2026-06-13

## Kết quả xác minh ba finding trong ảnh

| Finding | Đánh giá | Trạng thái |
|---|---|---|
| `calibrate()` refit bằng train + validation | Refit này chỉ hợp lệ khi còn external test, nhưng làm validation score không còn là score của final fit | Đã loại bỏ; validation chỉ dùng chọn candidate |
| `calibrate_two_stage()` bias update dùng lại warmup | Đúng, đây là double-dip so với protocol two-stage đã mô tả | Đã sửa; bias chỉ fit trên rows sau warmup |
| `tick_processor.py` dùng `shift(-1)` | Đúng, đây là look-ahead trực tiếp | Đã sửa bằng past-only regime confirmation |

## Thay đổi implementation

- `MarketQRW.calibrate()` giữ nguyên candidate được fit trên train; không final
  refit bằng validation.
- `MarketQRW.calibrate_two_stage()` dùng post-warmup slice cho bias update.
- `AdaptiveDecoherenceQRW.calibrate_two_stage()` dùng structural train,
  selection-only validation và post-warmup bias data tách biệt.
- `phase3_overfitting_audit.py` không còn dùng lại warmup khi update bias trong
  rolling/walk-forward folds.
- `TickProcessor` không còn đọc tick kế tiếp.
- Thêm hai regression test đối kháng: prefix kết thúc đúng tại tick cần quyết
  định, và thay đổi riêng giá trị tick `t+1`. Output đến `t` phải bất biến trong
  cả hai trường hợp.
- Calibration artifact ghi rõ:
  `final_refit_includes_validation=false`,
  `bias_update_reuses_warmup=false`, và cảnh báo sample size.

## Kết quả sau fix

Live June 12 được rebuild:

- Input rows: `1,939`
- Output causal-cleaned rows: `1,908`
- Structural adaptive fit events: `78`
- Parameters: `6`
- Events per structural parameter: `13.0`
- `low_sample_warning=true`

Post-fix overfitting audit:

- QRW test Brier: `0.119815`
- Fair affine test Brier: `0.103214`
- QRW minus fair affine: `+0.016600`
- 95% interval: `[0.006711, 0.025549]`
- Pooled walk-forward delta: `+0.033635`
- Walk-forward interval: `[0.012353, 0.055213]`

**Kết luận:** sau khi bỏ leakage/double-dip, QRW không còn predictive edge trên
live window và thua fair affine baseline có cùng OBI/tick-direction inputs.

## Artifact còn phải rebuild

Các processed/features June 1-7 vẫn được tạo bằng cleaner cũ có look-ahead.
Do đó:

- `reports/archive/invalidated_2026-06-13/phase3_multiday_edge_audit.*`
  là stale.
- `reports/archive/invalidated_2026-06-13/phase3_adaptive_decoherence_audit.*`
  là stale.
- Không được dùng các CI hoặc claim edge trong hai report này làm bằng chứng.
- June 1-7 cũng đã được xem nhiều lần, nên sau rebuild chỉ được dùng làm
  development data; confirmatory test phải dùng fresh untouched dates.

## Verification

`python -m pytest -q`: **46 passed**
