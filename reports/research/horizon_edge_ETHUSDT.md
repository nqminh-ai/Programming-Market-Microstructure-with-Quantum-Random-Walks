# Có edge ở horizon giao dịch được không? — ETHUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — nhãn dự báo mới, không phải kết quả confirmatory.

- Feature file: `features_ETHUSDT_2026-06-12.parquet` (2,872,918 dòng)
- Git commit: `6d492743cba0c21d9553ff27e3edf35f46d3d2e7` · Python 3.14.5
- Cửa sổ **không chồng lấp** (mỗi ETHUSDT nhãn cách nhau đúng `horizon` tick nên không chia sẻ tương lai)

## Độ chính xác đạt được so với ngưỡng cần có

| Horizon | Thời gian | Cửa sổ (train/test) | Lớp đa số | Mô hình tốt nhất | Độ chính xác | Ngưỡng maker 2bps | Lãi ròng/lệnh | Đạt? |
|---:|---:|---:|---:|---|---:|---:|---:|:--:|
| 1,000 | 30.1 giây | 1997/856 | 50.8% | Logistic L2 (5F) | 58.8% | 93.3% | -2.50 bps | ✘ |
| 5,000 | 2.5 phút | 400/172 | 51.7% | Logistic L2 + Pairwise | 56.4% | 69.5% | -2.11 bps | ✘ |
| 10,000 | 5.0 phút | 200/86 | 53.5% | Logistic L2 + Pairwise | 60.5% | 63.0% | -0.62 bps | ✘ |
| 50,000 | — | không đủ mẫu | — | — | — | — | — | — |

## Chi tiết từng mô hình

### Horizon 1,000 (30.1 giây)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 (5F) | 58.8% | ✅ |
| Logistic L2 + Pairwise | 58.4% | ✅ |
| OrderFlow AR(5) | 57.7% | ✅ |
| Majority class | 49.2% | ✘ |

### Horizon 5,000 (2.5 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 + Pairwise | 56.4% | ✅ |
| OrderFlow AR(5) | 55.2% | ✅ |
| Logistic L2 (5F) | 52.9% | ✅ |
| Majority class | 51.7% | ✘ |

### Horizon 10,000 (5.0 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 + Pairwise | 60.5% | ✅ |
| OrderFlow AR(5) | 54.7% | ✅ |
| Majority class | 53.5% | ✘ |
| Logistic L2 (5F) | 51.2% | ✘ |

## Kết luận

Đánh giá 3 horizon trên các cửa sổ **không chồng lấp**. Có 3/3 horizon mà mô hình tốt nhất vượt baseline đa số; cao nhất là h=10,000 (Logistic L2 + Pairwise, 60.5% so với 53.5%). **Không horizon nào đạt ngưỡng hoà vốn** kể cả ở mức phí maker 2bps. Điểm cốt lõi: **kỹ năng dự báo và khả năng sinh lời nằm ở hai đầu đối lập của thang horizon**. Ở h=10,000 độ chính xác cao nhất (60.5%) nhưng biên độ giá quá nhỏ nên lãi ròng vẫn là **-0.62 bps/lệnh**; ở horizon dài, biên độ đủ lớn thì kỹ năng lại biến mất. Cỡ mẫu kiểm định nhỏ nhất chỉ 86 cửa sổ — mọi con số ở đây có khoảng tin cậy rất rộng và không được coi là kết luận.
