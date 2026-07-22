# Có edge ở horizon giao dịch được không? — BTCUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — nhãn dự báo mới, không phải kết quả confirmatory.

- Feature file: `features_BTCUSDT_multiday.parquet` (32,439,057 dòng)
- Git commit: `6d492743cba0c21d9553ff27e3edf35f46d3d2e7` · Python 3.14.5
- Cửa sổ **không chồng lấp** (mỗi BTCUSDT nhãn cách nhau đúng `horizon` tick nên không chia sẻ tương lai)

## Độ chính xác đạt được so với ngưỡng cần có

| Horizon | Thời gian | Cửa sổ (train/test) | Lớp đa số | Mô hình tốt nhất | Độ chính xác | Ngưỡng maker 2bps | Lãi ròng/lệnh | Đạt? |
|---:|---:|---:|---:|---|---:|---:|---:|:--:|
| 1,000 | 26.6 giây | 22621/9695 | 51.2% | OrderFlow AR(5) | 65.7% | — | -2.56 bps | ✘ |
| 5,000 | 2.2 phút | 4532/1943 | 51.0% | Logistic L2 (5F) | 57.2% | 74.9% | -2.48 bps | ✘ |
| 10,000 | 4.4 phút | 2266/972 | 50.8% | Logistic L2 + Pairwise | 53.9% | 67.5% | -2.70 bps | ✘ |
| 50,000 | 22.2 phút | 452/194 | 55.7% | Majority class | 55.7% | 57.7% | -0.92 bps | ✘ |

## Chi tiết từng mô hình

### Horizon 1,000 (26.6 giây)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| OrderFlow AR(5) | 65.7% | ✅ |
| Logistic L2 (5F) | 65.5% | ✅ |
| Logistic L2 + Pairwise | 65.5% | ✅ |
| Majority class | 48.8% | ✘ |

### Horizon 5,000 (2.2 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 (5F) | 57.2% | ✅ |
| OrderFlow AR(5) | 57.0% | ✅ |
| Logistic L2 + Pairwise | 56.8% | ✅ |
| Majority class | 49.0% | ✘ |

### Horizon 10,000 (4.4 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 + Pairwise | 53.9% | ✅ |
| OrderFlow AR(5) | 53.0% | ✅ |
| Logistic L2 (5F) | 52.9% | ✅ |
| Majority class | 50.8% | ✘ |

### Horizon 50,000 (22.2 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Majority class | 55.7% | ✘ |
| Logistic L2 + Pairwise | 51.5% | ✘ |
| OrderFlow AR(5) | 51.0% | ✘ |
| Logistic L2 (5F) | 50.5% | ✘ |

## Kết luận

Đánh giá 4 horizon trên các cửa sổ **không chồng lấp**. Có 3/4 horizon mà mô hình tốt nhất vượt baseline đa số; cao nhất là h=1,000 (OrderFlow AR(5), 65.7% so với 51.2%). **Không horizon nào đạt ngưỡng hoà vốn** kể cả ở mức phí maker 2bps. Điểm cốt lõi: **kỹ năng dự báo và khả năng sinh lời nằm ở hai đầu đối lập của thang horizon**. Ở h=1,000 độ chính xác cao nhất (65.7%) nhưng biên độ giá quá nhỏ nên lãi ròng vẫn là **-2.56 bps/lệnh**; ở horizon dài, biên độ đủ lớn thì kỹ năng lại biến mất. Cỡ mẫu kiểm định nhỏ nhất chỉ 194 cửa sổ — mọi con số ở đây có khoảng tin cậy rất rộng và không được coi là kết luận.
