# Có edge ở horizon giao dịch được không? — BTCUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — nhãn dự báo mới, không phải kết quả confirmatory.

- Feature file: `features_BTCUSDT_69d.parquet` (227,586,094 dòng)
- Git commit: `9adde323946b06d11d80bff4cd1b0a0419595aed` · Python 3.14.5
- Cửa sổ **không chồng lấp** (mỗi BTCUSDT nhãn cách nhau đúng `horizon` tick nên không chia sẻ tương lai)

## Độ chính xác đạt được so với ngưỡng cần có

| Horizon | Thời gian | Cửa sổ (train/test) | Lớp đa số | Mô hình tốt nhất | Độ chính xác | KTC 95% | Ngưỡng maker 2bps | Lãi ròng/lệnh | Đạt? |
|---:|---:|---:|---:|---|---:|---:|---:|---:|:--:|
| 1,000 | 26.2 giây | 158666/68001 | 50.0% | OrderFlow AR(5) | 64.4% | [64.0, 64.8]% | — | -2.72 bps | ✘ |
| 5,000 | 2.2 phút | 31817/13636 | 50.1% | OrderFlow AR(5) | 56.4% | [55.5, 57.2]% | 76.3% | -2.67 bps | ✘ |
| 10,000 | 4.4 phút | 15920/6823 | 50.2% | Logistic L2 (5F) | 55.5% | [54.3, 56.7]% | 68.5% | -2.48 bps | ✘ |
| 50,000 | 21.8 phút | 3185/1365 | 50.3% | Majority class | 50.3% | [47.7, 53.0]% | 58.5% | -3.39 bps | ✘ |

## Chi tiết từng mô hình

### Horizon 1,000 (26.2 giây)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| OrderFlow AR(5) | 64.4% | ✅ |
| Logistic L2 + Pairwise | 64.3% | ✅ |
| Logistic L2 (5F) | 64.3% | ✅ |
| Majority class | 50.0% | ✘ |

### Horizon 5,000 (2.2 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| OrderFlow AR(5) | 56.4% | ✅ |
| Logistic L2 + Pairwise | 56.3% | ✅ |
| Logistic L2 (5F) | 56.2% | ✅ |
| Majority class | 50.1% | ✘ |

### Horizon 10,000 (4.4 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 (5F) | 55.5% | ✅ |
| OrderFlow AR(5) | 55.4% | ✅ |
| Logistic L2 + Pairwise | 55.3% | ✅ |
| Majority class | 49.8% | ✘ |

### Horizon 50,000 (21.8 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Majority class | 50.3% | ✘ |
| Logistic L2 (5F) | 50.3% | ✘ |
| OrderFlow AR(5) | 50.2% | ✘ |
| Logistic L2 + Pairwise | 50.0% | ✘ |

## Kết luận

Đánh giá 4 horizon trên các cửa sổ **không chồng lấp**. Có 3/4 horizon mà mô hình tốt nhất vượt baseline đa số; cao nhất là h=1,000 (OrderFlow AR(5), 64.4% so với 50.0%). **Không horizon nào vượt ngưỡng hoà vốn** kể cả ở mức phí maker 2bps. Điểm cốt lõi: **kỹ năng dự báo và khả năng sinh lời nằm ở hai đầu đối lập của thang horizon**. Ở h=1,000 độ chính xác cao nhất (64.4%) nhưng biên độ giá quá nhỏ nên lãi ròng vẫn là **-2.72 bps/lệnh**; ở horizon dài, biên độ đủ lớn thì kỹ năng lại biến mất. Cỡ mẫu kiểm định nhỏ nhất chỉ 1365 cửa sổ — mọi con số ở đây có khoảng tin cậy rất rộng và không được coi là kết luận.
