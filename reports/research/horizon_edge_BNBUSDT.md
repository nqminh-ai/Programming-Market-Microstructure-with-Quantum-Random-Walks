# Có edge ở horizon giao dịch được không? — BNBUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — nhãn dự báo mới, không phải kết quả confirmatory.

- Feature file: `features_BNBUSDT_69d.parquet` (54,096,780 dòng)
- Git commit: `2cef6a187e665bca3d1143b6861453abf40cd969` · Python 3.14.5
- Cửa sổ **không chồng lấp** (mỗi BNBUSDT nhãn cách nhau đúng `horizon` tick nên không chia sẻ tương lai)

## Độ chính xác đạt được so với ngưỡng cần có

| Horizon | Thời gian | Cửa sổ (train/test) | Lớp đa số | Mô hình tốt nhất | Độ chính xác | KTC 95% | Ngưỡng maker 2bps | Lãi ròng/lệnh | Đạt? |
|---:|---:|---:|---:|---|---:|---:|---:|---:|:--:|
| 1,000 | 1.8 phút | 37510/16076 | 50.0% | OrderFlow AR(5) | 54.4% | [53.6, 55.2]% | 97.3% | -5.52 bps | ✘ |
| 5,000 | 9.2 phút | 7536/3230 | 50.3% | OrderFlow AR(5) | 52.9% | [51.2, 54.6]% | 71.3% | -5.38 bps | ✘ |
| 10,000 | 18.4 phút | 3775/1618 | 51.4% | Logistic L2 + Pairwise | 51.9% | [49.4, 54.3]% | 65.2% | -5.49 bps | ✘ |
| 50,000 | 1.5 giờ | 753/323 | 50.8% | Logistic L2 + Pairwise | 53.6% | [48.1, 58.9]% | 57.3% | -3.43 bps | ✘ |

## Chi tiết từng mô hình

### Horizon 1,000 (1.8 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| OrderFlow AR(5) | 54.4% | ✅ |
| Logistic L2 (5F) | 54.3% | ✅ |
| Logistic L2 + Pairwise | 54.3% | ✅ |
| Majority class | 50.0% | ✘ |

### Horizon 5,000 (9.2 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| OrderFlow AR(5) | 52.9% | ✅ |
| Logistic L2 (5F) | 52.5% | ✅ |
| Logistic L2 + Pairwise | 51.7% | ✅ |
| Majority class | 50.3% | ✅ |

### Horizon 10,000 (18.4 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 + Pairwise | 51.9% | ✅ |
| OrderFlow AR(5) | 50.8% | ✘ |
| Logistic L2 (5F) | 50.4% | ✘ |
| Majority class | 48.6% | ✘ |

### Horizon 50,000 (1.5 giờ)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 + Pairwise | 53.6% | ✅ |
| Logistic L2 (5F) | 51.4% | ✅ |
| Majority class | 50.8% | ✘ |
| OrderFlow AR(5) | 50.2% | ✘ |

## Kết luận

Đánh giá 4 horizon trên các cửa sổ **không chồng lấp**. Có 4/4 horizon mà mô hình tốt nhất vượt baseline đa số; cao nhất là h=1,000 (OrderFlow AR(5), 54.4% so với 50.0%). **Không horizon nào vượt ngưỡng hoà vốn** kể cả ở mức phí maker 2bps. Điểm cốt lõi: **kỹ năng dự báo và khả năng sinh lời nằm ở hai đầu đối lập của thang horizon**. Ở h=1,000 độ chính xác cao nhất (54.4%) nhưng biên độ giá quá nhỏ nên lãi ròng vẫn là **-5.52 bps/lệnh**; ở horizon dài, biên độ đủ lớn thì kỹ năng lại biến mất. Cỡ mẫu kiểm định nhỏ nhất chỉ 323 cửa sổ — mọi con số ở đây có khoảng tin cậy rất rộng và không được coi là kết luận.
