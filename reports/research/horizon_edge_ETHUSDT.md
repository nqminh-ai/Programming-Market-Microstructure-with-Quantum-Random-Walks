# Có edge ở horizon giao dịch được không? — ETHUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — nhãn dự báo mới, không phải kết quả confirmatory.

- Feature file: `features_ETHUSDT_69d.parquet` (211,995,471 dòng)
- Git commit: `2cef6a187e665bca3d1143b6861453abf40cd969` · Python 3.14.5
- Cửa sổ **không chồng lấp** (mỗi ETHUSDT nhãn cách nhau đúng `horizon` tick nên không chia sẻ tương lai)

## Độ chính xác đạt được so với ngưỡng cần có

| Horizon | Thời gian | Cửa sổ (train/test) | Lớp đa số | Mô hình tốt nhất | Độ chính xác | KTC 95% | Ngưỡng maker 2bps | Lãi ròng/lệnh | Đạt? |
|---:|---:|---:|---:|---|---:|---:|---:|---:|:--:|
| 1,000 | 28.1 giây | 147616/63265 | 50.1% | OrderFlow AR(5) | 58.5% | [58.1, 58.9]% | — | -5.82 bps | ✘ |
| 5,000 | 2.3 phút | 29611/12691 | 50.1% | Logistic L2 + Pairwise | 53.8% | [52.9, 54.7]% | 86.0% | -5.83 bps | ✘ |
| 10,000 | 4.7 phút | 14808/6347 | 50.2% | Logistic L2 + Pairwise | 52.6% | [51.3, 53.8]% | 75.4% | -5.88 bps | ✘ |
| 50,000 | 23.4 phút | 2963/1271 | 51.3% | Logistic L2 (5F) | 51.0% | [48.2, 53.7]% | 61.5% | -5.92 bps | ✘ |

## Chi tiết từng mô hình

### Horizon 1,000 (28.1 giây)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| OrderFlow AR(5) | 58.5% | ✅ |
| Logistic L2 + Pairwise | 58.5% | ✅ |
| Logistic L2 (5F) | 58.5% | ✅ |
| Majority class | 50.1% | ✘ |

### Horizon 5,000 (2.3 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 + Pairwise | 53.8% | ✅ |
| Logistic L2 (5F) | 53.8% | ✅ |
| OrderFlow AR(5) | 53.6% | ✅ |
| Majority class | 50.1% | ✘ |

### Horizon 10,000 (4.7 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 + Pairwise | 52.6% | ✅ |
| Logistic L2 (5F) | 52.5% | ✅ |
| OrderFlow AR(5) | 52.2% | ✅ |
| Majority class | 49.8% | ✘ |

### Horizon 50,000 (23.4 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 (5F) | 51.0% | ✘ |
| Logistic L2 + Pairwise | 51.0% | ✘ |
| OrderFlow AR(5) | 50.8% | ✘ |
| Majority class | 48.7% | ✘ |

## Kết luận

Đánh giá 4 horizon trên các cửa sổ **không chồng lấp**. Có 3/4 horizon mà mô hình tốt nhất vượt baseline đa số; cao nhất là h=1,000 (OrderFlow AR(5), 58.5% so với 50.1%). **Không horizon nào vượt ngưỡng hoà vốn** kể cả ở mức phí maker 2bps. Điểm cốt lõi: **kỹ năng dự báo và khả năng sinh lời nằm ở hai đầu đối lập của thang horizon**. Ở h=1,000 độ chính xác cao nhất (58.5%) nhưng biên độ giá quá nhỏ nên lãi ròng vẫn là **-5.82 bps/lệnh**; ở horizon dài, biên độ đủ lớn thì kỹ năng lại biến mất. Cỡ mẫu kiểm định nhỏ nhất chỉ 1271 cửa sổ — mọi con số ở đây có khoảng tin cậy rất rộng và không được coi là kết luận.
