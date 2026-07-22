# Có edge ở horizon giao dịch được không? — BNBUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — nhãn dự báo mới, không phải kết quả confirmatory.

- Feature file: `features_BNBUSDT_multiday.parquet` (31,503,940 dòng)
- Git commit: `6d492743cba0c21d9553ff27e3edf35f46d3d2e7` · Python 3.14.5
- Cửa sổ **không chồng lấp** (mỗi BNBUSDT nhãn cách nhau đúng `horizon` tick nên không chia sẻ tương lai)

## Độ chính xác đạt được so với ngưỡng cần có

| Horizon | Thời gian | Cửa sổ (train/test) | Lớp đa số | Mô hình tốt nhất | Độ chính xác | Ngưỡng maker 2bps | Lãi ròng/lệnh | Đạt? |
|---:|---:|---:|---:|---|---:|---:|---:|:--:|
| 1,000 | 85.0 giây | 21854/9367 | 50.5% | OrderFlow AR(5) | 54.0% | 65.9% | -1.57 bps | ✘ |
| 5,000 | 7.1 phút | 4389/1882 | 50.7% | Logistic L2 + Pairwise | 52.0% | 57.0% | -1.49 bps | ✘ |
| 10,000 | 14.2 phút | 2198/943 | 50.8% | Majority class | 49.2% | 54.8% | -2.44 bps | ✘ |
| 50,000 | 70.8 phút | 438/189 | 51.3% | Logistic L2 + Pairwise | 51.9% | 52.2% | -0.36 bps | ✘ |

## Chi tiết từng mô hình

### Horizon 1,000 (85.0 giây)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| OrderFlow AR(5) | 54.0% | ✅ |
| Logistic L2 (5F) | 54.0% | ✅ |
| Logistic L2 + Pairwise | 53.9% | ✅ |
| Majority class | 49.5% | ✘ |

### Horizon 5,000 (7.1 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 + Pairwise | 52.0% | ✅ |
| Logistic L2 (5F) | 51.4% | ✅ |
| OrderFlow AR(5) | 51.1% | ✅ |
| Majority class | 50.7% | ✘ |

### Horizon 10,000 (14.2 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Majority class | 49.2% | ✘ |
| OrderFlow AR(5) | 49.0% | ✘ |
| Logistic L2 (5F) | 48.7% | ✘ |
| Logistic L2 + Pairwise | 46.6% | ✘ |

### Horizon 50,000 (70.8 phút)

| Mô hình | Độ chính xác | Hơn lớp đa số? |
|---|---:|:--:|
| Logistic L2 + Pairwise | 51.9% | ✅ |
| Majority class | 51.3% | ✘ |
| Logistic L2 (5F) | 51.3% | ✘ |
| OrderFlow AR(5) | 50.3% | ✘ |

## Kết luận

Đánh giá 4 horizon trên các cửa sổ **không chồng lấp**. Có 3/4 horizon mà mô hình tốt nhất vượt baseline đa số; cao nhất là h=1,000 (OrderFlow AR(5), 54.0% so với 50.5%). **Không horizon nào đạt ngưỡng hoà vốn** kể cả ở mức phí maker 2bps. Điểm cốt lõi: **kỹ năng dự báo và khả năng sinh lời nằm ở hai đầu đối lập của thang horizon**. Ở h=1,000 độ chính xác cao nhất (54.0%) nhưng biên độ giá quá nhỏ nên lãi ròng vẫn là **-1.57 bps/lệnh**; ở horizon dài, biên độ đủ lớn thì kỹ năng lại biến mất. Cỡ mẫu kiểm định nhỏ nhất chỉ 189 cửa sổ — mọi con số ở đây có khoảng tin cậy rất rộng và không được coi là kết luận.
