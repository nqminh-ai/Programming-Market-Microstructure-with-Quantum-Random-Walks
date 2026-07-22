# Khả thi giao dịch theo horizon — BTCUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — phân tích khả thi chi phí, không phải bằng chứng có lợi nhuận.

- Feature file: `features_BTCUSDT_recent_subset.parquet` (4,000,000 dòng)
- Git commit: `3a532125b1a1a3043f109798feb61e40939c7d2d` · Python 3.14.5
- Half-spread **đo được** từ dữ liệu: 0.202 bps (|price − mid| / mid)
- Nhịp giao dịch: 20.4 tick/giây

## Chi phí một vòng mua-bán

| Kịch bản | Chi phí vòng |
|---|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | +10.40 bps |
| Taker, 4bps/chiều (Binance futures base) | +8.40 bps |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | +3.60 bps |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | -0.40 bps |

## Độ chính xác hướng cần có để hoà vốn

`p > 0,5 + chi_phí / (2·E|biến động|)`. Ô ✅ nghĩa là ngưỡng hoà vốn nằm dưới 60% — mức còn có thể bàn tới. `—` nghĩa là không thể hoà vốn dù dự đoán đúng 100%. `⚠ spread` nghĩa là spread thu được lớn hơn phí, khiến công thức trên kết luận có lãi ở **mọi** độ chính xác; đó là ảo giác của mô hình vì nó **chưa tính adverse selection** — lệnh chờ có xu hướng được khớp đúng lúc thị trường đi ngược lại bạn. Muốn dùng kịch bản đó phải mô hình hoá hàng đợi lệnh và tỉ lệ khớp bằng dữ liệu L2 thật.

| Horizon | Thời gian | E\|biến động\| | Taker 5bps | Taker 4bps | Maker 2bps | Maker 0bps |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 giây | 5.36e-07 | — | — | — | ⚠ spread |
| 10 | 0.5 giây | 4.89e-06 | — | — | — | ⚠ spread |
| 100 | 4.9 giây | 4.56e-05 | — | — | — | ⚠ spread |
| 1,000 | 49.1 giây | 2.56e-04 | — | — | — | ⚠ spread |
| 5,000 | 4.1 phút | 6.14e-04 | — | — | 79.3% | ⚠ spread |
| 10,000 | 8.2 phút | 8.52e-04 | — | 99.3% | 71.1% | ⚠ spread |
| 50,000 | 40.9 phút | 1.81e-03 | 78.7% | 73.2% | 59.9% ✅ | ⚠ spread |
| 100,000 | 81.9 phút | 2.52e-03 | 70.7% | 66.7% | 57.1% ✅ | ⚠ spread |
| 200,000 | 2.7 giờ | 3.42e-03 | 65.2% | 62.3% | 55.3% ✅ | ⚠ spread |

## Horizon nhỏ nhất còn giao dịch được

| Kịch bản | Horizon | Thời gian |
|---|---:|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | không có | — |
| Taker, 4bps/chiều (Binance futures base) | không có | — |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | 50,000 tick | 40.9 phút |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | không có | — |

## Kết luận

Ở horizon 1 tick, biến động trung bình chỉ bằng **0.0005 lần** chi phí một vòng taker — nghĩa là **một mô hình dự đoán đúng 100% vẫn lỗ**. Đây là giới hạn của horizon, không phải của mô hình. Đặt lệnh chờ (maker, 2bps/chiều) trở nên khả thi từ **50,000 tick (~40.9 phút)**. Ở mức phí taker 5bps/chiều, **không horizon nào** trong lưới đạt ngưỡng khả thi. Cần nhấn mạnh: vượt ngưỡng chi phí là điều kiện **cần, không đủ**. Bảng này chỉ nói biến động đủ lớn để trả phí; nó **không** nói dự án có khả năng dự đoán đúng hướng ở horizon đó — điều chưa từng được chứng minh.
