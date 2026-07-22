# Khả thi giao dịch theo horizon — ETHUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — phân tích khả thi chi phí, không phải bằng chứng có lợi nhuận.

- Feature file: `features_ETHUSDT_2026-06-12.parquet` (2,872,918 dòng)
- Git commit: `3a532125b1a1a3043f109798feb61e40939c7d2d` · Python 3.14.5
- Half-spread **đo được** từ dữ liệu: 0.431 bps (|price − mid| / mid)
- Nhịp giao dịch: 33.3 tick/giây

## Chi phí một vòng mua-bán

| Kịch bản | Chi phí vòng |
|---|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | +10.86 bps |
| Taker, 4bps/chiều (Binance futures base) | +8.86 bps |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | +3.14 bps |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | -0.86 bps |

## Độ chính xác hướng cần có để hoà vốn

`p > 0,5 + chi_phí / (2·E|biến động|)`. Ô ✅ nghĩa là ngưỡng hoà vốn nằm dưới 60% — mức còn có thể bàn tới. `—` nghĩa là không thể hoà vốn dù dự đoán đúng 100%. `⚠ spread` nghĩa là spread thu được lớn hơn phí, khiến công thức trên kết luận có lãi ở **mọi** độ chính xác; đó là ảo giác của mô hình vì nó **chưa tính adverse selection** — lệnh chờ có xu hướng được khớp đúng lúc thị trường đi ngược lại bạn. Muốn dùng kịch bản đó phải mô hình hoá hàng đợi lệnh và tỉ lệ khớp bằng dữ liệu L2 thật.

| Horizon | Thời gian | E\|biến động\| | Taker 5bps | Taker 4bps | Maker 2bps | Maker 0bps |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 giây | 1.07e-06 | — | — | — | ⚠ spread |
| 10 | 0.3 giây | 9.65e-06 | — | — | — | ⚠ spread |
| 100 | 3.0 giây | 8.28e-05 | — | — | — | ⚠ spread |
| 1,000 | 30.1 giây | 3.67e-04 | — | — | 92.8% | ⚠ spread |
| 5,000 | 2.5 phút | 8.08e-04 | — | — | 69.4% | ⚠ spread |
| 10,000 | 5.0 phút | 1.14e-03 | 97.4% | 88.7% | 63.7% | ⚠ spread |
| 50,000 | 25.1 phút | 2.49e-03 | 71.8% | 67.8% | 56.3% ✅ | ⚠ spread |
| 100,000 | 50.1 phút | 3.59e-03 | 65.1% | 62.3% | 54.4% ✅ | ⚠ spread |
| 200,000 | 1.7 giờ | 5.27e-03 | 60.3% | 58.4% ✅ | 53.0% ✅ | ⚠ spread |

## Horizon nhỏ nhất còn giao dịch được

| Kịch bản | Horizon | Thời gian |
|---|---:|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | không có | — |
| Taker, 4bps/chiều (Binance futures base) | 200,000 tick | 1.7 giờ |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | 50,000 tick | 25.1 phút |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | không có | — |

## Kết luận

Ở horizon 1 tick, biến động trung bình chỉ bằng **0.0010 lần** chi phí một vòng taker — nghĩa là **một mô hình dự đoán đúng 100% vẫn lỗ**. Đây là giới hạn của horizon, không phải của mô hình. Đặt lệnh chờ (maker, 2bps/chiều) trở nên khả thi từ **50,000 tick (~25.1 phút)**. Ở mức phí taker 5bps/chiều, **không horizon nào** trong lưới đạt ngưỡng khả thi. Cần nhấn mạnh: vượt ngưỡng chi phí là điều kiện **cần, không đủ**. Bảng này chỉ nói biến động đủ lớn để trả phí; nó **không** nói dự án có khả năng dự đoán đúng hướng ở horizon đó — điều chưa từng được chứng minh.
