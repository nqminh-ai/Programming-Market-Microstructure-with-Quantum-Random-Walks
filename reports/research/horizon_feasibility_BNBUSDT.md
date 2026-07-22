# Khả thi giao dịch theo horizon — BNBUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — phân tích khả thi chi phí, không phải bằng chứng có lợi nhuận.

- Feature file: `features_BNBUSDT_multiday.parquet` (4,000,000 dòng)
- Git commit: `3a532125b1a1a3043f109798feb61e40939c7d2d` · Python 3.14.5
- Half-spread **đo được** từ dữ liệu: 0.669 bps (|price − mid| / mid)
- Nhịp giao dịch: 12.8 tick/giây

## Chi phí một vòng mua-bán

| Kịch bản | Chi phí vòng |
|---|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | +11.34 bps |
| Taker, 4bps/chiều (Binance futures base) | +9.34 bps |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | +2.66 bps |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | -1.34 bps |

## Độ chính xác hướng cần có để hoà vốn

`p > 0,5 + chi_phí / (2·E|biến động|)`. Ô ✅ nghĩa là ngưỡng hoà vốn nằm dưới 60% — mức còn có thể bàn tới. `—` nghĩa là không thể hoà vốn dù dự đoán đúng 100%. `⚠ spread` nghĩa là spread thu được lớn hơn phí, khiến công thức trên kết luận có lãi ở **mọi** độ chính xác; đó là ảo giác của mô hình vì nó **chưa tính adverse selection** — lệnh chờ có xu hướng được khớp đúng lúc thị trường đi ngược lại bạn. Muốn dùng kịch bản đó phải mô hình hoá hàng đợi lệnh và tỉ lệ khớp bằng dữ liệu L2 thật.

| Horizon | Thời gian | E\|biến động\| | Taker 5bps | Taker 4bps | Maker 2bps | Maker 0bps |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1 giây | 2.47e-06 | — | — | — | ⚠ spread |
| 10 | 0.8 giây | 1.76e-05 | — | — | — | ⚠ spread |
| 100 | 7.8 giây | 1.28e-04 | — | — | — | ⚠ spread |
| 1,000 | 78.1 giây | 5.06e-04 | — | — | 76.3% | ⚠ spread |
| 5,000 | 6.5 phút | 1.17e-03 | 98.6% | 90.0% | 61.4% | ⚠ spread |
| 10,000 | 13.0 phút | 1.67e-03 | 84.0% | 78.0% | 58.0% ✅ | ⚠ spread |
| 50,000 | 65.1 phút | 3.94e-03 | 64.4% | 61.8% | 53.4% ✅ | ⚠ spread |
| 100,000 | 2.2 giờ | 5.22e-03 | 60.9% | 59.0% ✅ | 52.6% ✅ | ⚠ spread |
| 200,000 | 4.3 giờ | 7.20e-03 | 57.9% ✅ | 56.5% ✅ | 51.8% ✅ | ⚠ spread |

## Horizon nhỏ nhất còn giao dịch được

| Kịch bản | Horizon | Thời gian |
|---|---:|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | 200,000 tick | 4.3 giờ |
| Taker, 4bps/chiều (Binance futures base) | 100,000 tick | 2.2 giờ |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | 10,000 tick | 13.0 phút |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | không có | — |

## Kết luận

Ở horizon 1 tick, biến động trung bình chỉ bằng **0.0022 lần** chi phí một vòng taker — nghĩa là **một mô hình dự đoán đúng 100% vẫn lỗ**. Đây là giới hạn của horizon, không phải của mô hình. Đặt lệnh chờ (maker, 2bps/chiều) trở nên khả thi từ **10,000 tick (~13.0 phút)**. Giao dịch chủ động (taker, 5bps/chiều) cần tới **200,000 tick (~4.3 giờ)**. Cần nhấn mạnh: vượt ngưỡng chi phí là điều kiện **cần, không đủ**. Bảng này chỉ nói biến động đủ lớn để trả phí; nó **không** nói dự án có khả năng dự đoán đúng hướng ở horizon đó — điều chưa từng được chứng minh.
