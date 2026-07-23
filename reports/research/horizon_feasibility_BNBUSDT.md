# Khả thi giao dịch theo horizon — BNBUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — phân tích khả thi chi phí, không phải bằng chứng có lợi nhuận.

- Feature file: `features_BNBUSDT_69d.parquet` (54,096,780 dòng)
- Git commit: `9adde323946b06d11d80bff4cd1b0a0419595aed` · Python 3.14.5
- Half-spread **đo được** từ dữ liệu: 0.958 bps (|price − mid| / mid)
- Nhịp giao dịch: 9.1 tick/giây

## Chi phí một vòng mua-bán

| Kịch bản | Chi phí vòng |
|---|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | +11.92 bps |
| Taker, 4bps/chiều (Binance futures base) | +9.92 bps |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | +2.08 bps |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | -1.92 bps |

## Độ chính xác hướng cần có để hoà vốn

`p > 0,5 + chi_phí / (2·E|biến động|)`. Ô ✅ nghĩa là ngưỡng hoà vốn nằm dưới 60% — mức còn có thể bàn tới. `—` nghĩa là không thể hoà vốn dù dự đoán đúng 100%. `⚠ spread` nghĩa là spread thu được lớn hơn phí, khiến công thức trên kết luận có lãi ở **mọi** độ chính xác; đó là ảo giác của mô hình vì nó **chưa tính adverse selection** — lệnh chờ có xu hướng được khớp đúng lúc thị trường đi ngược lại bạn. Muốn dùng kịch bản đó phải mô hình hoá hàng đợi lệnh và tỉ lệ khớp bằng dữ liệu L2 thật.

| Horizon | Thời gian | E\|biến động\| | Taker 5bps | Taker 4bps | Maker 2bps | Maker 0bps |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1 giây | 3.95e-06 | — | — | — | ⚠ spread |
| 10 | 1.1 giây | 2.79e-05 | — | — | — | ⚠ spread |
| 100 | 11.0 giây | 1.74e-04 | — | — | — | ⚠ spread |
| 1,000 | 1.8 phút | 6.38e-04 | — | — | 66.3% | ⚠ spread |
| 5,000 | 9.2 phút | 1.45e-03 | 91.1% | 84.2% | 57.2% ✅ | ⚠ spread |
| 10,000 | 18.4 phút | 2.06e-03 | 79.0% | 74.1% | 55.1% ✅ | ⚠ spread |
| 50,000 | 1.5 giờ | 4.65e-03 | 62.8% | 60.7% | 52.2% ✅ | ⚠ spread |
| 100,000 | 3.1 giờ | 6.58e-03 | 59.1% ✅ | 57.5% ✅ | 51.6% ✅ | ⚠ spread |
| 200,000 | 6.1 giờ | 9.36e-03 | 56.4% ✅ | 55.3% ✅ | 51.1% ✅ | ⚠ spread |

## Horizon nhỏ nhất còn giao dịch được

| Kịch bản | Horizon | Thời gian |
|---|---:|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | 100,000 tick | 3.1 giờ |
| Taker, 4bps/chiều (Binance futures base) | 100,000 tick | 3.1 giờ |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | 5,000 tick | 9.2 phút |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | không có | — |

## Kết luận

Ở horizon 1 tick, biến động trung bình chỉ bằng **0.0033 lần** chi phí một vòng taker — nghĩa là **một mô hình dự đoán đúng 100% vẫn lỗ**. Đây là giới hạn của horizon, không phải của mô hình. Đặt lệnh chờ (maker, 2bps/chiều) trở nên khả thi từ **5,000 tick (~9.2 phút)**. Giao dịch chủ động (taker, 5bps/chiều) cần tới **100,000 tick (~3.1 giờ)**. Cần nhấn mạnh: vượt ngưỡng chi phí là điều kiện **cần, không đủ**. Bảng này chỉ nói biến động đủ lớn để trả phí; nó **không** nói dự án có khả năng dự đoán đúng hướng ở horizon đó — điều chưa từng được chứng minh.
