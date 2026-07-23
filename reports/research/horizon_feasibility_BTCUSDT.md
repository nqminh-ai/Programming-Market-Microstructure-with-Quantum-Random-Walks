# Khả thi giao dịch theo horizon — BTCUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — phân tích khả thi chi phí, không phải bằng chứng có lợi nhuận.

- Feature file: `features_BTCUSDT_69d.parquet` (227,586,094 dòng)
- Git commit: `9adde323946b06d11d80bff4cd1b0a0419595aed` · Python 3.14.5
- Half-spread **đo được** từ dữ liệu: 0.239 bps (|price − mid| / mid)
- Nhịp giao dịch: 38.2 tick/giây

## Chi phí một vòng mua-bán

| Kịch bản | Chi phí vòng |
|---|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | +10.48 bps |
| Taker, 4bps/chiều (Binance futures base) | +8.48 bps |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | +3.52 bps |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | -0.48 bps |

## Độ chính xác hướng cần có để hoà vốn

`p > 0,5 + chi_phí / (2·E|biến động|)`. Ô ✅ nghĩa là ngưỡng hoà vốn nằm dưới 60% — mức còn có thể bàn tới. `—` nghĩa là không thể hoà vốn dù dự đoán đúng 100%. `⚠ spread` nghĩa là spread thu được lớn hơn phí, khiến công thức trên kết luận có lãi ở **mọi** độ chính xác; đó là ảo giác của mô hình vì nó **chưa tính adverse selection** — lệnh chờ có xu hướng được khớp đúng lúc thị trường đi ngược lại bạn. Muốn dùng kịch bản đó phải mô hình hoá hàng đợi lệnh và tỉ lệ khớp bằng dữ liệu L2 thật.

| Horizon | Thời gian | E\|biến động\| | Taker 5bps | Taker 4bps | Maker 2bps | Maker 0bps |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 giây | 6.22e-07 | — | — | — | ⚠ spread |
| 10 | 0.3 giây | 5.79e-06 | — | — | — | ⚠ spread |
| 100 | 2.6 giây | 5.30e-05 | — | — | — | ⚠ spread |
| 1,000 | 26.2 giây | 2.79e-04 | — | — | — | ⚠ spread |
| 5,000 | 2.2 phút | 6.68e-04 | — | — | 76.4% | ⚠ spread |
| 10,000 | 4.4 phút | 9.50e-04 | — | 94.6% | 68.5% | ⚠ spread |
| 50,000 | 21.8 phút | 2.11e-03 | 74.8% | 70.1% | 58.3% ✅ | ⚠ spread |
| 100,000 | 43.7 phút | 2.99e-03 | 67.5% | 64.2% | 55.9% ✅ | ⚠ spread |
| 200,000 | 87.3 phút | 4.19e-03 | 62.5% | 60.1% | 54.2% ✅ | ⚠ spread |

## Horizon nhỏ nhất còn giao dịch được

| Kịch bản | Horizon | Thời gian |
|---|---:|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | không có | — |
| Taker, 4bps/chiều (Binance futures base) | không có | — |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | 50,000 tick | 21.8 phút |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | không có | — |

## Kết luận

Ở horizon 1 tick, biến động trung bình chỉ bằng **0.0006 lần** chi phí một vòng taker — nghĩa là **một mô hình dự đoán đúng 100% vẫn lỗ**. Đây là giới hạn của horizon, không phải của mô hình. Đặt lệnh chờ (maker, 2bps/chiều) trở nên khả thi từ **50,000 tick (~21.8 phút)**. Ở mức phí taker 5bps/chiều, **không horizon nào** trong lưới đạt ngưỡng khả thi. Cần nhấn mạnh: vượt ngưỡng chi phí là điều kiện **cần, không đủ**. Bảng này chỉ nói biến động đủ lớn để trả phí; nó **không** nói dự án có khả năng dự đoán đúng hướng ở horizon đó — điều chưa từng được chứng minh.
