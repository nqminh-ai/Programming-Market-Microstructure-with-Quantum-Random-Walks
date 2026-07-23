# Khả thi giao dịch theo horizon — BTCUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — phân tích khả thi chi phí, không phải bằng chứng có lợi nhuận.

- Feature file: `features_BTCUSDT_69d.parquet` (227,586,094 dòng)
- Git commit: `bfe50182c2da57ceac78b42ed95960a5148c8268` · Python 3.14.5
- Half-spread **ước lượng Roll (1984)**: 0.0073 bps — nghịch đảo bid-ask bounce trong chính chuỗi giá khớp
- *(Đã thay thế)* |price − mid| / mid = 0.239 bps, **gấp 33×** — `mid_price` là VWAP trượt 100 lệnh nên đại lượng này đo độ phân tán giá, không phải spread
- Nhịp giao dịch: 38.2 tick/giây

## Chi phí một vòng mua-bán

| Kịch bản | Chi phí vòng |
|---|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | +10.01 bps |
| Taker, 4bps/chiều (Binance futures base) | +8.01 bps |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | +3.99 bps |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | -0.01 bps |

## Độ chính xác hướng cần có để hoà vốn

`p > 0,5 + chi_phí / (2·E|biến động|)`. Ô ✅ nghĩa là ngưỡng hoà vốn nằm dưới 60% — mức còn có thể bàn tới. `—` nghĩa là không thể hoà vốn dù dự đoán đúng 100%. `⚠ spread` nghĩa là spread thu được lớn hơn phí, khiến công thức trên kết luận có lãi ở **mọi** độ chính xác; đó là ảo giác của mô hình vì nó **chưa tính adverse selection** — lệnh chờ có xu hướng được khớp đúng lúc thị trường đi ngược lại bạn. Muốn dùng kịch bản đó phải mô hình hoá hàng đợi lệnh và tỉ lệ khớp bằng dữ liệu L2 thật.

| Horizon | Thời gian | E\|biến động\| | Taker 5bps | Taker 4bps | Maker 2bps | Maker 0bps |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 giây | 6.22e-07 | — | — | — | ⚠ spread |
| 10 | 0.3 giây | 5.79e-06 | — | — | — | ⚠ spread |
| 100 | 2.6 giây | 5.30e-05 | — | — | — | ⚠ spread |
| 1,000 | 26.2 giây | 2.79e-04 | — | — | — | ⚠ spread |
| 5,000 | 2.2 phút | 6.68e-04 | — | — | 79.8% | ⚠ spread |
| 10,000 | 4.4 phút | 9.50e-04 | — | 92.2% | 71.0% | ⚠ spread |
| 50,000 | 21.8 phút | 2.11e-03 | 73.7% | 69.0% | 59.4% ✅ | ⚠ spread |
| 100,000 | 43.7 phút | 2.99e-03 | 66.7% | 63.4% | 56.7% ✅ | ⚠ spread |
| 200,000 | 87.3 phút | 4.19e-03 | 61.9% | 59.6% ✅ | 54.8% ✅ | ⚠ spread |

## Horizon nhỏ nhất còn giao dịch được

| Kịch bản | Horizon | Thời gian |
|---|---:|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | không có | — |
| Taker, 4bps/chiều (Binance futures base) | 200,000 tick | 87.3 phút |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | 50,000 tick | 21.8 phút |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | không có | — |

## Kết luận

Ở horizon 1 tick, biến động trung bình chỉ bằng **0.0006 lần** chi phí một vòng taker — nghĩa là **một mô hình dự đoán đúng 100% vẫn lỗ**. Đây là giới hạn của horizon, không phải của mô hình. Đặt lệnh chờ (maker, 2bps/chiều) trở nên khả thi từ **50,000 tick (~21.8 phút)**. Ở mức phí taker 5bps/chiều, **không horizon nào** trong lưới đạt ngưỡng khả thi. Cần nhấn mạnh: vượt ngưỡng chi phí là điều kiện **cần, không đủ**. Bảng này chỉ nói biến động đủ lớn để trả phí; nó **không** nói dự án có khả năng dự đoán đúng hướng ở horizon đó — điều chưa từng được chứng minh.
