# Khả thi giao dịch theo horizon — BNBUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — phân tích khả thi chi phí, không phải bằng chứng có lợi nhuận.

- Feature file: `features_BNBUSDT_69d.parquet` (54,096,780 dòng)
- Git commit: `bfe50182c2da57ceac78b42ed95960a5148c8268` · Python 3.14.5
- Half-spread **ước lượng Roll (1984)**: 0.0316 bps — nghịch đảo bid-ask bounce trong chính chuỗi giá khớp
- *(Đã thay thế)* |price − mid| / mid = 0.958 bps, **gấp 30×** — `mid_price` là VWAP trượt 100 lệnh nên đại lượng này đo độ phân tán giá, không phải spread
- Nhịp giao dịch: 9.1 tick/giây

## Chi phí một vòng mua-bán

| Kịch bản | Chi phí vòng |
|---|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | +10.06 bps |
| Taker, 4bps/chiều (Binance futures base) | +8.06 bps |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | +3.94 bps |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | -0.06 bps |

## Độ chính xác hướng cần có để hoà vốn

`p > 0,5 + chi_phí / (2·E|biến động|)`. Ô ✅ nghĩa là ngưỡng hoà vốn nằm dưới 60% — mức còn có thể bàn tới. `—` nghĩa là không thể hoà vốn dù dự đoán đúng 100%. `⚠ spread` nghĩa là spread thu được lớn hơn phí, khiến công thức trên kết luận có lãi ở **mọi** độ chính xác; đó là ảo giác của mô hình vì nó **chưa tính adverse selection** — lệnh chờ có xu hướng được khớp đúng lúc thị trường đi ngược lại bạn. Muốn dùng kịch bản đó phải mô hình hoá hàng đợi lệnh và tỉ lệ khớp bằng dữ liệu L2 thật.

| Horizon | Thời gian | E\|biến động\| | Taker 5bps | Taker 4bps | Maker 2bps | Maker 0bps |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1 giây | 3.95e-06 | — | — | — | ⚠ spread |
| 10 | 1.1 giây | 2.79e-05 | — | — | — | ⚠ spread |
| 100 | 11.0 giây | 1.74e-04 | — | — | — | ⚠ spread |
| 1,000 | 1.8 phút | 6.38e-04 | — | — | 80.9% | ⚠ spread |
| 5,000 | 9.2 phút | 1.45e-03 | 84.7% | 77.8% | 63.6% | ⚠ spread |
| 10,000 | 18.4 phút | 2.06e-03 | 74.5% | 69.6% | 59.6% ✅ | ⚠ spread |
| 50,000 | 1.5 giờ | 4.65e-03 | 60.8% | 58.7% ✅ | 54.2% ✅ | ⚠ spread |
| 100,000 | 3.1 giờ | 6.58e-03 | 57.6% ✅ | 56.1% ✅ | 53.0% ✅ | ⚠ spread |
| 200,000 | 6.1 giờ | 9.36e-03 | 55.4% ✅ | 54.3% ✅ | 52.1% ✅ | ⚠ spread |

## Horizon nhỏ nhất còn giao dịch được

| Kịch bản | Horizon | Thời gian |
|---|---:|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | 100,000 tick | 3.1 giờ |
| Taker, 4bps/chiều (Binance futures base) | 50,000 tick | 1.5 giờ |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | 10,000 tick | 18.4 phút |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | không có | — |

## Kết luận

Ở horizon 1 tick, biến động trung bình chỉ bằng **0.0039 lần** chi phí một vòng taker — nghĩa là **một mô hình dự đoán đúng 100% vẫn lỗ**. Đây là giới hạn của horizon, không phải của mô hình. Đặt lệnh chờ (maker, 2bps/chiều) trở nên khả thi từ **10,000 tick (~18.4 phút)**. Giao dịch chủ động (taker, 5bps/chiều) cần tới **100,000 tick (~3.1 giờ)**. Cần nhấn mạnh: vượt ngưỡng chi phí là điều kiện **cần, không đủ**. Bảng này chỉ nói biến động đủ lớn để trả phí; nó **không** nói dự án có khả năng dự đoán đúng hướng ở horizon đó — điều chưa từng được chứng minh.
