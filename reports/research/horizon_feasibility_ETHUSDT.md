# Khả thi giao dịch theo horizon — ETHUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — phân tích khả thi chi phí, không phải bằng chứng có lợi nhuận.

- Feature file: `features_ETHUSDT_69d.parquet` (211,995,471 dòng)
- Git commit: `bfe50182c2da57ceac78b42ed95960a5148c8268` · Python 3.14.5
- Half-spread **ước lượng Roll (1984)**: 0.0210 bps — nghịch đảo bid-ask bounce trong chính chuỗi giá khớp
- *(Đã thay thế)* |price − mid| / mid = 0.454 bps, **gấp 22×** — `mid_price` là VWAP trượt 100 lệnh nên đại lượng này đo độ phân tán giá, không phải spread
- Nhịp giao dịch: 35.6 tick/giây

## Chi phí một vòng mua-bán

| Kịch bản | Chi phí vòng |
|---|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | +10.04 bps |
| Taker, 4bps/chiều (Binance futures base) | +8.04 bps |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | +3.96 bps |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | -0.04 bps |

## Độ chính xác hướng cần có để hoà vốn

`p > 0,5 + chi_phí / (2·E|biến động|)`. Ô ✅ nghĩa là ngưỡng hoà vốn nằm dưới 60% — mức còn có thể bàn tới. `—` nghĩa là không thể hoà vốn dù dự đoán đúng 100%. `⚠ spread` nghĩa là spread thu được lớn hơn phí, khiến công thức trên kết luận có lãi ở **mọi** độ chính xác; đó là ảo giác của mô hình vì nó **chưa tính adverse selection** — lệnh chờ có xu hướng được khớp đúng lúc thị trường đi ngược lại bạn. Muốn dùng kịch bản đó phải mô hình hoá hàng đợi lệnh và tỉ lệ khớp bằng dữ liệu L2 thật.

| Horizon | Thời gian | E\|biến động\| | Taker 5bps | Taker 4bps | Maker 2bps | Maker 0bps |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 giây | 1.25e-06 | — | — | — | ⚠ spread |
| 10 | 0.3 giây | 1.06e-05 | — | — | — | ⚠ spread |
| 100 | 2.8 giây | 8.77e-05 | — | — | — | ⚠ spread |
| 1,000 | 28.1 giây | 3.92e-04 | — | — | — | ⚠ spread |
| 5,000 | 2.3 phút | 9.05e-04 | — | 94.4% | 71.9% | ⚠ spread |
| 10,000 | 4.7 phút | 1.29e-03 | 89.0% | 81.3% | 65.4% | ⚠ spread |
| 50,000 | 23.4 phút | 2.84e-03 | 67.7% | 64.2% | 57.0% ✅ | ⚠ spread |
| 100,000 | 46.9 phút | 4.00e-03 | 62.5% | 60.0% | 54.9% ✅ | ⚠ spread |
| 200,000 | 1.6 giờ | 5.67e-03 | 58.9% ✅ | 57.1% ✅ | 53.5% ✅ | ⚠ spread |

## Horizon nhỏ nhất còn giao dịch được

| Kịch bản | Horizon | Thời gian |
|---|---:|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | 200,000 tick | 1.6 giờ |
| Taker, 4bps/chiều (Binance futures base) | 200,000 tick | 1.6 giờ |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | 50,000 tick | 23.4 phút |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | không có | — |

## Kết luận

Ở horizon 1 tick, biến động trung bình chỉ bằng **0.0012 lần** chi phí một vòng taker — nghĩa là **một mô hình dự đoán đúng 100% vẫn lỗ**. Đây là giới hạn của horizon, không phải của mô hình. Đặt lệnh chờ (maker, 2bps/chiều) trở nên khả thi từ **50,000 tick (~23.4 phút)**. Giao dịch chủ động (taker, 5bps/chiều) cần tới **200,000 tick (~1.6 giờ)**. Cần nhấn mạnh: vượt ngưỡng chi phí là điều kiện **cần, không đủ**. Bảng này chỉ nói biến động đủ lớn để trả phí; nó **không** nói dự án có khả năng dự đoán đúng hướng ở horizon đó — điều chưa từng được chứng minh.
