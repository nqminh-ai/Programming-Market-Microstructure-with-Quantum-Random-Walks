# Khả thi giao dịch theo horizon — BNBUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — phân tích khả thi chi phí, không phải bằng chứng có lợi nhuận.

- Feature file: `features_BNBUSDT_69d.parquet` (54,096,780 dòng)
- Git commit: `2cef6a187e665bca3d1143b6861453abf40cd969` · Python 3.14.5
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
| 1 | 0.1 giây | 3.95e-06 | — | — | — | 84.3% |
| 10 | 1.1 giây | 2.79e-05 | — | — | — | — |
| 100 | 11.0 giây | 1.74e-04 | — | — | — | 99.5% |
| 1,000 | 1.8 phút | 6.38e-04 | — | — | 97.7% | 66.3% |
| 5,000 | 9.2 phút | 1.45e-03 | 84.7% | 77.8% | 71.4% | 57.7% ✅ |
| 10,000 | 18.4 phút | 2.06e-03 | 74.5% | 69.6% | 65.2% | 55.5% ✅ |
| 50,000 | 1.5 giờ | 4.65e-03 | 60.8% | 58.7% ✅ | 57.2% ✅ | 52.9% ✅ |
| 100,000 | 3.1 giờ | 6.58e-03 | 57.6% ✅ | 56.1% ✅ | 55.1% ✅ | 52.1% ✅ |
| 200,000 | 6.1 giờ | 9.36e-03 | 55.4% ✅ | 54.3% ✅ | 53.6% ✅ | 51.5% ✅ |

## Horizon nhỏ nhất còn giao dịch được

| Kịch bản | Horizon | Thời gian |
|---|---:|---:|
| Taker, 5bps/chiều (mức signal engine đang dùng) | 100,000 tick | 3.1 giờ |
| Taker, 4bps/chiều (Binance futures base) | 50,000 tick | 1.5 giờ |
| Maker, 2bps/chiều (đặt lệnh chờ, ăn spread) | 50,000 tick | 1.5 giờ |
| Maker, 0bps/chiều (bậc phí ưu đãi nhất) | 5,000 tick | 9.2 phút |

## Kết luận

Ở horizon 1 tick, biến động trung bình chỉ bằng **0.0039 lần** chi phí một vòng taker — nghĩa là **một mô hình dự đoán đúng 100% vẫn lỗ**. Đây là giới hạn của horizon, không phải của mô hình. Đặt lệnh chờ (maker, 2bps/chiều) trở nên khả thi từ **50,000 tick (~1.5 giờ)**. Giao dịch chủ động (taker, 5bps/chiều) cần tới **100,000 tick (~3.1 giờ)**. Cần nhấn mạnh: vượt ngưỡng chi phí là điều kiện **cần, không đủ**. Bảng này chỉ nói biến động đủ lớn để trả phí; nó **không** nói dự án có khả năng dự đoán đúng hướng ở horizon đó — điều chưa từng được chứng minh.
