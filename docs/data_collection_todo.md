# Pre-registration thu thập dữ liệu confirmatory

## Trạng thái

Chưa bắt đầu. Dữ liệu hiện có chỉ dùng cho phát triển và không được tái gắn
nhãn thành confirmatory.

## Điều kiện bắt đầu

- Đóng băng protocol, code commit và dependency lock trước khi mở nhãn mới.
- Ghi SHA-256 của manifest pre-registration.
- Không thay endpoint, horizon, baseline hoặc rule loại dữ liệu sau khi xem
  kết quả holdout.

## Phạm vi dữ liệu mới

- Tài sản: BTCUSDT, ETHUSDT và BNBUSDT.
- Tối thiểu **20 ngày UTC tương lai, đầy đủ cho mỗi tài sản**, được thu sau
  commit đóng băng protocol.
- Trade feed và L2 limit-order-book snapshots phải đồng bộ timestamp.
- Lưu raw bất biến dưới `data/assets/<symbol>/raw/`.
- Ghi rõ venue, timezone, packet loss, reconnect, gap và clock drift.
- Không thay L2 OBI bằng trade-flow proxy trong tập confirmatory.

## Chia dữ liệu

- Chỉ các ngày UTC hoàn chỉnh mới là đơn vị split; không chia ngẫu nhiên tick.
- Với đúng 20 ngày hợp lệ/tài sản: 10 ngày sớm nhất train, 5 ngày tiếp theo
  validation và 5 ngày cuối untouched test. Nếu có hơn 20 ngày, chốt trước
  khi mở nhãn theo tỉ lệ ngày 50%/25%/25%, làm tròn phần dư vào test.
- Fit chỉ trên các ngày train; chọn hyperparameter trên toàn bộ ngày
  validation; báo cáo từng ngày test và tuyệt đối không refit sau khi mở test.
- Confirmatory labels chỉ được mở một lần sau khi pipeline đã tạo manifest đầu
  vào và xác nhận source tree sạch.
- Mọi tài sản dùng cùng quy tắc split và cùng endpoint đăng ký trước.

## Endpoint đăng ký trước

1. Primary: mean fixed-origin marginal CRPS trên các horizon
   `1, 5, 10, 20, 50, 100, 200, 500` khi đủ dữ liệu.
2. Tie-break: mean directional log loss trên đúng các horizon đó.
3. Diebold–Mariano: loss tuyệt đối rolling-origin một bước, căn chỉnh theo cùng
   timestamp; Newey–West lag cố định là `min(20, n - 2)` để khớp implementation.
4. Variance scaling, coverage, interval width, ACF và tail là diagnostics;
   ACF/tail không áp dụng cho QRW fixed-origin marginals.

Kết quả chính là trung bình theo ngày test. Khoảng tin cậy resample nguyên cụm
ngày UTC; báo cáo sensitivity với seed `2026, 2027, 2028` và block ngày
`1, 2, 5` khi số ngày cho phép. Mọi so sánh model/metric đã chọn được hiệu chỉnh
Benjamini–Hochberg trong một family toàn cục.

## Baseline cố định

- CRW Simple, CRW Biased, CRW Correlated.
- GARCH(1,1), GBM.
- QRW Directional Link.
- Logistic L2 với đúng năm feature: OBI, tick direction, thay đổi OBI, trị
  tuyệt đối OBI và log trade intensity.
- Logistic L2 với mọi tương tác theo cặp của năm feature.
- Liên kết phi tuyến bậc hai, hiệu chuẩn isotonic chỉ trên validation.
- OrderFlow AR(5).
- Marked Hawkes conditional-mark logit với kernel mũ theo inter-event time;
  không gọi đây là full arrival-process likelihood.

Mọi baseline dùng chung train/validation/test events. Grid L2 là
`1e-4, 1e-3, 1e-2, 1e-1`; grid decay Hawkes là `0.50, 0.80, 0.95`. Không thêm
model, feature, grid hoặc endpoint sau khi xem test.

## Provenance bắt buộc

Mỗi artifact phải ghi:

- protocol version;
- full Git commit;
- canonical feature path;
- SHA-256 của từng input/output;
- dependency lock;
- seed;
- ngày UTC và asset;
- nguồn OBI (`lob`, không phải proxy).

Nếu bất kỳ trường nào không khớp, pipeline phải hard-fail và không xuất verdict
confirmatory.
