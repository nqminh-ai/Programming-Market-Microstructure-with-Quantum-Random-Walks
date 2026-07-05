# Báo cáo cuối: Quantum Random Walk cho vi cấu trúc thị trường

## Trạng thái khoa học

**Kết luận hiện tại: chưa có bằng chứng cho thấy QRW vượt trội hơn các baseline
cổ điển.** Kết quả này phải được giữ nguyên kể cả khi một lần chia holdout đơn
lẻ cho kết quả thuận lợi hơn.

Báo cáo Phase 4/5 cũ dùng protocol v2 đã bị vô hiệu hóa. Mã nguồn hiện dùng
`fixed_origin_marginal_density_matrix_ar1_obi_v4`; vì vậy mọi bảng điểm, biểu
đồ và PDF cũ chưa được tái tạo bằng protocol v4 đều không được dùng làm bằng
chứng.

## 1. Mục tiêu

Dự án kiểm tra liệu một quantum random walk rời rạc, có coin thích nghi và
decoherence, có mô tả hữu ích phân phối giá theo horizon hay không. Đây là mô
hình toán học lấy cảm hứng từ quantum walk, không phải tuyên bố thị trường vận
hành theo cơ học lượng tử.

Baseline gồm CRW đơn giản, CRW lệch, CRW tương quan, GARCH(1,1) và GBM. Tất cả
phải dùng cùng mốc train/holdout và cùng dữ liệu có sẵn tại thời điểm dự báo.

## 2. Ngữ nghĩa phép đo đã sửa

QRW tiến hóa density matrix qua từng bước và chỉ được đo như **phân phối biên
fixed-origin** tại mỗi horizon. Các mẫu ở hai horizon khác nhau không tạo thành
một trajectory chung.

Do đó:

- không lấy `diff` giữa hai cột marginal QRW;
- không tính ACF hoặc tail index từ marginal QRW;
- không vẽ từng hàng marginal như một sample path;
- không chạy Diebold–Mariano trên chuỗi loss fixed-origin;
- chỉ dùng CRPS biên làm endpoint chính và directional log loss làm tiêu chí
  phụ khi hòa;
- Diebold–Mariano chỉ dùng loss rolling-origin một bước, được căn chỉnh theo
  cùng timestamp.

Các thống kê ACF và tail vẫn có thể được xuất cho baseline có trajectory thật,
nhưng phải ghi rõ `trajectory_only` và không được đưa QRW vào so sánh đó.

## 3. Dữ liệu thực tế

Artifact BTCUSDT đang hoạt động chỉ có **1.908 tick**, bao phủ khoảng **118,5
giây** trong ngày 12-06-2026. Quy mô này không đủ để xác nhận tính ổn định theo
ngày, theo chế độ thị trường hoặc theo tài sản.

Biến `obi` hiện tại là **proxy trade-flow imbalance** suy ra từ giao dịch, không
phải order-book imbalance từ L2 limit order book đồng bộ. Vì vậy báo cáo không
được gọi dữ liệu hiện tại là LOB thật.

Dữ liệu BTCUSDT, ETHUSDT và BNBUSDT có nhiều file theo ngày, nhưng đánh giá
cross-asset trước đây chỉ chọn ngày cuối. Nó không chứng minh được độ bền trên
31 ngày.

## 4. Kiểm soát rò rỉ dữ liệu

- Calibration và xác suất di chuyển chỉ dùng warmup/train.
- Holdout được tách theo thời gian và không chồng lấp với train.
- OBI tương lai trong mô phỏng fixed-origin được dự báo bằng AR(1) fit trên
  train; feature holdout tương lai không được đưa vào simulator.
- Live collector tính imbalance từ lịch sử trước khi append giao dịch hiện tại,
  loại bỏ self-prediction leakage.

## 5. Kết quả kiểm toán dự báo

Walk-forward là bằng chứng chính. Trên ba fold đã kiểm toán, QRW không thắng
baseline affine ở fold nào. Chênh lệch Brier pooled theo hướng
`QRW - baseline` là khoảng **+0,049889**, với khoảng tin cậy 95% khoảng
**[+0,020831; +0,078994]**. Giá trị dương nghĩa là QRW kém hơn.

Một holdout đơn lẻ từng cho kết quả ngược lại chỉ là chẩn đoán phụ và không
được phép lật ngược kết luận walk-forward. Vì thế verdict hợp lệ là:

> QRW kém hơn baseline affine trong walk-forward pooled; chưa có bằng chứng về
> lợi thế dự báo QRW.

## 6. AIC/BIC và scorecard

AIC/BIC chỉ được xếp hạng trong cùng họ likelihood:

- Bernoulli định hướng: QRW/CRW;
- Gaussian liên tục: GARCH/GBM.

Không so sánh trực tiếp AIC/BIC giữa hai nhóm này. Scorecard không còn lấy
trung bình hạng của các metric không đồng nhất. Endpoint đăng ký trước là mean
marginal CRPS; directional log loss chỉ dùng làm tie-break. Variance scaling,
coverage và interval width là chẩn đoán, không phải bằng chứng độc lập về ưu
thế.

## 7. Mô hình heavy-tail

Module heavy-tail hiện tại là bộ sinh bước nhảy Bernoulli/Pareto cổ điển. Nó
không triển khai một heavy-tailed unitary shift và không chứng minh cải thiện
tail index của QRW. Các tuyên bố về khoảng tail index `1,1–2,5`, bootstrap CI
cải thiện hoặc DM bền vững đã bị rút lại vì chưa được hỗ trợ bởi protocol hợp
lệ.

## 8. Hạn chế chưa giải quyết

1. Chưa có ít nhất 20 ngày UTC mới, untouched, cho mỗi tài sản.
2. Chưa có L2 LOB đồng bộ với trade feed.
3. Artifact Phase 3–6 chính thức chưa được tái tạo hoàn chỉnh dưới protocol v4.
4. Dữ liệu cross-asset chưa được đánh giá theo toàn bộ ngày có sẵn.
5. Chưa thể đưa ra kết luận confirmatory hoặc production-readiness.

## 9. Kết luận

Dự án đã sửa ngữ nghĩa đo QRW và quy tắc thống kê theo hướng có thể kiểm toán,
nhưng bằng chứng thực nghiệm hiện tại vẫn là **kết quả âm**. QRW chưa vượt qua
baseline cổ điển trong walk-forward, dữ liệu hoạt động quá ngắn, và OBI chưa
phải L2 order-book imbalance.

Kết luận chỉ được cập nhật sau khi protocol được đóng băng, provenance khớp
commit/data hash, và một đánh giá mới trên dữ liệu multi-day untouched được
thực hiện đúng pre-registration.
