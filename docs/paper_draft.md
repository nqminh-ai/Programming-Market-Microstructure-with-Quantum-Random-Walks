# Đánh giá thăm dò Quantum Random Walk trong vi cấu trúc tiền mã hóa

**Trạng thái:** bản thảo phương pháp; chưa đủ điều kiện gửi công bố.

## Tóm tắt

Nghiên cứu này xây dựng một quantum random walk rời rạc với density matrix,
coin thích nghi theo proxy mất cân bằng luồng giao dịch và decoherence phụ
thuộc cường độ giao dịch. Mô hình được so sánh với ba classical random walk,
GARCH(1,1) và GBM dưới cùng một giao thức thời gian.

QRW tạo ra phân phối biên fixed-origin tại từng horizon; các draw giữa horizon
không phải trajectory chung. Vì vậy đánh giá QRW chỉ dùng proper marginal
scores và marginal variance scaling. ACF và tail diagnostics chỉ áp dụng cho
model có trajectory thật; Diebold–Mariano dùng loss rolling-origin một bước.

Dữ liệu hoạt động hiện chỉ gồm 1.908 tick BTCUSDT trong khoảng 118,5 giây và
biến OBI là trade-flow proxy, không phải L2 LOB. Walk-forward pooled cho thấy
QRW kém baseline affine. Do đó nghiên cứu chưa hỗ trợ tuyên bố QRW vượt trội.

## Phương pháp

Density matrix tiến hóa qua coin-shift và kênh dephasing. OBI tương lai trong
dự báo nhiều bước được ngoại suy bằng AR(1) fit hoàn toàn trên train. Mỗi
horizon được đo độc lập từ trạng thái lượng tử chưa bị collapse ở horizon
trước.

Endpoint chính đăng ký trước là mean marginal CRPS. Directional log loss chỉ
là tie-break. AIC/BIC chỉ được so sánh trong cùng họ likelihood.

## Heavy-tail

Prototype `qrw_heavy_tail.py` hiện dùng bước nhảy Bernoulli/Pareto cổ điển. Nó
không phải heavy-tailed unitary shift, nên bị loại khỏi tuyên bố về cơ chế QRW.
Không có bằng chứng hợp lệ cho khoảng tail index 1,1–2,5, cải thiện bootstrap
scorecard hoặc tính bền vững theo Diebold–Mariano.

## Phạm vi bằng chứng

Các file theo ngày của BTCUSDT, ETHUSDT và BNBUSDT tồn tại trong workspace,
nhưng benchmark cross-asset trước đây chỉ chấm ngày cuối. Không được mô tả kết
quả đó là kiểm định 31 ngày. Một nghiên cứu confirmatory mới cần ít nhất 20
ngày UTC untouched cho mỗi tài sản, L2 LOB đồng bộ và protocol đóng băng trước
khi mở nhãn holdout.

## Kết luận

Đóng góp hiện tại là một pipeline mô phỏng và đánh giá có kiểm soát ngữ nghĩa,
không phải bằng chứng về ưu thế lượng tử. Kết quả walk-forward âm và các hạn
chế dữ liệu phải được báo cáo nguyên trạng.
