# Đánh giá thăm dò Quantum Random Walk trong vi cấu trúc tiền mã hóa

**Trạng thái:** bản thảo phương pháp; **exploratory, chưa confirmatory**; chưa đủ
điều kiện gửi công bố. Mọi con số dưới đây có artifact JSON + SHA-256 tương ứng
trong [`reports/research/`](../reports/research/); prose được ràng buộc khớp
artifact bằng `tests/test_report_numbers.py`.

## Tóm tắt

Nghiên cứu này xây dựng một quantum random walk (QRW) rời rạc trên density
matrix, với coin thích nghi theo proxy mất cân bằng luồng giao dịch (OBI) và
kênh decoherence phụ thuộc cường độ giao dịch. Mô hình được so sánh với ba
classical random walk, GARCH(1,1) và GBM dưới cùng một giao thức thời gian, trên
**493,7 triệu tick** trải **69 ngày UTC trùng khớp** cho ba tài sản (BTCUSDT
227,6M dòng, ETHUSDT 212,0M, BNBUSDT 54,1M; 2026-05-13 → 2026-07-20).

Kết luận chính: **không có bằng chứng cho một lợi thế dự báo bền vững của QRW, và
cơ chế lượng tử (pha) đóng góp bằng 0.** Tham số pha `alpha_phase` hội tụ về ≈ 0
(bậc 10⁻⁵) trên cả ba tài sản, nên bất kỳ phần "không thua" nào cũng đến từ hệ số
decoherence thấp và windowing cổ điển, không phải giao thoa lượng tử.

QRW tạo ra phân phối biên fixed-origin tại từng horizon; các draw giữa horizon
không phải trajectory chung. Vì vậy đánh giá QRW chỉ dùng proper marginal scores
và marginal variance scaling. ACF và tail diagnostics chỉ áp dụng cho model có
trajectory thật; Diebold–Mariano dùng loss rolling-origin một bước.

OBI là **trade-flow proxy**, không phải L2 LOB thật — giới hạn nền tảng chưa gỡ
được vì cần ≥20 ngày UTC thu L2 thời gian thực.

## Endpoint hướng (directional Brier)

Một lỗi fit/predict mismatch từng khiến walk-forward pooled đánh giá QRW qua một
công thức cổ điển bỏ sót tham số pha (`alpha_phase`) mà bước calibrate có thể
chọn, tạo ra "lợi thế" giả **+0,049889**. Sau khi truy ra và sửa, chạy lại trên
**toàn bộ 32,4 triệu tick BTCUSDT** (không còn giới hạn bộ nhớ nhờ nạp theo cột +
downcast) cho edge QRW−affine = **−0,013091** (folds=3) và **−0,012771**
(folds=5) — QRW thắng baseline affine, KI 95% nằm trọn bên âm. Kết quả **tái lập**
trên 100 triệu tick từ giai đoạn khác: −0,015280 / −0,014766 / −0,013758 theo
folds 2/3/5. Quy luật edge co lại khi tăng số fold được giữ nguyên.

Nhưng lợi thế mảnh đó **không sống sót trước baseline cổ điển mạnh**. Đấu cùng
feature causal với OrderFlow AR(5), Logistic L2 + Pairwise và Hawkes,
windowed-QRW thua có ý nghĩa thống kê trên **cả ba** tài sản (BTC hạng 4/7, BNB
4/7, ETH **xếp chót 7/7**). OrderFlow AR(5) — một logistic đơn giản trên hướng
tick trễ — thắng QRW ở mọi tài sản.

## Endpoint chính đăng-ký-trước (marginal CRPS)

Endpoint PRIMARY là mean fixed-origin marginal CRPS, đo bằng `BenchmarkSuite`
(protocol v4). Trên 5 window không chồng lấp mỗi tài sản, QRW **không cạnh tranh
đồng đều**: hạng 1/6 trên ETH, 3/6 trên BTC, 4/6 trên BNB. Đây là chiều QRW thua
**ít dứt khoát nhất**, không phải chiều QRW thắng.

Kiểm tra robustness trên 40 window/tài sản (dữ liệu 69 ngày) cho thấy **thứ hạng
CRPS không ổn định** giữa các thiết lập: BNB nhảy 4/6 → 1/6, còn BTC (3/6) và ETH
(1/6) tái lập. Điều này củng cố "không có lợi thế nhất quán" theo một hướng khác
với bảng 5-window, chứ không lật nó.

Directional log loss chỉ là tie-break. AIC/BIC chỉ so trong cùng họ likelihood.

## Tuyên bố volatility (không xác lập được)

Bản trước giải thích kết quả CRPS bằng "QRW thua đậm ở window biến động cao vì
không mô hình hóa volatility" — đọc từ 5 window/tài sản. Đo lại bằng tương quan
Spearman giữa realised volatility của window và khoảng cách CRPS **tương đối** của
QRW trên **40 window/tài sản**: tương quan chạy đúng chiều đã khẳng định trên cả
ba tài sản (rho = +0,222 / +0,047 / +0,213) nhưng không tài sản nào tự đạt ý nghĩa
thống kê, và hai cách gộp nằm **hai bên** α = 0,05 (Fisher p = 0,072, Stouffer
p = 0,043). Kết luận: **dữ liệu nghiêng về chiều đó nhưng chưa xác lập được** như
một phát hiện.

## Khả thi giao dịch

Ngay cả khi có dự báo tốt hơn cũng chưa giao dịch được. Với horizon `h`, một cược
hướng đúng `p` phần trăm thu về `(2p−1)·E|r_h|` trước phí, nên hòa vốn đòi
`p > 0,5 + chi_phí/(2·E|r_h|)`. Ở horizon dự án đang dùng (1 tick), phí một vòng
taker lớn hơn biên độ giá kỳ vọng tới **1.610 lần** (BTC), đẩy ngưỡng hòa vốn
**vượt 100%**: một mô hình dự đoán đúng *hoàn hảo* vẫn lỗ. Không horizon nào trên
bất kỳ tài sản nào vượt ngưỡng hòa vốn, kể cả ở mức phí maker.

Half-spread đo bằng estimator Roll (1984), không phải độ phân tán quanh VWAP như
một phân tích trước từng nhầm (phóng đại 22–33×). Realised half-spread **âm** trên
mọi horizon và mọi tài sản (~−1,2 bps): lệnh chờ **trả tiền** chứ không ăn spread,
nên lợi thế maker giả định trước đó **sai cả dấu**.

## Heavy-tail

Prototype `qrw_heavy_tail.py` hiện dùng bước nhảy Bernoulli/Pareto cổ điển. Nó
không phải heavy-tailed unitary shift, nên bị loại khỏi tuyên bố về cơ chế QRW.
Không có bằng chứng hợp lệ cho khoảng tail index 1,1–2,5, cải thiện bootstrap
scorecard hay tính bền vững theo Diebold–Mariano.

## Phạm vi bằng chứng và điều kiện confirmatory

Toàn bộ Phase 1–6 là **exploratory**. Protocol confirmatory đã viết, đóng băng và
pre-register (endpoint chính, chia train/val/test theo ngày UTC, xử lý gap), có
runner + test hard-code, nhưng **không kết quả nào được gắn nhãn confirmatory** —
một lựa chọn có chủ đích, cưỡng chế bằng code — cho tới khi có ≥20 ngày UTC L2 LOB
untouched cho mỗi tài sản. Đây là lý do các giới hạn #1 (OBI proxy), #2
(confirmatory) và #6 (xác suất khớp lệnh maker) vẫn mở.

## Kết luận

Đóng góp không nằm ở "QRW thắng thị trường" mà ở một pipeline mô phỏng và đánh giá
đủ chặt để **phát hiện rằng model của chính mình không thắng** — và làm việc đó
**sáu lần**, mỗi lần đều công bố thay vì giấu: một bug biến −0,0131 thành +0,0499;
một artifact BNB không tái lập, chạy lại thì xấu hơn; ba lỗi khiến chiến lược demo
trông có lãi, sửa xong lỗ 4,2%; một "half-spread" đo nhầm đại lượng khác (sai
22–33×); giả định lệnh chờ ăn spread hóa ra sai dấu; và lời giải thích volatility
của chính dự án không đứng vững khi đo trên 40 window. Không lỗi nào do người
ngoài chỉ ra; cả sáu đều làm kết quả xấu đi. Cả kết quả lẫn hạn chế phải được báo
cáo nguyên trạng cho tới khi một đánh giá confirmatory đầy đủ được thực hiện.
