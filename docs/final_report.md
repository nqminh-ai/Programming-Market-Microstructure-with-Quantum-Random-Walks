# Báo cáo cuối: Quantum Random Walk cho vi cấu trúc thị trường

## Trạng thái khoa học

**Kết luận hiện tại (cập nhật sau ablation Phase 1): KHÔNG có bằng chứng cho
một lợi thế dự báo bền vững của QRW, và cơ chế "giao thoa lượng tử" (tham số
pha `alpha_phase`) đóng góp bằng 0.** Một nghiên cứu ablation cô lập
([reports/research/alpha_phase_ablation.md](../reports/research/alpha_phase_ablation.md),
gắn nhãn exploratory) chỉ ra hai điều:

- Lợi thế Brier của QRW so với baseline affine **không nhất quán và không bền
  vững**: kiểm chứng cross-asset cho thấy QRW thắng ổn định trên BNB, thua ổn
  định trên ETH, và trên BTC thì mong manh — thắng ở walk-forward 2–4 fold
  nhưng **đảo dấu thành thua** ở 5–8 fold. Con số dương báo cáo trước đây
  (−0,007383, BTC 3 fold) vừa **không tổng quát sang asset khác** vừa là
  **artifact của lựa chọn số fold**.
- Nơi QRW có vẻ thắng, lợi thế đến **hoàn toàn từ windowing/decoherence, không
  phải từ pha**: model bỏ pha (`alpha_phase=0`, refit) cho Brier giống hệt
  model pha-tự-do tới 6 chữ số; ép pha lớn hơn còn làm dự báo **xấu đi đơn
  điệu**. Đóng góp biên của pha đo được là ~9×10⁻⁸ Brier (nhỏ hơn edge chính 5
  bậc độ lớn) và mang dấu bất lợi.

Kết luận trước đó ("QRW có lợi thế Brier có ý nghĩa thống kê") vì vậy bị **thu
hẹp mạnh**: lợi thế đó chỉ tồn tại ở protocol 3-fold cụ thể, không bền vững, và
không đến từ cơ chế lượng tử. Xem §5b để biết chi tiết. Các caveat cũ vẫn giữ
nguyên: dữ liệu hoạt động ngắn, OBI là proxy trade-flow, chưa chạy trên toàn bộ
dataset gốc.

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

**Lỗi fit/predict mismatch đã được phát hiện và sửa.** `calibrate()` có một
nhánh "quantum refinement" có thể chọn tham số pha `alpha_phase` (coin SU(2)
không giao hoán) khi nó thắng Brier score trên validation. Nhưng trước khi
sửa, đường dự đoán walk-forward thực tế gọi thẳng `_one_step_right_probability`
— một công thức cổ điển hoàn toàn không có tham số `alpha_phase`. Nói cách
khác: khi nhánh quantum thắng ở bước calibrate, bước dự đoán vẫn âm thầm chạy
công thức cổ điển. Verdict cũ dưới đây được tính từ pipeline có lỗi này, và vì
vậy không đáng tin cậy:

> ~~Trên ba fold đã kiểm toán, QRW không thắng baseline affine ở fold nào.
> Chênh lệch Brier pooled theo hướng `QRW - baseline` là khoảng **+0,049889**,
> với khoảng tin cậy 95% khoảng **[+0,020831; +0,078994]**.~~ (đã vô hiệu)

Đã sửa: `MarketQRW` giờ lưu `quantum_improved` như một thuộc tính instance, và
toàn bộ đường dự đoán (`simulate_price_path`, AIC/BIC cuối, predictor trong
`benchmark_suite.py`, vòng lặp walk-forward trong `phase3_overfitting_audit.py`)
dùng chung một hàm dispatch (`predict_right_probability(s)`) — luôn dùng đúng
công thức mà `calibrate()` đã chọn, thay vì mặc định về cổ điển.

**Kết quả sau khi vá:**

- Trên full dataset BTCUSDT 10 ngày (~32,4 triệu tick, cùng dataset đã tạo ra
  con số +0,049889 cũ), audit bị crash do hết bộ nhớ (máy chỉ còn ~4,4 GB free)
  ở bước `rolling_stability()` — một vấn đề hiệu năng/bộ nhớ riêng (copy
  dataframe lặp lại nhiều lần theo từng block), không liên quan tới logic vừa
  sửa. Trước khi crash, `calibrate()` đã in ra: nhánh quantum thắng validation
  (Brier 0,0715 so với 0,0769 cổ điển) — tức `quantum_improved=True` trên dữ
  liệu thật.
- Trên subset 3 ngày gần nhất (2026-07-03 → 2026-07-05, ~4 triệu tick), audit
  chạy hoàn chỉnh và `quantum_improved=True` cũng được chọn:

| | Chênh lệch Brier (QRW − baseline) | KI 95% |
|---|---:|---|
| Trước vá (full 10 ngày) | +0,049889 | [+0,020831; +0,078994] |
| Sau vá (subset 3 ngày gần nhất) | **−0,007383** | **[−0,008306; −0,006482]** |

Giá trị âm nghĩa là QRW **thắng** baseline affine, có ý nghĩa thống kê (KI 95%
hoàn toàn âm).

**Diễn giải quan trọng — không phải bằng chứng cho "giao thoa lượng tử":**
tham số `alpha_phase` mà nhánh quantum chọn được là khoảng **-8×10⁻⁶ đến
-1×10⁻⁵**, gần như bằng 0. Lợi thế quan sát được đến chủ yếu từ `gamma=0,01`
(hệ số decoherence rất thấp, gần như hoàn toàn coherent) kết hợp
`alpha_direction≈-1,03` (hệ số đảo chiều mạnh), chứ không phải từ thành phần
pha không giao hoán (SU(2) interference) mà tên "quantum refinement" ngụ ý.
Kết quả này xác nhận **cơ chế dispatch giờ hoạt động đúng như thiết kế**,
nhưng KHÔNG xác nhận "giao thoa lượng tử" là nguồn gốc của lợi thế dự báo.

**Caveat về dữ liệu:** con số mới (−0,007383) được tính trên subset 3 ngày gần
nhất, KHÔNG phải cùng cửa sổ 10 ngày đã tạo ra con số cũ (+0,049889) — máy
hiện tại không đủ RAM để chạy lại đúng dataset gốc sau khi vá lỗi. Đây là so
sánh "trước/sau" trên hai cửa sổ dữ liệu khác nhau, không phải một thí nghiệm
kiểm soát biến đơn lẻ (chỉ đổi code, giữ nguyên toàn bộ dữ liệu). Xác nhận đầy
đủ trên toàn bộ 10 ngày — hoặc lý tưởng hơn, trên ≥20 ngày UTC untouched theo
pre-registration (xem §8) — vẫn cần được thực hiện trước khi coi đây là kết
luận cuối.

> **Verdict tạm thời (cập nhật sau khi vá lỗi fit/predict mismatch):** QRW có
> lợi thế Brier score có ý nghĩa thống kê so với baseline affine trên subset 3
> ngày gần nhất đã kiểm toán lại. Verdict "QRW kém hơn" trước đó bị vô hiệu vì
> được tính từ một pipeline không thực sự chạy cơ chế mà nó tuyên bố kiểm
> định. Lợi thế này không đến từ giao thoa pha (alpha_phase ≈ 0) mà chủ yếu từ
> hệ số decoherence rất thấp. Cần chạy lại trên toàn bộ dataset gốc — hoặc
> dataset multi-day mới theo pre-registration — trước khi coi đây là kết luận
> cuối.

## 5b. Ablation cô lập alpha_phase (Phase 1, exploratory)

Để trả lời trực tiếp câu hỏi cốt lõi — cơ chế lượng tử có đóng góp không —
một ablation cô lập được chạy trên đúng dataset đã tạo ra con số dương
(`features_BTCUSDT_recent_subset.parquet`, 4 triệu tick, folds=3). Bốn cấu hình
khác nhau **duy nhất** ở pha / dispatch, giữ nguyên phần còn lại của warmup fit:

| Config | pooled Brier | Ghi chú |
|---|---:|---|
| A_full (pha tự do, ≈3,6×10⁻⁵) | 0,100424 | model đã báo cáo |
| B_refit (pha ghim = 0, refit) | 0,100424 | |
| B_posthoc (pha zeroed) | 0,100424 | |
| C_affine (baseline OLS độc lập) | 0,113292 | |

Cơ sở lý thuyết: pha vào coin qua `phi = alpha_phase·direction/window`
([qrw_market_sim.py](../src/models/qrw_market_sim.py)); khi `alpha_phase=0` mọi
coin SU(2) suy biến thành phép quay thực SO(2) **giao hoán**, xoá đúng hiệu ứng
giao thoa. Vì vậy toggle `alpha_phase` cô lập chính xác cơ chế lượng tử.

**Phân rã cơ chế (paired Brier, block-bootstrap 95%):**

- Pha thuần túy (A_full − B_refit): **+9×10⁻⁸** — bằng 0 về mặt thực tiễn (nhỏ
  hơn edge chính 5 bậc độ lớn), dấu bất lợi. Phase sweep xác nhận: Brier tăng
  đơn điệu khi ép pha 0 → 2,0.
- Windowing không pha (B_refit − C_affine): **−0,012867** [−0,0138; −0,0119] —
  toàn bộ lợi thế của QRW đến từ đây, không phải từ pha.

**Độ bền theo số fold (A_full vs affine):**

| folds | QRW Brier | affine Brier | edge | QRW thắng? |
|---:|---:|---:|---:|:--:|
| 2 | 0,100419 | 0,113134 | −0,012716 | ✔ |
| 3 | 0,100424 | 0,113292 | −0,012867 | ✔ |
| 4 | 0,102915 | 0,113350 | −0,010435 | ✔ |
| 5 | 0,142450 | 0,113378 | +0,029072 | ✘ |
| 6 | 0,158408 | 0,113255 | +0,045152 | ✘ |
| 8 | 0,177498 | 0,113242 | +0,064256 | ✘ |

QRW Brier suy thoái đơn điệu khi tăng fold; affine phẳng. Verdict "QRW thắng
affine" đảo dấu ở fold ≥ 5 — không bền vững.

**Cross-asset (ETH, BNB — mỗi asset ~4 triệu tick):** ablation được lặp lại
([_ETHUSDT.md](../reports/research/alpha_phase_ablation_ETHUSDT.md),
[_BNBUSDT.md](../reports/research/alpha_phase_ablation_BNBUSDT.md)).

| Asset | Đóng góp pha (A−B_refit) | QRW vs affine (3-fold) | Ổn định theo fold |
|---|---:|---:|---|
| BTC | ~+9×10⁻⁸ (≈0) | −0,0129 (thắng) | Không — đảo dấu ở fold ≥ 5 |
| ETH | ~+1×10⁻⁵ (≈0) | +0,0098 (**thua**) | Có (thua ổn định mọi fold) |
| BNB | ~+1×10⁻⁶ (≈0) | −0,0111 (**thắng**) | Có (thắng ổn định mọi fold) |

Hai kết luận cross-asset: (a) **đóng góp của pha lượng tử ≈ 0 trên cả ba
asset**, và trên cả ba phase sweep đều cho Brier xấu đi đơn điệu — đây là phát
hiện vững; (b) việc windowed-QRW có thắng affine hay không **phụ thuộc asset**
(BNB thắng, ETH thua, BTC mong manh theo fold), nên không tồn tại một lợi thế
QRW tổng quát, và ngay cả thành phần windowing/decoherence cũng không nhất
quán giữa các asset.

**Nghi phạm nguyên nhân của fold-fragility trên BTC (đầu mối cho Phase 2):** `calibrate_bias()`
([qrw_market_sim.py](../src/models/qrw_market_sim.py)) tối ưu bias dưới công
thức **cổ điển** `_direction_probability`, nhưng bước predict dùng công thức
**quantum windowed** — một fit/predict inconsistency còn sót (họ hàng của
C1/C2). Càng nhiều fold, bias re-estimate dưới sai công thức càng lệch, khiến
QRW suy thoái trong khi affine (nhất quán nội tại) ổn định. Cần verify và sửa.

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
6. Walk-forward audit sau khi vá lỗi fit/predict mismatch (§5) mới chạy được
   trên subset 3 ngày gần nhất, chưa chạy được trên toàn bộ 10 ngày gốc vì
   `rolling_stability()`/`walk_forward_evaluation()` copy dataframe lặp lại
   theo từng block, vượt quá RAM khả dụng trên máy hiện tại (~4,4 GB free cho
   dataset ~32,4 triệu tick). Cần tối ưu bộ nhớ của hai hàm này hoặc chạy trên
   máy có nhiều RAM hơn để có con số thay thế chính xác cho +0,049889 cũ.

## 9. Kết luận

Dự án đã sửa ngữ nghĩa đo QRW và quy tắc thống kê theo hướng có thể kiểm toán.
Một lỗi fit/predict mismatch nghiêm trọng cũng vừa được phát hiện và sửa: cơ
chế "quantum refinement" (`alpha_phase`) mà `calibrate()` có thể chọn trước
đây không hề được dùng ở bước dự đoán walk-forward thực tế, khiến verdict âm
trước đó (QRW kém hơn baseline) được tính từ một pipeline chưa từng thực sự
kiểm định cơ chế nó tuyên bố kiểm định.

Sau khi vá, walk-forward audit ở protocol 3-fold từng cho kết quả dương với
QRW. Nhưng ablation Phase 1 (§5b) đã **thu hẹp mạnh** kết luận đó: (1) lợi thế
so với affine **không bền vững** — đảo dấu thành thua có ý nghĩa thống kê khi
số fold ≥ 5, nên con số dương là artifact của lựa chọn 3-fold; (2) đóng góp
của cơ chế giao thoa lượng tử (`alpha_phase`) đo được **bằng 0** — nơi QRW
thắng, lợi thế đến hoàn toàn từ windowing/decoherence; (3) chưa chạy lại được
trên toàn bộ 10 ngày dữ liệu gốc do giới hạn bộ nhớ (§8-6); (4) dữ liệu hoạt
động vẫn ngắn và OBI chưa phải L2 order-book imbalance thật. Tổng hợp: **hiện
chưa có bằng chứng cho một lợi thế dự báo bền vững của QRW, càng không phải từ
cơ chế lượng tử.**

Kết luận khoa học cuối cùng chỉ nên được đưa ra sau khi: protocol được đóng
băng, provenance khớp commit/data hash, walk-forward được chạy lại thành công
trên toàn bộ dataset gốc (hoặc dataset multi-day mới ≥20 ngày UTC untouched
theo pre-registration), và diễn giải "quantum interference" được kiểm định
tách biệt khỏi hiệu ứng decoherence/gamma.
