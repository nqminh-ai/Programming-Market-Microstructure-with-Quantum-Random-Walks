# Báo cáo cuối: Quantum Random Walk cho vi cấu trúc thị trường

## Trạng thái khoa học

**Kết luận hiện tại (cập nhật sau Phase 1–5): KHÔNG có lợi thế dự báo BỀN VỮNG
của QRW, và cơ chế lượng tử cụ thể (pha) đóng góp BẰNG 0. (1) Pha `alpha_phase`
= 0 trên cả ba asset. (2) Ở endpoint DIRECTIONAL (Brier), windowing thắng
baseline affine yếu nhưng THUA có ý nghĩa thống kê các baseline cổ điển mạnh
(OrderFlow AR(5), Logistic+Pairwise) trên CẢ BA asset — ETH xếp chót 7/7. (3) Ở
endpoint CHÍNH đăng-ký-trước là marginal CRPS, QRW KHÔNG ĐỒNG ĐỀU (hạng 1 trên
ETH, 3 trên BTC, 4 trên BNB) — thua ít dứt khoát hơn chiều directional nhưng
vẫn không có lợi thế nhất quán, và không đến từ pha.** Năm nghiên cứu
ablation/so-sánh (gắn nhãn exploratory, [reports/research/](../reports/research/))
chỉ ra:

- **Pha lượng tử đóng góp 0.** Bỏ pha (`alpha_phase=0`, refit) cho Brier giống
  hệt model pha-tự-do tới ≥5 chữ số trên cả ba asset (chênh lệch ~10⁻⁷–10⁻⁵,
  dưới ngưỡng 10⁻⁴); ép pha lớn hơn làm dự báo **xấu đi đơn điệu**. Về lý
  thuyết `alpha_phase=0` biến coin SU(2) thành phép quay SO(2) giao hoán, xoá
  đúng hiệu ứng giao thoa — nên toggle nó cô lập chính xác cơ chế lượng tử.
- **Lợi thế (nơi có) là windowing/decoherence cổ điển, không phải pha.** Trên
  BTC/BNB model bỏ pha vẫn thắng affine y hệt model đầy đủ; trên ETH cả hai đều
  thua. Không có asset nào mà pha tạo ra khác biệt.
- **Fold-fragility từng thấy trên BTC là một bug, đã sửa (Phase 2).** Con số
  dương báo cáo trước đây (−0,007383, BTC 3 fold) từng đảo dấu thành thua ở
  fold ≥ 5; Phase 2 truy ra nguyên nhân (fit/predict inconsistency trong
  `calibrate_bias`) và sửa, sau đó edge ổn định ~−0,013 ở mọi fold. Xem §5b.
- **Windowing thua baseline mạnh ở chiều directional (Phase 3).** Khi so với bộ
  baseline pre-registered (OrderFlow AR(5), Logistic+Pairwise, Hawkes…) dùng
  đúng cùng feature causal, windowed-QRW thua có ý nghĩa thống kê trên cả ba
  asset (BTC hạng 4/7, ETH 7/7, BNB 4/7). OrderFlow AR(5) thắng QRW ở mọi asset.
  Lợi thế vs affine ở §5b không sống sót. Xem §5c.
- **CRPS phân phối — không đồng đều (Phase 5).** Ở endpoint chính đăng-ký-trước
  (mean marginal CRPS, đo bằng `BenchmarkSuite` v4, 5 window/asset), QRW đứng
  hạng 1/6 trên ETH nhưng 3/6 trên BTC và 4/6 trên BNB. Đây là chiều QRW thua
  **ít dứt khoát nhất**, không phải chiều QRW thắng; hạng dao động 1–4 theo
  asset. Số BNB đã được chạy lại 2026-07-22 sau khi phát hiện input cũ
  không tái lập được — xem hộp "Sửa số BNB" trong §5d. Xem §5d.
- **Tuyên bố volatility không xác lập được (§5d′).** Câu "QRW thua đậm ở window
  biến động cao vì không mô hình hóa volatility" vốn đọc từ **5 window/asset**.
  Đo lại trên **40 window/asset**: tương quan chạy đúng chiều trên cả ba asset
  nhưng không asset nào đạt ý nghĩa thống kê, và hai cách gộp nằm hai bên
  α = 0,05 (Fisher 0,072 / Stouffer 0,043). Dữ liệu **nghiêng về** chiều đó,
  không đủ để gọi là phát hiện. Xem §5d′.

Các caveat cũ vẫn giữ nguyên: dữ liệu hoạt động ngắn, OBI là proxy trade-flow,
và toàn bộ Phase 1–5 là exploratory (chưa đóng băng protocol/pre-registration
confirmatory); windowing CRPS trong-file mỏng hơn chuẩn day-cluster.

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

## 5b. Ablation cô lập alpha_phase + sửa fold-fragility (Phase 1–2, exploratory)

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

**Độ bền theo số fold — trước (Phase 1) và sau khi vá (Phase 2):**

Phase 1 phát hiện edge đảo dấu theo số fold; Phase 2 truy được nguyên nhân là
một fit/predict inconsistency trong `calibrate_bias` và sửa nó (xem dưới). Sau
khi vá, edge trở nên **ổn định ở mọi fold**.

| folds | edge trước vá | QRW thắng? | edge sau vá | QRW thắng? |
|---:|---:|:--:|---:|:--:|
| 2 | −0,012716 | ✔ | −0,012714 | ✔ |
| 3 | −0,012867 | ✔ | −0,012868 | ✔ |
| 4 | −0,010435 | ✔ | −0,012927 | ✔ |
| 5 | **+0,029072** | ✘ | **−0,012953** | ✔ |
| 6 | **+0,045152** | ✘ | **−0,012828** | ✔ |
| 8 | **+0,064256** | ✘ | **−0,012813** | ✔ |

Trước vá, QRW Brier nổ từ 0,1004 (fold 2) lên 0,1775 (fold 8) — đảo verdict
thành thua. Sau vá, QRW Brier phẳng ~0,10042 ở mọi fold và thắng affine ổn
định ~−0,0128. Fold-fragility là **bug, không phải bản chất**.

**Cross-asset (ETH, BNB — mỗi asset ~4 triệu tick), sau khi vá Phase 2:**
ablation được lặp lại với `calibrate_bias` đã sửa
([_ETHUSDT_postfix.md](../reports/research/alpha_phase_ablation_ETHUSDT_postfix.md),
[_BNBUSDT_postfix.md](../reports/research/alpha_phase_ablation_BNBUSDT_postfix.md),
[_BTC_postfix.md](../reports/research/alpha_phase_ablation_BTC_postfix.md)).

| Asset | Đóng góp pha (A−B_refit) | QRW vs affine (mọi fold 2–8) | Ổn định theo fold |
|---|---:|---:|---|
| BTC | +3×10⁻⁷ (≈0) | −0,0128 (**thắng**) | Có (sau vá; trước vá đảo dấu) |
| ETH | −1×10⁻⁵ (≈0) | +0,0096 (**thua**) | Có |
| BNB | +3×10⁻⁷ (≈0) | −0,0110 (**thắng**) | Có |

Hai kết luận cross-asset: (a) **đóng góp của pha lượng tử ≈ 0 (dưới ngưỡng
10⁻⁴ Brier) trên cả ba asset**, và phase sweep cho Brier xấu đi đơn điệu — đây
là phát hiện **vững nhất** của Phase 1–2; (b) sau khi sửa fold-fragility,
windowed-QRW **thắng affine ổn định trên BTC và BNB nhưng thua ổn định trên
ETH** — một hiệu ứng windowing/decoherence **phụ thuộc asset**, KHÔNG phải từ
giao thoa lượng tử và KHÔNG tồn tại như một lợi thế QRW tổng quát.

**Nguyên nhân đã xác nhận và sửa (Phase 2):** `calibrate_bias()`
([qrw_market_sim.py](../src/models/qrw_market_sim.py)) trước đây tối ưu bias
dưới công thức **cổ điển** `_direction_probability`, nhưng bước predict dùng
công thức **quantum windowed** — một fit/predict inconsistency còn sót (họ hàng
của C1/C2 mà bản vá C1 chưa xử lý). Càng nhiều fold, bias re-estimate dưới sai
công thức càng lệch, khiến QRW suy thoái trong khi affine (nhất quán nội tại)
ổn định. Phase 2 sửa `calibrate_bias` để dùng đúng công thức mà `predict` sẽ
dùng (dispatch trên `quantum_improved`), có test regression bảo vệ. Bảng
"trước/sau vá" ở trên xác nhận: fold-fragility biến mất hoàn toàn. **Lưu ý diễn
giải:** đây là sửa một lỗi kỹ thuật làm verdict ổn định, KHÔNG phải bằng chứng
mới cho cơ chế lượng tử — đóng góp của pha vẫn bằng 0 sau khi vá.

## 5c. So sánh với baseline MẠNH (Phase 3, exploratory)

Affine (chỉ OBI + tick direction) là baseline yếu. Câu hỏi quyết định: lợi thế
windowing của QRW ở §5b có sống sót trước bộ baseline mạnh trong
pre-registration không? Tôi tái dùng module có sẵn
[directional_baselines](../src/evaluation/directional_baselines.py): Logistic
L2 (5 feature), Logistic L2 + Pairwise, Nonlinear calibrated, **OrderFlow
AR(5)**, Marked Hawkes logit, và QRW directional-link (xấp xỉ logistic). Fit
trên chronological train/validation (50/25), chấm trên test disjoint (25%). Mọi
model dùng **cùng event set** (`directional_events` lọc y hệt `market_events`)
và **cùng feature causal**, nên so sánh Brier là paired và block-bootstrap.
Script: [strong_baseline_comparison.py](../scripts/research/strong_baseline_comparison.py).

Test-set Brier (thấp = tốt), mỗi asset ~4 triệu tick:

| Asset | Windowed-QRW | Hạng | Baseline mạnh nhất | Brier tốt nhất | edge QRW−best (KI 95%) |
|---|---:|:--:|---|---:|---|
| BTC | 0,1019 | 4/7 | Logistic L2 + Pairwise | 0,0496 | +0,052 [+0,051; +0,054] |
| ETH | 0,1001 | **7/7** | OrderFlow AR(5) | 0,0657 | +0,034 [+0,033; +0,036] |
| BNB | 0,1767 | 4/7 | OrderFlow AR(5) | 0,1466 | +0,030 [+0,029; +0,031] |

Trên **cả ba asset**, windowed-QRW **thua có ý nghĩa thống kê** baseline mạnh
nhất (KI 95% hoàn toàn dương). OrderFlow AR(5) — một logistic đơn giản trên
hướng tick trễ — thắng QRW ở cả ba; trên ETH windowed-QRW **xếp chót 7/7**.
Xem report chi tiết:
[BTC](../reports/research/strong_baseline_BTCUSDT.md),
[ETH](../reports/research/strong_baseline_ETHUSDT.md),
[BNB](../reports/research/strong_baseline_BNBUSDT.md).

**Kết luận Phase 3:** lợi thế "QRW thắng affine" (§5b) chỉ tồn tại vì affine cố
tình yếu (không có momentum/lag). Khi đấu với model cổ điển cạnh tranh dùng
đúng cùng feature causal, windowed-QRW **rút signal kém hơn và thua đậm**.
Tổng hợp Phase 1–3: KHÔNG có bằng chứng cho bất kỳ lợi thế dự báo nào của QRW —
dù là cơ chế "giao thoa lượng tử" (pha = 0) hay thành phần windowing cổ điển
(thua OrderFlow AR trên mọi asset).

## 5d. Endpoint CHÍNH đăng ký trước — CRPS phân phối (Phase 5, exploratory)

Phase 1–3 đo endpoint **directional** (Brier). Nhưng endpoint đăng-ký-trước
CHÍNH lại là **mean fixed-origin marginal CRPS** (một bài toán phân phối, không
phải phân loại hướng). Tôi dùng chính `BenchmarkSuite` (protocol v4,
[benchmark_suite.py](../src/evaluation/benchmark_suite.py)) — tiến hóa density
matrix QRW thành fixed-origin position marginals rồi chấm CRPS từng horizon so
với holdout thật — đấu với CRW (3 biến thể), GARCH(1,1) và GBM. Để tránh
single-origin fragility (bài học Phase 2), chạy **5 window không chồng lấp** mỗi
asset. Script: [marginal_crps_comparison.py](../scripts/research/marginal_crps_comparison.py).

Mean marginal CRPS trung bình qua 5 window (thấp = tốt):

| Asset | Hạng QRW | Model tốt nhất | QRW CRPS | Best CRPS | QRW thắng window |
|---|:--:|---|---:|---:|:--:|
| BTC | 3/6 | GBM | 2,347 | 1,858 | 0/5 |
| ETH | **1/6** | **QRW** | 0,0962 | 0,0962 | 3/5 |
| BNB | 4/6 | CRW Correlated | 0,0614 | 0,0555 | 1/5 |

> **Sửa số BNB (2026-07-22).** Bản báo cáo trước ghi BNB hạng 2/6 (QRW 0,0958,
> thắng 3/5 window). Rà soát repo phát hiện artifact đó trỏ vào một file
> `bnb_combined.parquet` nằm trong thư mục tạm của phiên làm việc, **đã không
> còn tồn tại** — tức input không tái lập được, vi phạm chính tiêu chí trong
> [ARTIFACT_STATUS.md](../reports/ARTIFACT_STATUS.md). Đã dựng lại
> `features_BNBUSDT_multiday.parquet` (31.503.940 dòng, toàn bộ 31 ngày) **bên
> trong repo** và chạy lại; số trong bảng là kết quả mới, có canonical path +
> SHA-256. Hai lần chạy dùng **input khác nhau** nên không so sánh trực tiếp
> được; số cũ bị thay thế chứ không phải bị bác bỏ. Số ETH/BTC không đổi vì
> input của chúng vốn đã nằm trong repo.

**Kết luận Phase 5 — tinh tế, khác chiều directional:** trên endpoint phân phối,
QRW **không bị đè bẹp như ở chiều directional**, nhưng cũng **không cạnh tranh
đồng đều**: tốt nhất trên ETH (1/6), giữa bảng trên BTC (3/6) và **dưới trung
bình trên BNB (4/6)** — thua cả CRW Correlated lẫn GBM. **Không có lợi thế nhất
quán**; hạng dao động 1–4 theo asset. Ba caveat quan trọng: (a) biên nhỏ và phụ
thuộc window — bản báo cáo trước đọc từ 5 window rằng QRW thắng ở window **biến
động thấp** và thua đậm ở window **biến động cao**; §5d′ dưới đây đo lại điều đó
trên 40 window mỗi asset và **không chứng minh được** nó; (b) windowing trong-file
mỏng hơn pre-registration (ETH chỉ gói gọn 1 ngày UTC); (c) exploratory. Diễn
giải trung thực: CRPS là chiều QRW **thua ít dứt khoát nhất**, không phải chiều
QRW thắng — và phần không thua đó vẫn không đến từ pha lượng tử (pha vẫn = 0).

### 5d′. Đo lại tuyên bố volatility: 40 window/asset thay vì 5 (exploratory)

Câu "QRW thắng window yên tĩnh, thua đậm window biến động — vì không mô hình hóa
volatility" là **diễn giải đọc từ 5 điểm**, và giới hạn #5 dựa hẳn vào nó. Năm
điểm không kết luận được theo chiều nào, nên tôi đo nó thay vì tiếp tục khẳng
định: chạy lại trên store 69 ngày với **40 window không chồng lấp** mỗi asset
(20 triệu dòng/asset), rồi tương quan Spearman giữa **realised volatility** của
window và **khoảng cách CRPS tương đối** của QRW so với đối thủ tốt nhất.

Hai điều chỉnh trong chính phép đo trước khi tin nó:

- **Khoảng cách phải là tương đối.** Bản đầu tôi viết dùng hiệu tuyệt đối, nhưng
  thang CRPS chênh nhau ba bậc giữa các window (BTC: 0,01 → 8,0), nên chênh 0,02
  là thảm bại ở window này và là làm tròn ở window kia. Trên BTC, `|gap|` tuyệt
  đối bám thang CRPS của window ở Spearman **+0,90** — tức phép đo đang hỏi một
  phần "window này to cỡ nào", không phải câu cần hỏi. Đã đổi sang
  `(CRPS_QRW − CRPS_đối_thủ_tốt_nhất) / CRPS_đối_thủ_tốt_nhất`.
- **Kiểm định một phía, đúng chiều báo cáo khẳng định.** Hai phía tiêu tốn nửa
  ngân sách cho khả năng QRW làm *tốt hơn* khi volatility tăng — điều báo cáo
  không hề khẳng định.

| Asset | Window | Spearman | p (hai phía) | Ủng hộ? | Hạng QRW | QRW thắng |
|---|---:|---:|---:|:--:|:--:|:--:|
| BNBUSDT | 40 | +0,222 | 0,169 | không | **1/6** | 22/40 |
| ETHUSDT | 40 | +0,047 | 0,773 | không | **1/6** | 31/40 |
| BTCUSDT | 40 | +0,213 | 0,187 | không | 3/6 | 3/40 |

Gộp ba asset một phía: **Fisher p = 0,0720**, **Stouffer p = 0,0425**.

**Kết luận: không xác lập được ở α = 0,05.** Tương quan chạy **đúng chiều báo
cáo khẳng định trên cả ba asset** — đó không phải là không có gì — nhưng không
asset nào tự đạt ý nghĩa thống kê, và hai cách gộp **nằm hai bên α**: Fisher nói
không, Stouffer nói có. Khi kết luận phụ thuộc vào việc trích dẫn kiểm định nào
thì nó chưa được xác lập. Câu đúng để viết là "dữ liệu **nghiêng về** chiều đó",
không phải "chúng tôi tìm thấy điều đó". Giới hạn #5 đã được viết lại theo đúng
mức đó.

> **Hạng BNB đổi từ 4/6 → 1/6.** Trên dữ liệu 69 ngày với 40 window, BNB QRW
> đứng **nhất**, thắng 22/40 window — trong khi bảng §5d (31 ngày, 5 window) ghi
> 4/6. Hạng BTC (3/6) và ETH (1/6) **tái lập nguyên vẹn**. Ba khác biệt cùng có
> thể giải thích: giai đoạn dữ liệu khác, số window khác (40 vs 5), và cỡ window
> khác (500 nghìn dòng vs 6,3 triệu — tức lượng dữ liệu fit mỗi model khác 12,6
> lần). Tôi **không thay số §5d bằng số này**: §5d là bảng theo protocol 5 window
> đã đăng ký, còn đây là robustness check. Điều đáng rút ra không phải "QRW tốt
> hơn ta tưởng trên BNB" mà là **thứ hạng CRPS của BNB không ổn định** giữa hai
> thiết lập — càng củng cố kết luận "không có lợi thế nhất quán", chỉ là theo
> một cách khác với cách §5d nói.

Script: [volatility_claim_summary.py](../scripts/research/volatility_claim_summary.py) ·
Artifact: [`volatility_claim.md`](../reports/research/volatility_claim.md)

## 5e. Khả thi giao dịch: kỹ năng và lợi nhuận ở hai đầu đối lập (exploratory)

Phase 1–5 hỏi "mô hình nào dự báo tốt hơn". Phần này hỏi câu khác hẳn: **có dự
báo tốt hơn thì có giao dịch được không?** Đây là câu hỏi biến dự án từ nghiên
cứu định lượng thành quant trading, và câu trả lời là **chưa**.

Mạch của phần này: trước hết sửa ba lỗi trong **cách đo hiệu quả chiến lược**;
rồi sửa hai lỗi trong **cách đo chi phí giao dịch** — cả hai đều làm chi phí
tăng lên; rồi mới áp bộ chi phí đã đúng vào từng horizon để hỏi horizon nào trả
nổi phí của chính nó; cuối cùng kiểm tra kết luận có bền khi nhân bảy lượng dữ
liệu hay không. **Năm lỗi, cả năm đều được tìm ra bởi chính pipeline kiểm toán
của dự án, và cả năm đều làm kết quả xấu đi.**

### Ba lỗi đo lường phải sửa trước

Bộ chỉ số backtest có ba lỗi, và **không lỗi nào chỉ ảnh hưởng hiển thị**:
`n_trades` đếm mỗi *bar đang giữ vị thế* thành một lệnh (thổi phồng 15–34 lần);
`sharpe` thực chất là `mean/std·√n_bars`, tức một **t-statistic** tăng vô hạn
theo cỡ mẫu; và bản sao công thức trong `optimizer.py` **không trừ phí giao
dịch**. Vì optimizer tối ưu theo `sharpe` và lọc theo `min_trades`, cả ba đang
lái việc **chọn tham số**. Sau khi sửa, grid search chọn khác (θ_buy 0,62 →
0,68) và dấu của chiến lược đảo: profit factor 265 → **0,095**, lãi ròng +0,04%
→ **−4,2%**, Sharpe quy đổi năm **−48,6**. Giữ nguyên backtest và chỉ đổi công
thức cũng tái hiện đúng sự đảo chiều đó, nên đây là lỗi **đo lường**, không phải
dữ liệu.

### Sửa: half-spread cũ không phải spread

Mô hình chi phí **trừ** half-spread khỏi phí maker, tức ghi có cho lệnh chờ
khoản spread ăn được ở cả hai chiều. Half-spread đó tính bằng `|price − mid|`,
nhưng `mid_price` trong feature store là **VWAP trượt 100 lệnh**, không phải mid
của sổ lệnh. Đại lượng được đo thực chất là độ lệch giá khớp so với trung bình
trượt của chính nó — tức độ phân tán giá trong cửa sổ trung bình, bị chi phối
bởi độ trôi, và không liên quan gì tới chênh lệch mua–bán.

Trên một ngày BTCUSDT nó bằng **1,753 USD ≈ 175 tick**, cho một cặp có spread
thật 1–10 tick. Ước lượng Roll (1984) — nghịch đảo bid-ask bounce ngay trong
chuỗi giá khớp, chỉ cần giá giao dịch — cho **0,0085 bps** thay vì 0,28 bps.
Trên cả ba store 69 ngày, thống kê cũ phóng đại **33× (BTC), 22× (ETH), 30×
(BNB)**.

Báo cáo trước có ghi chú `mid_price` "là giá tham chiếu suy ra, không phải L2
mid thật, nên số hạng spread là xấp xỉ". Ghi chú đó nói nhẹ quá: nó **không
phải xấp xỉ của spread, mà là một đại lượng khác**.

Hệ quả **của riêng phép sửa này** (chưa tính adverse selection ở mục kế tiếp):
ngưỡng hoà vốn tăng ở mọi ô và **ô "vượt ngưỡng" duy nhất biến mất**. BNB ở
h=50.000 từng đạt 53,6% so với ngưỡng 52,3% (lãi +1,14 bps/lệnh); chỉ riêng
spread đo đúng đã đẩy ngưỡng lên **54,3%** và lãi ròng xuống **−0,71 bps**.
Khoản lãi đó hoàn toàn là tạo tác của một ước lượng spread hào phóng gấp 30 lần.
ETH ở h=1.000 cũng vượt mốc 100% (89,2% → **100,2%**), nhập vào cùng nhóm với
BTC: dự đoán đúng hoàn hảo vẫn lỗ. Các con số **cuối cùng** trong bảng kết quả
bên dưới còn xấu hơn, vì mục kế tiếp cộng thêm adverse selection.

Thống kê cũ vẫn được tính và in kèm hệ số phóng đại trong mọi báo cáo, để việc
sửa **kiểm toán được** thay vì lặng lẽ thay số.

### Adverse selection: lệnh chờ không ăn spread, mà trả tiền

Mô hình chi phí ghi có cho lệnh chờ khoản half-spread ở cả hai chiều. Giả định
đó chỉ đúng nếu luồng lệnh **không mang thông tin**. Đo trực tiếp bằng phần mà
bên thụ động còn giữ được sau `h` tick — `d·(P_t − P_{t+h})/P_t`, với `d = +1`
khi taker mua:

| | Roll effective | realised (h=1.000) | adverse selection |
|---|---:|---:|---:|
| BTC | 0,0073 bps | **−1,192 bps** | 1,199 |
| ETH | 0,0210 | **−1,245** | 1,266 |
| BNB | 0,0316 | **−1,042** | 1,074 |

Realised half-spread **âm ở mọi horizon và mọi asset**. Người đặt lệnh chờ không
những không thu được spread, mà **mất khoảng 1,2 bps** — lớn hơn hai bậc độ lớn
so với cái spread mà mô hình đang ghi có cho họ. Adverse selection còn **tăng
theo horizon và không hoàn lại** (BNB: 1,07 bps ở h=1.000 → 1,43 ở h=200.000),
tức đây là tác động giá **vĩnh viễn**, không phải dao động tạm thời.

Kết quả này khớp với phần "Có kỹ năng thật" bên dưới từ hướng ngược lại: chính
thứ order flow cho BTC 64,4% độ chính xác dự báo hướng là thứ khiến bên thụ động
bị "nhặt". Hai phép đo độc lập, cùng một hiện tượng.

Hệ quả nặng nhất: **lợi thế của maker gần như biến mất**. Trước đây maker khả thi
sớm hơn taker nhiều; nay với BTC cả hai cùng khả thi từ h=200.000 (~87 phút), và
với BNB cả hai cùng từ ~1,5 giờ. Đặt lệnh chờ không còn là cách né phí — nó chỉ
đổi phí sàn lấy adverse selection.

**Giới hạn của chính phép đo này:** tham chiếu là giá khớp ở `t+h`, mà giá đó
mang bid-ask bounce của chính nó. Vì dấu lệnh có autocorr 0,965, bounce không tự
triệt tiêu ở horizon ngắn — con số ở h=1 (−0,006 bps với BTC) chủ yếu là tạo tác
đó chứ không phải price impact. Các horizon dài, nơi dấu lệnh đã hết tương quan,
mới là phần có ý nghĩa. Mô hình hàng đợi lệnh đầy đủ vẫn cần **L2 thật**
(§8 hạn chế #1).

### Horizon 1 tick là bất khả thi về mặt toán học

Với horizon `h`, một cược hướng có tỉ lệ đúng `p` thu về `(2p−1)·E|r_h|` trước
phí, nên hoà vốn đòi `p > 0,5 + chi_phí/(2·E|r_h|)`. Ở horizon dự án đang dùng —
**1 tick** — biến động trung bình chỉ bằng 0,0006 (BTC), 0,0012 (ETH) và 0,0039
(BNB) lần một vòng taker, tức ngưỡng hoà vốn **vượt 100%**: một mô hình dự đoán
đúng *hoàn hảo* vẫn lỗ. Đây là giới hạn của **horizon**, không phải của mô hình,
và không kỹ thuật mô hình hoá nào cứu được.

Áp hai phép sửa chi phí ở trên vào: với half-spread Roll (0,0073–0,0316 bps,
nhỏ hơn phí sàn ba bậc độ lớn) và realised half-spread âm ~1,2 bps, đặt lệnh chờ
(maker 2bps/chiều) chỉ khả thi từ **~87 phút (BTC), ~47 phút (ETH), ~1,5 giờ
(BNB)** — so với ~20 phút khi còn giả định maker ăn được spread.
Script: [horizon_feasibility.py](../scripts/research/horizon_feasibility.py).

### Có kỹ năng thật, nhưng không ở nơi có tiền

Đổi nhãn sang **dấu lợi suất qua `h` tick**, chạy lại bộ baseline causal đã đăng
ký trên các cửa sổ **không chồng lấp** (anchor cách nhau đúng `h` tick, nên
không nhãn nào chia sẻ tương lai với nhãn khác):

Bảng dưới chạy trên **69 ngày mỗi asset** (BTC 227,6M dòng, ETH 212,0M, BNB
54,1M — xem "Mở rộng dữ liệu" bên dưới). Mỗi ô là độ chính xác của mô hình tốt
nhất, kèm khoảng tin cậy Wilson 95% và cỡ mẫu kiểm định:

| Horizon | BTC | ETH | BNB |
|---|---:|---:|---:|
| 1.000 | **64,4%** [64,0–64,8] n=68.001 | 58,5% [58,1–58,9] n=63.265 | 54,4% [53,6–55,2] n=16.076 |
| 5.000 | 56,4% [55,5–57,2] n=13.636 | 53,8% [52,9–54,7] n=12.691 | 52,9% [51,2–54,6] n=3.230 |
| 10.000 | 55,5% [54,3–56,7] n=6.823 | 52,6% [51,3–53,8] n=6.347 | 51,9% [49,4–54,3] n=1.618 |
| 50.000 | 50,3% *(= hằng số)* n=1.365 | 51,0% *(thua hằng số 51,3%)* n=1.271 | 53,6% [48,1–58,9] n=323 |
| **Ngưỡng hoà vốn maker 2bps ở h=50.000** | 65,5% | 61,5% | 57,3% |
| **Ngưỡng ở h=1.000** | 163,9% | 132,3% | 97,3% |

**Order flow có sức dự báo thật, và nó lặp lại được.** Ở lần chạy trước trên
32,4M dòng, BTC h=1.000 đạt 65,7% với 9.695 cửa sổ. Trên **7 lần** lượng dữ liệu
đó (68.001 cửa sổ), con số là 64,4% với khoảng tin cậy chỉ còn ±0,4 điểm. Một
hiệu ứng giả do cỡ mẫu nhỏ sẽ không sống sót qua phép nhân bảy này.

Nhưng ở horizon đó giá chưa dịch đủ để trả phí — ngưỡng hoà vốn của BTC tại
h=1.000 là **163,9%** và của ETH là **132,3%**, tức vượt 100% ngay cả ở phí
maker: dự đoán đúng *hoàn hảo* vẫn lỗ. Kéo horizon ra tới khi biên độ đủ lớn thì
**kỹ năng biến mất**: BTC ở h=50.000 rơi đúng bằng hằng số, ETH còn *thua* hằng
số. Câu hỏi mà việc mở rộng dữ liệu nhằm trả lời — liệu h=50.000 của ETH, trước
đây bị bỏ trống vì "thiếu mẫu", có edge hay không — nay đã có đáp án dứt khoát:
**không**.

**Không horizon nào trên bất kỳ asset nào vượt ngưỡng hoà vốn**, kể cả ở mức phí
maker, và sau khi sửa ước lượng spread thì **không còn ô nào sát ngưỡng**. Ô duy
nhất từng vượt (BNB h=50.000) đã được xử lý hai lần độc lập: kiểm định nhị thức
cho p = 0,345 với khoảng tin cậy [48,1%–58,9%] chứa cả mức tung đồng xu; và
spread đo đúng đẩy ngưỡng của nó từ 52,3% lên 54,3%, và tính thêm adverse
selection thì lên **57,3%** — bỏ xa 53,6% đạt được, lãi ròng **−3,43 bps**.
Script vẫn gắn dấu `⚠ p=…` thay vì dấu tick cho bất kỳ ô nào vượt ngưỡng mà
không qua kiểm định, để lần chạy sau không tuyên bố nhầm.

Con số 64,4% được **kiểm tra chứ không báo cáo thẳng**: ablation từng feature
truy ra `tick_direction` (tương quan +0,356 với lợi suất tương lai), một biến có
autocorr **0,965 ở lag 1** — đúng hiện tượng long-memory of order flow. Đối
chiếu với các cửa sổ tương lai **rời nhau** cho thấy tương quan sụp từ +0,329
xuống +0,011 ngay ở cửa sổ kế tiếp, tức tác động flow ngắn hạn **thật**; rò rỉ
thì sẽ duy trì qua mọi cửa sổ. Script:
[horizon_label_baselines.py](../scripts/research/horizon_label_baselines.py).

### Mở rộng dữ liệu: 7× và kết luận không đổi

Toàn bộ §5e ban đầu chạy trên 31 ngày BTC (32,4M dòng), với ETH và BNB lệch
ngày nhau. Để kiểm tra xem kết luận có phải là hệ quả của cỡ mẫu nhỏ hay không,
dữ liệu được kéo lên **69 ngày trùng khớp cho cả ba asset** (2026-05-13 →
2026-07-20, tổng 493,7M dòng). Cỡ mẫu kiểm định ở horizon dài tăng từ **194 lên
1.271–1.365 cửa sổ**, đưa sai số chuẩn của độ chính xác từ 3,6% xuống 1,4%.

Half-spread cũng được đo lại trên **toàn bộ** store thay vì 4 triệu dòng đầu:
BTC 0,239 bps, ETH 0,454 bps, BNB 0,958 bps. Ngưỡng khả thi cho lệnh maker
2bps/chiều là h=50.000 với BTC (~21,8 phút) và ETH (~23,4 phút), h=5.000 với
BNB (~9,2 phút).

**Kết luận không đổi ở bất kỳ điểm nào.** Đây là điều đáng nói nhất: dữ liệu lớn
gấp bảy lần không lật được kết quả nào, chỉ làm các khoảng tin cậy hẹp lại quanh
đúng những con số cũ.

### Hạn chế của chính phần này

Khử chồng lấp làm cỡ mẫu tụt còn **323–68.001 cửa sổ**, và ở horizon dài nhất
của BNB (323 cửa sổ) khoảng tin cậy vẫn rộng hơn ±5 điểm — không kết luận nào ở
riêng ô đó là dứt khoát.

Adverse selection **nay đã được đo và tính vào chi phí** (mục ở trên), nhưng chỉ
ở mức dữ liệu giao dịch cho phép: nó ước lượng bên thụ động *mất bao nhiêu khi
đã khớp*, chứ không mô hình hoá **khi nào thì khớp**. Xác suất khớp, vị trí
trong hàng đợi, và việc lệnh bị huỷ trước khi tới lượt đều cần **L2 thật**
(§8 hạn chế #1). Nói cách khác: con số hiện tại là **cận dưới** của chi phí thực
mà một chiến lược maker phải chịu, vì nó bỏ qua chi phí cơ hội của những lần
không được khớp.

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

**Tình trạng cũ (giữ nguyên phần rút lại):** `qrw_heavy_tail.py` là bộ sinh bước
nhảy Bernoulli/Pareto **cổ điển** (chính file tự khai *"this is not a unitary
heavy-tail QRW"*). Nó không triển khai heavy-tailed unitary shift, nên mọi tuyên
bố cũ về tail index `1,1–2,5`, bootstrap CI cải thiện hay DM bền vững **vẫn bị
rút lại**.

**Đã bổ sung (Phase 6):** cơ chế còn thiếu nay đã có —
[`src/models/heavy_tail_unitary.py`](../src/models/heavy_tail_unitary.py) cài
một **Lévy shift unitary chính xác**. Trên vòng N vị trí, shift là chéo hoá
trong không gian động lượng; tổng quát pha thành

    φ_α(k) = sign(k)·|k|^α ,   k ∈ [−π, π)

giữ mọi trị riêng trên đường tròn đơn vị ⟹ **unitary theo cấu trúc với mọi α**
(không phải xấp xỉ). Hai tính chất then chốt, đều có test bảo vệ
([test_heavy_tail_unitary.py](../tests/test_heavy_tail_unitary.py), 13 test):

- **α = 1 tái tạo CHÍNH XÁC** shift lân-cận-gần-nhất ±1 — đây là tổng quát hoá
  chặt của walk hiện có, không phải mô hình khác.
- **α < 1** cho pha có điểm kỳ dị `|k|^α` tại k=0 ⟹ hệ số Fourier suy giảm lũy
  thừa ⟹ biên độ nhảy `~|x|^{-(1+α)}` = **Lévy flight sinh bởi unitary**, chứ
  không phải lấy mẫu nhảy cổ điển.

**Đánh giá thực nghiệm** ([heavy_tail_unitary_evaluation.py](../scripts/research/heavy_tail_unitary_evaluation.py),
horizon 50 tick, lattice 16.001). So sánh **hình dạng đuôi** bằng tỉ lệ quantile
của |x − median| (scale-free). *Cố ý không dùng phương sai/kurtosis*: với Lévy
α<2 mô men bậc hai **không tồn tại**, nên σ đo trên lattice hữu hạn chỉ phản ánh
kích thước lattice chứ không phải phân phối.

| Asset | Empirical q999/q75 | α khớp nhất | Lévy đạt | Walk chuẩn (α=1) đạt |
|---|---:|:--:|---:|---:|
| BTC | 5,07 | 0,7 | 7,74 | **1,15** |
| ETH | 3,09 | 0,9 | **3,10** | **1,15** |
| BNB | 4,00 | 0,9 | 3,10 | **1,15** |

**Kết luận:** walk lượng tử thường **về bản chất không thể** tạo đuôi nặng —
marginal của nó là bimodal/ballistic với giá đỡ compact, q999/q75 ≈ 1,15 so với
3–5 của thị trường thật (sai lệch **định tính**). Lévy unitary shift khắc phục
đúng khiếm khuyết này, với α ≈ 0,7–0,9 khớp cả ba asset (ETH gần như khớp hệt).

**Giới hạn diễn giải — quan trọng:** đây là kiểm định **hình dạng phân phối**,
KHÔNG phải kỹ năng dự báo. Một mô hình cổ điển đuôi nặng (Lévy-stable,
Student-t) cũng khớp được đuôi như vậy; α còn là tham số **fit** khác nhau theo
asset. Vì vậy kết quả này **đóng khoảng trống cơ chế** của §7 (giờ đã có một
heavy-tail unitary hợp lệ, có test, tái lập được) nhưng **không** là bằng chứng
cho ưu thế lượng tử, và không mâu thuẫn với kết luận §5c–5d.

## 8. Hạn chế chưa giải quyết

1. Chưa có ít nhất 20 ngày UTC mới, untouched, cho mỗi tài sản. **Hạ tầng đã
   sẵn sàng** ([collect_confirmatory.py](../scripts/operations/collect_confirmatory.py),
   11 test) — phân đoạn theo ngày UTC, resumable, manifest bất biến có SHA-256,
   `--status` báo tiến độ. Phần còn thiếu là **thời gian thực**: ≥20 ngày UTC
   tương lai không thể rút ngắn.
2. Chưa có L2 LOB đồng bộ với trade feed **trong tập confirmatory**. Runner ở
   mục 1 thu trade + L2 depth trên cùng một stream kết hợp và hard-code
   `obi_source="lob"`, nên khi chạy nó sẽ đóng mục này; dữ liệu exploratory
   hiện tại vẫn là trade-flow proxy.
3. Artifact Phase 3–6 chính thức chưa được tái tạo hoàn chỉnh dưới protocol v4.
4. Dữ liệu cross-asset chưa được đánh giá theo toàn bộ ngày có sẵn.
5. Chưa thể đưa ra kết luận confirmatory hoặc production-readiness.
6. ~~Walk-forward audit sau vá chưa chạy được trên toàn bộ 10 ngày gốc vì
   giới hạn bộ nhớ.~~ **ĐÃ GIẢI QUYẾT (Phase 4):** dùng loader tiết kiệm bộ nhớ
   (chỉ 7 cột cần thiết + downcast float32) toàn bộ **32.439.057 tick chỉ chiếm
   1.070 MB**, chạy trọn walk-forward directional trên full dataset gốc trong
   ~3,8 GB RAM. Con số thay thế cho +0,049889 cũ: post-fix edge QRW−affine =
   **−0,013091** (folds=3) và **−0,012771** (folds=5) — QRW thắng affine, ổn
   định, khớp kết quả subset 4M. Nói cách khác, trên **cùng** full dataset,
   việc vá bug `calibrate_bias` (Phase 2) lật verdict từ +0,05 (thua) sang
   −0,013 (thắng); alpha_phase = −1,2×10⁻⁵ (pha vẫn ≈ 0). Xem
   [full_dataset_confirmation.md](../reports/research/full_dataset_confirmation.md).
   Lưu ý: điều này chỉ củng cố so sánh vs affine ở full scale; §5c vẫn cho thấy
   QRW thua baseline mạnh.

   **Lặp lại trên 100M dòng, giai đoạn khác (2026-07-23).** Chạy lại đúng quy
   trình đó trên store 69 ngày mới, lấy 100.000.000 dòng đầu — gấp **3,1 lần**
   và phủ khoảng thời gian khác hẳn:

   | Dữ liệu | folds=2 | folds=3 | folds=5 |
   |---|---:|---:|---:|
   | 32,4M (gốc) | — | −0,013091 | −0,012771 |
   | **100M (mới)** | **−0,015280** | **−0,014766** | **−0,013758** |

   Cùng dấu, cùng bậc độ lớn, khoảng tin cậy 95% đều nằm trọn bên âm, và giữ
   nguyên quy luật edge co lại khi tăng số fold. Kết quả trung tâm **không phải
   là hiện tượng của riêng một mẫu**. Xem
   [confirmation_btcusdt_69d_100M.md](../reports/research/confirmation_btcusdt_69d_100M.md).

   Không chạy được trên trọn 227,6M dòng: khâu calibrate cần thêm ~3 GB trên
   frame 7,5 GB và máy 15,8 GB không đủ. Đây là giới hạn phần cứng, đã ghi rõ
   thay vì lặng lẽ báo cáo phần chạy được như thể là toàn bộ.

## 9. Kết luận

Dự án đã sửa ngữ nghĩa đo QRW và quy tắc thống kê theo hướng có thể kiểm toán.
Một lỗi fit/predict mismatch nghiêm trọng cũng vừa được phát hiện và sửa: cơ
chế "quantum refinement" (`alpha_phase`) mà `calibrate()` có thể chọn trước
đây không hề được dùng ở bước dự đoán walk-forward thực tế, khiến verdict âm
trước đó (QRW kém hơn baseline) được tính từ một pipeline chưa từng thực sự
kiểm định cơ chế nó tuyên bố kiểm định.

Năm bước ablation/so-sánh (§5b–5d, Phase 1–5) đã làm rõ trọn vẹn: **(1)** đóng
góp của cơ chế giao thoa lượng tử (`alpha_phase`) đo được **bằng 0** trên cả ba
asset —
bỏ pha cho kết quả giống hệt, ép pha lớn hơn làm xấu đi. **(2)** Fold-fragility
từng thấy trên BTC (edge đảo dấu ở fold ≥ 5) là một **bug** fit/predict
inconsistency trong `calibrate_bias`, đã sửa ở Phase 2; sau khi sửa,
windowed-QRW thắng baseline **affine** ổn định trên BTC/BNB (thua ETH). **(3)**
Nhưng lợi thế đó chỉ vì affine yếu: khi đấu với baseline cổ điển **mạnh**
(OrderFlow AR(5), Logistic+Pairwise) dùng đúng cùng feature causal, windowed-QRW
**thua có ý nghĩa thống kê trên cả ba asset** (§5c; ETH xếp chót 7/7). **(4)**
Ở endpoint CHÍNH đăng-ký-trước là marginal CRPS (§5d), bức tranh **tinh tế hơn**:
QRW hạng 1 trên ETH nhưng 3 trên BTC và 4 trên BNB — thua ít dứt khoát hơn
directional, song vẫn không có lợi thế nhất quán. **(5)** Phase 4 đóng limitation
#6: chạy được walk-forward directional trên **toàn bộ 32,4M tick gốc** (loader
downcast, 1.070 MB), edge post-fix = −0,013 (thắng affine, thay số cũ
+0,049889). Tổng hợp: **chưa có bằng chứng cho một lợi thế dự báo bền vững của QRW, và cơ
chế lượng tử cụ thể (pha) đóng góp bằng 0**; ở directional một logistic tự hồi quy
đơn giản (OrderFlow AR) đánh bại QRW ở mọi asset, còn ở phân phối (CRPS) QRW chỉ
dẫn đầu trên một trong ba asset. **(6)** §5e đi thêm một bước và hỏi liệu bất kỳ
dự báo nào ở đây có **giao dịch được** không: không. Horizon 1 tick mà dự án dùng
có ngưỡng hoà vốn **vượt 100%** — dự đoán đúng hoàn hảo vẫn lỗ. Order flow có kỹ
năng thật ở horizon ngắn (BTC 64,4% trên 68.001 cửa sổ không chồng lấp) nhưng giá
chưa dịch đủ để trả phí, còn ở horizon dài đủ trả phí thì kỹ năng biến mất. Kết
luận đó **đứng vững sau khi dữ liệu được nhân bảy** lên 69 ngày cho cả ba asset.
Các hạn chế còn lại: dữ liệu hoạt động vẫn ngắn và OBI chưa phải L2 order-book
imbalance thật; toàn bộ là exploratory.

Kết luận khoa học cuối cùng chỉ nên được đưa ra sau khi: protocol được đóng
băng, provenance khớp commit/data hash, walk-forward được chạy lại thành công
trên toàn bộ dataset gốc (hoặc dataset multi-day mới ≥20 ngày UTC untouched
theo pre-registration), và diễn giải "quantum interference" được kiểm định
tách biệt khỏi hiệu ứng decoherence/gamma.
