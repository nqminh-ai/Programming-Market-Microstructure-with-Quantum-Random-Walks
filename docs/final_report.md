# Báo cáo cuối: Quantum Random Walk cho vi cấu trúc thị trường

## Trạng thái khoa học

**Kết luận hiện tại (cập nhật sau Phase 1–3): KHÔNG có bằng chứng cho bất kỳ
lợi thế dự báo nào của QRW. (1) Cơ chế "giao thoa lượng tử" (pha `alpha_phase`)
đóng góp BẰNG 0 trên cả ba asset. (2) Thành phần windowing/decoherence cổ điển
thắng baseline affine yếu nhưng THUA có ý nghĩa thống kê các baseline cổ điển
mạnh (OrderFlow AR(5), Logistic+Pairwise) trên CẢ BA asset — trên ETH thậm chí
xếp chót 7/7.** Ba nghiên cứu ablation/so-sánh (gắn nhãn exploratory,
[reports/research/](../reports/research/)) chỉ ra:

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
- **Windowing thua baseline mạnh (Phase 3).** Khi so với bộ baseline
  pre-registered (OrderFlow AR(5), Logistic+Pairwise, Hawkes…) dùng đúng cùng
  feature causal, windowed-QRW thua có ý nghĩa thống kê trên cả ba asset (BTC
  hạng 4/7, ETH 7/7, BNB 4/7). OrderFlow AR(5) thắng QRW ở mọi asset. Lợi thế
  vs affine ở §5b không sống sót. Xem §5c.

Các caveat cũ vẫn giữ nguyên: dữ liệu hoạt động ngắn, OBI là proxy trade-flow,
chưa chạy trên toàn bộ dataset gốc, và toàn bộ Phase 1–3 là exploratory (chưa
đóng băng protocol/pre-registration confirmatory).

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

## 9. Kết luận

Dự án đã sửa ngữ nghĩa đo QRW và quy tắc thống kê theo hướng có thể kiểm toán.
Một lỗi fit/predict mismatch nghiêm trọng cũng vừa được phát hiện và sửa: cơ
chế "quantum refinement" (`alpha_phase`) mà `calibrate()` có thể chọn trước
đây không hề được dùng ở bước dự đoán walk-forward thực tế, khiến verdict âm
trước đó (QRW kém hơn baseline) được tính từ một pipeline chưa từng thực sự
kiểm định cơ chế nó tuyên bố kiểm định.

Ba bước ablation/so-sánh (§5b–5c) đã làm rõ trọn vẹn: **(1)** đóng góp của cơ
chế giao thoa lượng tử (`alpha_phase`) đo được **bằng 0** trên cả ba asset —
bỏ pha cho kết quả giống hệt, ép pha lớn hơn làm xấu đi. **(2)** Fold-fragility
từng thấy trên BTC (edge đảo dấu ở fold ≥ 5) là một **bug** fit/predict
inconsistency trong `calibrate_bias`, đã sửa ở Phase 2; sau khi sửa,
windowed-QRW thắng baseline **affine** ổn định trên BTC/BNB (thua ETH). **(3)**
Nhưng lợi thế đó chỉ vì affine yếu: khi đấu với baseline cổ điển **mạnh**
(OrderFlow AR(5), Logistic+Pairwise) dùng đúng cùng feature causal, windowed-QRW
**thua có ý nghĩa thống kê trên cả ba asset** (ETH xếp chót 7/7). Tổng hợp:
**chưa có bằng chứng cho bất kỳ lợi thế dự báo nào của QRW — dù là cơ chế
lượng tử hay thành phần windowing cổ điển; một logistic tự hồi quy đơn giản
(OrderFlow AR) đánh bại nó ở mọi asset.** Các hạn chế còn lại: chưa chạy trên
toàn bộ 10 ngày gốc do giới hạn bộ nhớ (§8-6); dữ liệu hoạt động vẫn ngắn và
OBI chưa phải L2 order-book imbalance thật.

Kết luận khoa học cuối cùng chỉ nên được đưa ra sau khi: protocol được đóng
băng, provenance khớp commit/data hash, walk-forward được chạy lại thành công
trên toàn bộ dataset gốc (hoặc dataset multi-day mới ≥20 ngày UTC untouched
theo pre-registration), và diễn giải "quantum interference" được kiểm định
tách biệt khỏi hiệu ứng decoherence/gamma.
