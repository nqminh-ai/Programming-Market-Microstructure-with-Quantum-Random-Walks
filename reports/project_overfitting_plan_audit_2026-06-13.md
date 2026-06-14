# Audit Overfitting và Mức Độ Hoàn Thành Kế Hoạch

**Ngày audit:** 2026-06-13  
**Tài liệu đối chiếu:** `docs/ke_hoach_QRW_market_microstructure.md`  
**Phạm vi:** source code, tests, config, dữ liệu, calibration artifacts, báo cáo
Phase 1-3 và toàn bộ deliverable của Phase 1-6.

## Kết luận tổng quan

- **Cập nhật sau remediation:** ba lỗi trong calibration/cleaning đã được sửa.
  Live June 12 đã được rebuild còn `1,908` rows. Audit mới cho thấy QRW thua
  fair affine baseline: delta Brier `+0.016600`, CI
  `[0.006711, 0.025549]`. Các historical edge report June 1-7 hiện là stale
  và đã được chuyển vào `reports/archive/invalidated_2026-06-13/`. Derived
  processed/features June 1-7 đã bị xóa; raw data vẫn được giữ để rebuild.
- **Engineering correctness:** PASS trên phần đã triển khai. Test suite hiện tại
  đạt `46/46`.
- **Tiến độ kế hoạch:** trong 31 task, có **10 hoàn thành, 7 hoàn thành một
  phần/sai khác thiết kế, 14 chưa làm**. Một trong 14 task là dashboard tùy chọn.
- **Overfitting:** **RỦI RO CAO, CHƯA ĐỦ GIÁ TRỊ CONFIRMATORY.** Kết quả Phase 3
  hiện là bằng chứng exploratory, chưa phải chứng minh predictive edge vững.
- **Predictive edge hiện tại:** adaptive QRW thắng raw logistic dùng năm input
  gốc trên ngày 5-7 tháng 6 năm 2026, nhưng thua pairwise logistic ở cả ba ngày.
  Chưa có bằng chứng cho lợi thế riêng của QRW hoặc quantum dynamics.

Vấn đề chính không nằm ở unit test. Backtest hiện kết hợp preprocessing dùng
thông tin tick kế tiếp, holdout ba ngày đã được xem nhiều lần, historical OBI là
trade-flow proxy, và bootstrap ở cấp event trong dữ liệu có phụ thuộc mạnh.

## Các vấn đề nghiêm trọng

### Critical 1: June 5-7 không còn là untouched holdout

Trong code, split theo thời gian là hợp lý: June 1-3 để fit, June 4 để validation
và June 5-7 để prequential test
(`scripts/phase3_adaptive_decoherence_audit.py:420-548`). Tuy nhiên Phase 3 đã
được sửa và đánh giá lặp lại sau khi xem kết quả June 5-7. Ở cấp quy trình
nghiên cứu, ba ngày này đã trở thành development data.

Repo không có preregistration, immutable model hash, fresh-holdout manifest hay
multiple-testing correction. Dù vậy checkpoint vẫn kết luận edge đã được
"established" (`reports/phase3_checkpoint.md:137-165`).

**Ảnh hưởng:** CI và p-value trên June 5-7 chỉ có tính exploratory, bất kể số
lượng trade events lớn.

**Cần làm:** coi June 1-7 là development data, đóng băng toàn bộ pipeline và
đánh giá đúng một lần trên dữ liệu chưa từng xem. Cửa sổ hợp lý tiếp theo là
June 14-20, 2026 UTC hoặc nhiều tuần rời nhau.

### Critical 2: Phase 4 và Phase 5 gần như chưa triển khai

Kế hoạch yêu cầu CRW, GARCH, GBM và benchmark chung
(`docs/ke_hoach_QRW_market_microstructure.md:428-525`), sau đó là distribution,
variance, autocorrelation và tail tests
(`docs/ke_hoach_QRW_market_microstructure.md:529-657`). Các source file và CSV bắt
buộc đều chưa tồn tại.

Baseline mạnh nhất hiện có là pairwise logistic và nó tốt hơn QRW:

- QRW trừ raw-logistic Brier: `-0.013822`
- QRW trừ pairwise-logistic Brier: `+0.003401`
- Pairwise logistic có Brier thấp hơn trong cả June 5, June 6 và June 7.

Kết quả cũ được lưu để audit tại
`reports/archive/invalidated_2026-06-13/phase3_adaptive_decoherence_audit.json`.

**Ảnh hưởng:** dự án chưa thể trả lời câu hỏi trung tâm của kế hoạch là QRW có
mô hình hóa thị trường tốt hơn CRW/GARCH/GBM hay không.

### High 1: Tick cleaning từng có one-tick look-ahead, đã sửa trong code

Rolling mean và std dùng quá khứ, nhưng quyết định loại outlier cuối cùng dùng
`next_price = price.shift(-1)` để kiểm tra tick tiếp theo có quay về reference
hay không (`src/data/tick_processor.py:55-100`).

Checkpoint Phase 2 gọi bước này là causal
(`reports/phase2_checkpoint.md:11-12`). Validator chỉ kiểm tra metadata string
`trailing_with_one_tick_confirmation`
(`scripts/phase2_pipeline.py:318-331`, `488-507`), không kiểm tra information
availability thực tế.

**Ảnh hưởng:** việc row `t` có tồn tại trong dữ liệu processed phụ thuộc row
`t+1`. Điều này có thể thay đổi label next-tick, feature và event spacing bằng
thông tin tương lai. Toàn bộ historical processed/features hiện tại bị ảnh
hưởng.

**Trạng thái:** code hiện dùng
`trailing_with_past_only_regime_confirmation`; hai regression test đối kháng cắt
prefix đúng tại tick quyết định và thay đổi riêng tick `t+1`, xác minh future
ticks không làm đổi output quá khứ. June 12 đã rebuild. Bảy historical
days vẫn phải rebuild trước khi tái sử dụng.

### High 2: Predictive audit đang test nonlinear classifier, không phải multi-step QRW

Probability trong audit là:

`0.5 + 0.5 * exp(-gamma_t) * tanh(beta' x)`

được tính trực tiếp tại
`scripts/phase3_adaptive_decoherence_audit.py:145-178`. Density-matrix evolution
không được chạy cho từng prediction. Fast path simulator cũng chỉ lấy các bước
Bernoulli độc lập `+1/-1` từ probability này
(`src/models/adaptive_market_qrw.py:424-457`).

Exact density simulation có tồn tại và giữ state qua thời gian
(`src/models/adaptive_market_qrw.py:401-421`), nhưng không phải cơ chế được chấm
predictive edge. Benchmark 1,000 paths x 1,000 steps cũng đo Bernoulli sampler,
không đo 1,000 exact density paths (`scripts/phase3_pipeline.py:95-138`).

Raw logistic có cùng năm input nhưng không có intensity-dependent shrinkage.
Pairwise logistic 21 terms là nonlinear stress test gần hơn và đang thắng.

**Ảnh hưởng:** edge có thể đến từ nonlinear feature map hoặc dynamic confidence
scaling. Chưa thể quy nó cho quantum interference, density evolution hay QRW
path dynamics.

### High 3: Historical edge dùng synthetic trade imbalance, không dùng real LOB OBI

Bảy historical feature files dùng trade-volume-imbalance proxy. Chỉ artifact
June 12 dùng real LOB:

- Historical proxy rows trước cleanup: `47,627,346`
- Real-LOB rows: `1,937`
- Real-LOB aligned window: khoảng `118.5` giây

Proxy đã được shift một trade nên không có self-leakage
(`src/data/feature_engineer.py:305-343`). Tuy nhiên report hiện tại cho biết với
window 100, correlation proxy-real OBI chỉ `0.5925`, MAE `0.5583`, và `62.16%`
giá trị historical có `|OBI| > 0.95`
(`reports/phase2_synthetic_obi.md`).

**Ảnh hưởng:** edge hiện tại là edge trên signed trade flow có persistence mạnh,
không phải edge trên resting Level-2 order-book imbalance theo kế hoạch.

### High 4: Bootstrap đánh giá uncertainty quá hẹp

Audit resample block mean của mỗi 1,024 events như các đơn vị exchangeable
(`scripts/phase3_adaptive_decoherence_audit.py:291-328`). Có 4,482 event blocks
nhưng chỉ có ba test days.

Phương pháp này không capture day-level regime uncertainty, long-memory,
model-selection uncertainty và việc holdout đã được xem nhiều lần. Hàng triệu
trade phụ thuộc nhau không tương đương hàng triệu quan sát độc lập.

**Cần làm:** thu thập nhiều ngày hơn, công bố effect theo từng ngày, dùng
day/session clustered bootstrap hoặc stationary bootstrap, và đặt feature/
hyperparameter selection bên trong từng training fold.

### High 5: Evaluation biết trước event kế tiếp sẽ có price move

Audit chỉ chấm các row mà immediate next trade có price change khác zero
(`scripts/phase3_adaptive_decoherence_audit.py:58-81`). Flat moves bị loại và
chỉ có horizon 1.

Checkpoint xác nhận điều này ở `reports/phase3_checkpoint.md:100-102`, trong khi
kế hoạch yêu cầu hit rate ở horizon 1, 5, 10 cùng các distribution/path metrics
(`docs/ke_hoach_QRW_market_microstructure.md:493-510`).

**Ảnh hưởng:** task thực tế là "dự báo hướng với điều kiện đã biết sẽ có move",
không phải unconditional next-event prediction. Accuracy trên 90% không so sánh
được với model phải dự báo cả việc có move hay không.

**Cần làm:** dùng target ba lớp `down/flat/up`, hoặc two-stage model
`P(move)` rồi `P(up | move)`, và chấm horizon 1, 5, 10.

### High 6: Calibration có dấu hiệu chạm biên và weak identification

Real-LOB calibration chỉ có 112 moving warmup events cho sáu structural
parameters. Model chọn regularization yếu nhất `0.0001`, và `alpha_obi = 5.0`
chạm upper bound (`reports/live_window_BTCUSDT_2026-06-12.md:25-41`;
`src/models/adaptive_market_qrw.py:231-302`).

Synthetic calibration có `gamma_intensity = -2.0` chạm lower bound. Multiday
model có `gamma_intensity = -1.9154`, vẫn gần biên.

**Ảnh hưởng:** đây là dấu hiệu saturation, weak identification hoặc
regularization chưa đủ. Short real-LOB fit chỉ phù hợp để kiểm tra pipeline.

## Các vấn đề mức trung bình

### Medium 1: Gamma được fit trên event population khác evaluation

`tick_direction` forward-fill hướng price move gần nhất qua các flat-price
trades (`src/data/feature_engineer.py:98-109`). Gamma được ước lượng từ
autocorrelation của toàn bộ direction series
(`scripts/phase3_adaptive_decoherence_audit.py:88-115`), tạo `rho_1` khoảng
`0.95`.

Prediction lại chỉ chấm trên các row mà next price thay đổi. Rho cao một phần là
hệ quả construction của flat trades, chưa chắc là persistence của moving events.

### Medium 2: Artifact và checkpoint chưa nhất quán

- `docs/theory_notes.pdf` đã được đặt đúng thư mục; vẫn thiếu
  `docs/theory_summary.md`.
- LOB HDF5 hiện chứa 175 snapshots trong bốn interval rời nhau, trải khoảng
  3,807 giây. Live report chỉ mô tả final segment 113 snapshots/120 giây.
- Không có `.git`, root `README.md`, `Makefile`, `figures/`,
  `src/evaluation/`, `src/visualization/` hoặc `src/dashboard/`.

Các điểm này không trực tiếp tạo leakage nhưng làm giảm khả năng audit và
reproduce chính xác.

### Medium 3: Phase 3 có design deviations chưa được cập nhật vào kế hoạch

Implementation đang dùng:

- Hadamard coefficient `1 - 1/sqrt(2) ~= 0.292893`, không phải `0.4-0.6`.
- Full-dephasing `Var/T -> 1` với step `+1/-1`, không phải `0.5`.
- Phase/tanh OBI coin và CPTP basis-dephasing channel thay cho angle-linear coin
  và state-vector mixture ban đầu.

Các sửa toán học là hợp lý, nhưng file kế hoạch chưa được cập nhật. Phase 4 vẫn
lặp lại acceptance criterion CRW `Var/T ~= 0.5` sai với convention hiện tại
(`docs/ke_hoach_QRW_market_microstructure.md:518-525`).

### Medium 4: Reproducibility environment chưa hoàn chỉnh

`requirements.txt` dùng lower bounds thay vì pinned versions và chưa có các
dependency Phase 4-6 như `arch`, `statsmodels`, plotting/dashboard packages.
`pyproject.toml` chỉ chứa pytest settings.

## Các control đang làm đúng

- Synthetic trade imbalance loại current trade bằng one-trade lag.
- LOB attachment dùng backward as-of join với tolerance năm giây
  (`src/data/feature_engineer.py:397-404`).
- Feature normalization của multiday audit chỉ fit trên June 1-3.
- Regularization được chọn theo thứ tự thời gian trên June 4.
- QRW structural coefficients được giữ cố định trong June 5-7; chỉ bias được
  cập nhật từ dữ liệu quá khứ.
- Metadata phân biệt rõ real LOB và proxy imbalance.
- QRW engine có test normalization, symmetry, trace và decoherence.

Các control này giảm direct leakage nhưng không giải quyết các finding Critical
và High ở trên.

## Ma trận đối chiếu kế hoạch

| Task | Trạng thái | Kết quả audit |
|---|---|---|
| 1.1 DTQRW formalism | Hoàn thành | Note tồn tại và vượt 800 từ |
| 1.2 QRW vs CRW theory | Hoàn thành | Note tồn tại |
| 1.3 Market mapping | Hoàn thành | Note tồn tại |
| 1.4 Theory notebook | Hoàn thành | 6/6 code cells đã chạy, không error; verification PDF tồn tại |
| 1.5 Compiled theory document | Một phần | Có `docs/theory_notes.pdf`; thiếu `docs/theory_summary.md` |
| 2.1 Data source/environment | Hoàn thành | Config và working requirements tồn tại |
| 2.2 Tick collection | Hoàn thành | Bảy historical days, hơn 47.6M raw records |
| 2.3 LOB collection | Một phần | Có collector/HDF5; chỉ có một short synchronized real-LOB segment |
| 2.4 Tick cleaning | Một phần | Có artifacts/reports nhưng outlier filter dùng one-tick look-ahead |
| 2.5 Feature engineering | Một phần | Có feature; historical OBI là saturated trade-flow proxy |
| 2.6 Data tests | Hoàn thành | 22 tests pass; gồm hai test đối kháng cho look-ahead |
| 3.1 Pure-state QRW | Hoàn thành | Core engine và numerical tests tồn tại |
| 3.2 Coin operators | Một phần/sai khác | Có implementation nhưng khác thiết kế gốc |
| 3.3 Density matrix/decoherence | Hoàn thành | Có implementation và trace tests |
| 3.4 Market QRW | Một phần | Có calibration/exact simulation; predictive audit dùng closed-form classifier |
| 3.5 QRW validation | Hoàn thành/sai khác | Sáu tests pass với coefficient đã sửa đúng toán |
| 3.6 Performance | Một phần | Speed pass cho Bernoulli path sampler, không phải 1,000 exact density paths |
| 4.1 CRW baselines | Chưa làm | Thiếu `src/models/classical_rw.py` |
| 4.2 GARCH baseline | Chưa làm | Thiếu source và parameter artifact |
| 4.3 GBM baseline | Chưa làm | Thiếu source |
| 4.4 Benchmark suite | Chưa làm | Thiếu suite và `benchmark_results.csv` |
| 5.1 Distribution tests | Chưa làm | Thiếu KS/AD/Wasserstein suite và CSV |
| 5.2 Variance scaling | Chưa làm | Thiếu result CSV và figure |
| 5.3 Autocorrelation tests | Chưa làm | Thiếu Ljung-Box/ACF artifacts |
| 5.4 Tail analysis | Chưa làm | Thiếu Hill/VaR/CVaR artifact |
| 5.5 Results compiler | Chưa làm | Thiếu final table và scorecard |
| 6.1 Visualization suite | Chưa làm | Thiếu plot module và bảy figures |
| 6.2 Dashboard | Chưa làm, tùy chọn | Thiếu dashboard và screenshot |
| 6.3 Final technical report | Chưa làm | Thiếu `docs/final_report.pdf` |
| 6.4 Presentation slides | Chưa làm | Thiếu slides |
| 6.5 Repository documentation | Chưa làm | Thiếu root README, Makefile và pinned environment |

## Thứ tự công việc bắt buộc

### P0: Làm cho edge test tiếp theo hợp lệ

1. Thay future-aware outlier filter và regenerate toàn bộ data artifacts.
2. Đánh dấu June 1-7 là development-only.
3. Freeze features, model equations, hyperparameter grids, baselines, metrics,
   bootstrap và exclusion rules trước khi xem label mới.
4. Thu thập fresh test set, ưu tiên ít nhất 20 UTC days và có synchronized LOB.
5. Chạy protocol đúng một lần, lưu model/config hash cùng kết quả.

### P1: Hoàn thiện fair comparison

1. Implement CRW Simple/Biased/Correlated, GARCH và GBM của Phase 4.
2. Giữ pairwise logistic; thêm GAM, gradient boosting và classical baseline có
   cùng tanh/intensity-shrinkage link với QRW.
3. Chấm toàn bộ events và horizon 1, 5, 10.
4. Dùng day-clustered uncertainty và multiple-comparison correction.
5. Ablation riêng density evolution, adaptive decoherence, OBI, direction và
   intensity.

### P2: Hoàn thành deliverables nghiên cứu

1. Implement toàn bộ Phase 5.
2. Tạo figures, final report và slides của Phase 6.
3. Khôi phục đúng path tài liệu Phase 1.
4. Thêm root README, Makefile, pinned lock/freeze file và artifact manifests.

## Xác minh đã chạy

- `python -m pytest -q`: **46 passed**
- Notebook metadata: **6/6 code cells executed, 0 error outputs**
- Historical feature rows đã kiểm tra trước cleanup: **47,629,283**
- Current active feature rows sau cleanup: **1,908**
- Real-LOB aligned feature rows: **1,937**
- Đã kiểm tra riêng từng deliverable Phase 4-6: đều thiếu như bảng trên

## Đánh giá cuối

Dự án đã có một engineering prototype Phase 1-3 tốt và bằng chứng exploratory
rằng nonlinear adaptive link dự báo conditional next-move direction tốt hơn raw
logistic. Dự án **chưa chứng minh hợp lệ QRW có predictive superiority**.

Phát biểu có thể bảo vệ ở thời điểm hiện tại là:

> Engineering pass; có exploratory edge so với raw logistic chưa đủ mạnh; không
> có edge so với pairwise logistic; classical benchmark, robust inference,
> historical real-LOB validation và các phase nghiên cứu cuối vẫn chưa hoàn tất.
