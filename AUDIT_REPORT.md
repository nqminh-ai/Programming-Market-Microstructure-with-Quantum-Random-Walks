# AUDIT: AI Quantum — QRW cho vi cấu trúc thị trường — 86/100

> Phạm vi chấm: **CHỈ code + notebook** (`src/`, `scripts/`, `tests/`, `notebooks/`).
> Bỏ qua report/slide/deploy theo yêu cầu. Repo: `d:\Project made by me\AI_Quantum\Quantum 1`.
> Inventory: **128 file `.py` + 1 notebook** (≈26.590 dòng src+scripts, 7.186 dòng test),
> **383 hàm test** trong 42 file test. Không có lỗi mức **CRITICAL** → trần điểm 65 **không** áp dụng.

---

## 1. Bảng điểm

| Hạng mục                                   |      Tối đa |           Trừ |        Cuối | Lý do trừ điểm chính                                                                      |
| -------------------------------------------- | ------------: | -------------: | -----------: | ---------------------------------------------------------------------------------------------- |
| 1. Tính đúng đắn logic & phương pháp |            25 |            −2 | **23** | Chọn hyperparameter L2 bằng*training loss* trong script mang kết luận đầu bảng (M2)   |
| 2. Xử lý dữ liệu & data leakage          |            15 |            −2 | **13** | `merge_asof` cho phép khớp cùng-instant, rủi ro leakage lý thuyết (M1)                 |
| 3. Chất lượng code & kiến trúc          |            15 |            −5 | **10** | 62 hàm >80 dòng;`calibrate` 575 dòng; nhiều `main()`/`tab_*` 240–265 dòng (H1, L3) |
| 4. Reproducibility                           |            12 |            −1 | **11** | Lock chỉ pin cho CPython 3.14/Windows (L2)                                                    |
| 5. Vệ sinh notebook & mạch trình bày     |            10 |            −1 |  **9** | Chỉ 1 notebook; output đã commit sẵn (bề mặt hẹp, không lỗi thực chất)              |
| 6. Hiệu năng & tối ưu                    |             8 |            −1 |  **7** | Vòng lặp Python O(n) trên event trong`_marked_hawkes_state` (M3)                          |
| 7. Kiểm thử & xác thực kết quả         |             8 |            −1 |  **7** | Không có đo coverage; một số script/dashboard mỏng test                                  |
| 8. Tài liệu, naming, readability           |             7 |            −1 |  **6** | Vài magic constant inline (L4); còn lại xuất sắc                                          |
| **Tổng**                              | **100** | **−14** | **86** | Nợ chính là độ dài hàm, không phải sai phương pháp                                 |

Quy đổi: **86  (75–89)**. Không CRITICAL. Phần kéo điểm nặng nhất là
maintainability (hàm quá dài), **không** phải tính đúng đắn hay leakage — hai trục đó đạt mức hiếm thấy.

---

## 2. Findings theo mức độ

| ID | Mức   | File:dòng                                                                       | Mô tả lỗi                                                                                                                                                                                                                   | Điểm trừ | Cách sửa                                                                                                                                                |
| -- | ------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------: | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H1 | HIGH   | `src/models/qrw_market_sim.py:94`                                              | Hàm`calibrate` dài **575 dòng**; toàn repo có **62 hàm >80 dòng, 118 hàm >50 dòng**. Không test được từng phần, khó review, dễ hồi quy khi sửa                                                |         −4 | Tách`calibrate` thành các bước thuần: ước lượng ρ→γ, resolve tick size, fit coin, validate. Mỗi bước một hàm có test riêng          |
| M1 | MEDIUM | `src/data/feature_engineer.py:412-430`                                         | `merge_asof(direction="backward")` với `allow_exact_matches=True` (mặc định) cho phép tick khớp snapshot LOB **cùng timestamp** → rủi ro same-instant leakage nếu feed phát tick và LOB update trùng ns |       −1.5 | Đã tự flag (audit M6). Khi có L2 thật: đặt`allow_exact_matches=False`, và sửa fixture test đang cố tình tái dùng timestamp                |
| M2 | MEDIUM | `scripts/research/horizon_label_baselines.py:216-231`                          | Chọn regularization L2 bằng**training log loss** (dòng 223–225, không có penalty) → luôn chọn λ nhỏ nhất, tức không thực sự regularize. Nằm trong script sinh con số "order flow 64% directional"      |         −2 | Chọn λ bằng validation fold riêng (chia train→val→test), không bằng training loss                                                                 |
| M3 | MEDIUM | `src/evaluation/directional_baselines.py:235`                                  | `_marked_hawkes_state` dùng `for index in range(len(features))` — vòng lặp Python O(n) để tích lũy kernel mũ                                                                                                      |         −1 | Kernel mũ có công thức đệ quy đóng; dùng`scipy.signal.lfilter` hoặc numba. (Giảm nhẹ: chỉ chạy trên daily fold, không phải 227M dòng) |
| L1 | LOW    | `src/strategy/optimizer.py:282`                                                | `import warnings` đặt trong thân method thay vì đầu module                                                                                                                                                             |       −0.3 | Đưa import lên đầu file                                                                                                                              |
| L2 | LOW    | `requirements.lock:1`                                                          | Lock ghi rõ "resolved on CPython 3.14 / Windows" → không tái lập trực tiếp trên Linux/macOS hoặc Python khác                                                                                                         |         −1 | Cung cấp thêm lock đa nền tảng, hoặc pin qua`pip-tools`/`uv` có hash cross-platform                                                            |
| L3 | LOW    | `src/dashboard/platform.py:1143`, `scripts/pipelines/phase6_pipeline.py:411` | `tab_optimizer` (265 dòng), `main` (243 dòng) trộn orchestration + trình bày trong một hàm                                                                                                                          |       −0.7 | Tách phần dựng dữ liệu ra khỏi phần render/ghi file                                                                                                |
| L4 | LOW    | `scripts/research/horizon_feasibility.py:58-87`                                | Kịch bản phí (5/4/2/0 bps) và`PLAUSIBLE_ACCURACY_CEILING=0.60` là hằng số inline                                                                                                                                      |       −0.3 | Có docstring giải thích nên rủi ro thấp; đưa vào config/CLI nếu cần quét                                                                      |

Tổng trừ có bằng chứng: **−11.1** trên các finding + **−2.9** phân bổ đều cho nợ độ-dài-hàm/coverage
ở mục 3 và 7 (đã gộp vào bảng điểm mục 1). Điểm cuối **86**.

### 2b. Bảng kiểm bắt buộc (trả lời từng mục)

| Mục kiểm                                                   | Kết luận                                | Bằng chứng                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------ | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Fit scaler/encoder trước khi split?                        | **Không**                          | `src/evaluation/directional_baselines.py:375-377` — mean/scale lấy **chỉ từ train**; `scripts/research/horizon_label_baselines.py:204-208` `_standardise` fit trên train, transform sang test                                                                                                           |
| Target encoding không CV?                                   | **Không áp dụng**                | Không dùng target/mean encoding; feature là OBI/direction/intensity số học thuần                                                                                                                                                                                                                                   |
| Feature chứa thông tin tương lai?                        | **Không**                          | Mọi feature causal:`tick_direction` = ffill sign trong segment (`feature_engineer.py:106-117`), synthetic OBI/VWAP `.rolling(...).shift(1)` (`feature_engineer.py:341-346`), target = move `i→i+1` (`directional_baselines.py:73-78`)                                                                      |
| Split có stratify / shuffle sai cho time-series?            | **Đúng cách**                    | Chia**theo thời gian**, không shuffle: `chronological_day_split` (`directional_baselines.py:85-108`), `evaluate_models` `[:split]/[split:]` (`horizon_label_baselines.py:234-243`)                                                                                                                     |
| Test set bị dùng để tune?                                | **Không**                          | Tune trên validation (`directional_baselines.py:426-497`), score trên test chưa đụng (`score_directional_baselines`); optimizer `grid_search` chỉ đọc `train_df`, `evaluate_out_of_sample` áp param đông cứng (`optimizer.py:124,217`). *Ngoại lệ M2*: 1 script chọn λ bằng training loss |
| Metric hợp imbalance + có baseline?                        | **Có**                             | So majority-class chứ không so 50% (`horizon_label_baselines.py:20-22,251`); bộ 6 baseline mạnh + GBM/CRW/RW; Brier + log loss + Deflated Sharpe cho selection bias (`deflated_sharpe.py`)                                                                                                                       |
| Random seed set đủ thư viện?                             | **Có**                             | `np.random.SeedSequence(...).spawn(...)` (`strong_baseline_comparison.py:168`, `alpha_phase_ablation.py:374`), notebook `default_rng(2026)` + ghi seed ra JSON. Không dùng torch/sklearn nên phủ đủ thư viện đang dùng                                                                                 |
| Hard-coded path tuyệt đối / credential lộ?               | **Không**                          | Grep`C:\\`/`D:\\` trong `.py` = 0; default path là relative (`horizon_feasibility.py:595`); credential qua env `SSI_CONSUMER_SECRET` + file ngoài Git (`ssi_live_collector.py:63-73`)                                                                                                                      |
| Notebook chạy sai thứ tự / cell chết / output sót?      | **Không**                          | `execution_count` 1→6 tăng đơn điệu, 0 cell chết, chỉ có stream output, import gom ở cell #1                                                                                                                                                                                                                 |
| Code lặp / hàm >50 dòng / nesting sâu / biến 1 ký tự? | **Có (hàm dài)**                 | 118 hàm >50 dòng (H1). Biến 1 ký tự chỉ trong ngữ cảnh toán học hợp lý (`x,y` trong Roll cov, `p` xác suất)                                                                                                                                                                                            |
| Silent failure (`except: pass`, fillna bừa)?              | **Không**                          | 0`except: pass`; 2 `except Exception` đều log `exc_info=True` khi đóng socket (`ssi_live_collector.py:252,542`); mọi `fillna` có ngữ nghĩa rõ (obi→0, mid→price)                                                                                                                                    |
| Vòng lặp Python trên DataFrame thay vì vectorize?        | **Có (1 chỗ, tác động thấp)** | M3. Ngoài ra`iterrows` chỉ trên frame ~6 dòng để vẽ (`statistical_tests.py:558`); `.apply(pd.to_numeric)` là ép kiểu theo cột, không phải per-row                                                                                                                                                     |
| Kết luận over-claim so với code?                          | **Không**                          | Ngược lại — under-claim có kỷ luật: mọi artifact gắn`EXPLORATORY_ONLY_NOT_CONFIRMATORY` (`horizon_feasibility.py:651`), verdict tự ghi "điều kiện cần không đủ" (dòng 583-587)                                                                                                                     |
| Dependency có file + pin version?                           | **Có**                             | `requirements.txt` exact pin toàn bộ + `requirements.lock` transitive. Hạn chế: lock đơn nền tảng (L2)                                                                                                                                                                                                       |

---

## 3. ĐÃ TỐT

- **Causal feature engineering không nhân nhượng.** Synthetic OBI và VWAP đều `.rolling(window).sum().shift(1)` theo từng `segment_id`, loại chính trade hiện tại khỏi feature của nó — `src/data/feature_engineer.py:341-346`, `364-366`. Autocorrelation cũng không bắc cầu qua gap — `feature_engineer.py:443-447`.
- **Split thời gian đúng chuẩn, scaler chỉ fit trên train.** `chronological_day_split` train→val→test disjoint theo timestamp; tune trên validation, score trên test bất khả xâm phạm — `src/evaluation/directional_baselines.py:85-108, 375-377, 526-537`.
- **Chống selection bias bằng Deflated Sharpe** thay cho penalty ad hoc, có ghi rõ lý do thay thế trong comment — `src/strategy/optimizer.py:43-72, 147-153`.
- **Ước lượng chi phí giao dịch có nền lý thuyết:** half-spread bằng **Roll (1984)** đảo bid-ask bounce (`horizon_feasibility.py:119-175`), đo cả **adverse selection** qua realised half-spread (`horizon_feasibility.py:288-344`), và tự bác một chỉ số cũ sai 28–109× với comment kiểm toán được (`horizon_feasibility.py:178-193`).
- **Provenance đóng vào từng artifact:** git commit + SHA-256 input + phiên bản Python + seed ghi thẳng ra JSON — `horizon_feasibility.py:649-662`, `strong_baseline_comparison.py:227`.
- **Notebook là bằng chứng, không phải nháp:** kiểm unitarity cả symbolic (`sympy`) lẫn numeric với assertion `<1e-10`, seed 2026, lưu kết quả JSON, và tự hạ giọng "kiểm tra implementation, chưa phải bằng chứng phù hợp thị trường" — `notebooks/01_theory_verification.ipynb` cell #1,#3,#9,#11.
- **Test dày và đúng trọng tâm:** 383 hàm test gồm leakage, provenance, thống kê, và "fake world" ghép đồng hồ với collector để test luật thu dữ liệu **không cần mạng** — `tests/test_collect_confirmatory.py`, `tests/test_directional_baselines.py`.
- **Comment giải thích "tại sao", không phải "cái gì":** ví dụ vì sao chunk hóa (8GB→1 scalar), vì sao không downcast, vì sao đơn vị timestamp suy từ magnitude — `horizon_feasibility.py:106-108, 178-194, 219-227`.

## 4. CHƯA TỐT

- **`calibrate` 575 dòng** gánh quá nhiều trách nhiệm (ρ→γ, tick size, fit, validate) trong một scope — `src/models/qrw_market_sim.py:94`. Hậu quả: không thể unit-test riêng bước calibrate γ tách khỏi bước fit; đây đúng là loại hàm mà bug `calibrate_bias` lịch sử từng ẩn trong đó.
- **Chọn λ bằng training loss** trong script sinh con số đầu bảng — `scripts/research/horizon_label_baselines.py:223-231`. Hậu quả: bước "tune regularization" thực chất vô hiệu (luôn về λ nhỏ nhất); may là kết luận dựa trên so-với-majority nên không sụp, nhưng phương pháp không khớp phần còn lại của repo (vốn tune trên validation).
- **Rủi ro same-instant leakage còn để ngỏ** — `src/data/feature_engineer.py:412-430`. Hậu quả: khi cắm L2 LOB thật, một tick trùng ns với snapshot có thể đọc chính update sinh ra nó; hiện chỉ được chặn bởi đặc thù `tick_processor` không tạo collision.
- **Nhiều `main()`/`tab_*` 230–265 dòng** trộn logic tính toán với I/O/render — `phase6_pipeline.py:411`, `platform.py:1143`, `research_dashboard.py:232`. Hậu quả: khó test phần tính toán mà không dựng cả Streamlit/ghi file.
- **Vòng lặp Python trên event** — `directional_baselines.py:235`. Hậu quả: chậm tuyến tính; chấp nhận được ở daily fold nhưng sẽ là nút cổ chai nếu mở rộng Hawkes ra toàn store.

## 5. ĐÃ TỐI ƯU

- **Chunk hóa để chặn RAM ở mức hằng số:** `expected_absolute_move`, `roll_half_spread`, `realised_half_spread`, `measure_half_spread` đều cộng dồn theo chunk 10M, biến ~6–8GB cấp phát tạm thành một scalar mà **không đổi kết quả** (tổng chạy / đếm chạy) — `horizon_feasibility.py:249-285, 119-175, 288-344`.
- **Nạp theo cột + downcast float32 + tách chuỗi trade_sign** để không nổ 11GB khi pandas biến 227M string thành object — `horizon_feasibility.py:106-116` (qua `src/data/feature_store.py`).
- **Chỉ convert hai điểm biên timestamp** thay vì cả cột (tiết kiệm ~1.7GB/lần) vì unit detection chỉ cần magnitude lớn nhất — `horizon_feasibility.py:232-239`.
- **Overlap 2 dòng giữa các chunk** để cặp sai phân straddle mép chunk vẫn được tạo, giữ đúng covariance — `horizon_feasibility.py:147-149`.
- **Heavy-tail unitary shift áp bằng FFT O(N log N)** thay vì nhân ma trận O(N²) — `src/models/heavy_tail_unitary.py`.
- **Seed spawn thay vì +k thủ công** ở các baseline độc lập, tránh tương quan seed — `strong_baseline_comparison.py:168-169`.

## 6. CHƯA TỐI ƯU

| Vị trí                                        | Hiện tại                                                                                                                                             | Nên làm                                                                                     | Lợi ích ước tính                                                                     |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `src/evaluation/directional_baselines.py:235` | `for index in range(len(features))` tích lũy kernel mũ                                                                                            | `scipy.signal.lfilter` với hệ số phân rã, hoặc numba `@njit`                        | ~10–50× trên fold lớn; mở đường chạy Hawkes toàn store                          |
| `scripts/research/*` (nhiều file)            | Mỗi hàm đo (`expected_absolute_move`, `roll_half_spread`, `realised_half_spread`) quét lại **toàn bộ** cột giá một lượt riêng | Gộp các thống kê per-horizon vào**một** vòng chunk chia sẻ mảng giá đã load | Giảm số lần đọc/convert mảng giá từ ~N_horizon xuống 1; đáng kể ở 227M dòng |
| `src/evaluation/statistical_tests.py:558`     | `iterrows` để vẽ (frame ~6 dòng)                                                                                                                 | Không cần sửa gấp; nếu muốn:`itertuples`                                              | Không đáng kể (frame nhỏ) — liệt kê để minh bạch, không tính là nợ         |
| `src/models/qrw_market_sim.py:94`             | `calibrate` gọi nhiều `np.diff`/mask lặp trên cùng mảng                                                                                      | Tính một lần, tái dùng                                                                   | Nhỏ; ưu tiên tách hàm (H1) hơn là vi tối ưu                                      |

## 7. Top 10 việc cần sửa (xếp theo điểm thu hồi được)

| #  | Việc                                                                                        |              Điểm lấy lại | Công sức |
| -- | -------------------------------------------------------------------------------------------- | ----------------------------: | ---------- |
| 1  | Tách`calibrate` (575 dòng) và các hàm >80 dòng thành đơn vị có test riêng (H1) |                            ~4 | L          |
| 2  | Sửa M2: chọn λ bằng validation fold, không bằng training loss                          |                            ~2 | S          |
| 3  | Đặt`allow_exact_matches=False` + sửa fixture khi có L2 thật (M1)                      |                          ~1.5 | M          |
| 4  | Cung cấp lock đa nền tảng (uv/pip-tools hash) (L2)                                       |                            ~1 | S          |
| 5  | Vectorize`_marked_hawkes_state` bằng `lfilter`/numba (M3)                               |                            ~1 | M          |
| 6  | Tách logic tính toán khỏi render trong`main()`/`tab_*` dài (L3)                     |                          ~0.7 | M          |
| 7  | Thêm đo coverage (pytest-cov) vào CI/Makefile                                             |                          ~0.5 | S          |
| 8  | Đưa magic constant phí/ceiling vào config/CLI (L4)                                       |                          ~0.3 | S          |
| 9  | Chuyển`import warnings` lên đầu module (L1)                                            |                          ~0.3 | S          |
| 10 | Gộp các lượt quét cột giá per-horizon vào một vòng chunk chung                     | 0 (perf, không tính điểm) | M          |
|    | **Tổng thu hồi khả dĩ**                                                            |          **~11.3 / 14** |            |

## 8. Phán quyết

Đây là code data-science **trên mức đồ án môn học rõ rệt** và tiệm cận chất lượng nghiên cứu có thể xuất bản
về mặt kỷ luật phương pháp: split thời gian đúng, scaler chỉ fit trên train, baseline mạnh có thật, provenance

+ seed đóng vào từng artifact, và — hiếm nhất — code **tự bác bỏ chính mình** thay vì tô hồng (Roll thay cho
  half-spread sai, Deflated Sharpe thay penalty ad hoc, nhãn `EXPLORATORY` gắn cả lên kết quả có lợi cho tác giả).
  Không tìm thấy leakage trong đường dẫn sinh kết luận, không hard-code credential, không over-claim. Điểm bị kéo
  xuống 86 gần như hoàn toàn vì **nợ maintainability**: 62 hàm quá dài mà đỉnh là `calibrate` 575 dòng, cộng một
  sai lệch phương pháp nhỏ (chọn λ bằng training loss) trong đúng script mang con số 64%.

**Rủi ro lớn nhất nếu nộp/ship nguyên trạng:** không nằm ở tính đúng của con số đã báo cáo, mà ở chỗ những hàm
600 dòng như `calibrate` là nơi bug tiếp theo sẽ ẩn mà test hiện tại khó bắt — đúng vẫn là lớp lỗi mà dự án đã
một lần dính (`calibrate_bias`). Và với M2, bất kỳ ai tái dùng `horizon_label_baselines` cho tài sản/horizon khác
sẽ vô tình chạy "không regularization" mà không hay biết. Cả hai đều sửa được với công sức nhỏ–vừa và không đe
dọa các kết luận exploratory hiện có.
