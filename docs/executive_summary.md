# Tóm tắt điều hành — QRW cho vi cấu trúc thị trường

*Dành cho hội đồng. Mọi con số trong tài liệu này đều truy ngược được về một
file trong [`reports/research/`](../reports/research/) có ghi git commit, SHA-256
của input và seed. Chi tiết phương pháp: [`final_report.md`](final_report.md).*

---

## 1. Kết luận trong năm dòng

Chúng tôi hỏi một câu hỏi có thể trả lời **sai**: *cơ chế lượng tử trong quantum
random walk có mang lại lợi thế dự báo thật trên dữ liệu thị trường không?*

Câu trả lời là **không** — và chúng tôi chứng minh điều đó bằng bốn thí nghiệm
độc lập thay vì bằng lập luận:

1. **Giao thoa lượng tử đóng góp bằng 0.** Tắt pha (`alpha_phase = 0`) không làm
   đổi kết quả tới ≥5 chữ số trên cả ba tài sản.
2. **Phần "thắng" còn lại là windowing/decoherence cổ điển**, và nó **thua có ý
   nghĩa thống kê** các baseline cổ điển mạnh trên cả ba tài sản.
3. **Một "lợi thế" từng công bố (+0,049889) là bug**, đã truy ra nguyên nhân,
   sửa, và thay bằng con số tái lập được (−0,013091 trên toàn bộ 32,4 triệu tick).
4. **Sắc thái duy nhất:** ở endpoint chính đăng-ký-trước (marginal CRPS), QRW
   thua *ít dứt khoát hơn* — dẫn đầu trên ETH, nhưng chỉ hạng 3/6 trên BTC và
   4/6 trên BNB. Không nhất quán, và **vẫn không đến từ pha**.
5. **Và kể cả nếu có dự báo tốt cũng chưa giao dịch được:** ở horizon dự án dùng,
   ngưỡng hoà vốn **vượt 100%** — dự đoán đúng hoàn hảo vẫn lỗ vì phí giao dịch
   lớn hơn biên độ giá.

> **Giá trị của dự án không nằm ở "QRW thắng", mà ở chỗ nó chỉ ra chính xác
> *tại sao* các tuyên bố ưu thế lượng tử trên dữ liệu tài chính thường không
> sống sót — và cung cấp bộ công cụ để kiểm chứng điều đó.**

---

## 2. Vì sao câu hỏi này khó, và chúng tôi xử lý thế nào

| Cạm bẫy phổ biến | Hệ quả | Cách dự án chặn |
|---|---|---|
| So với baseline yếu (random walk, affine) | Mọi model đều "thắng" | Bộ 7 baseline **đăng ký trước**, có OrderFlow AR(5), Logistic+Pairwise, Marked Hawkes |
| Không cô lập được phần "lượng tử" | Không biết cái gì tạo ra kết quả | Toggle `alpha_phase` — **xoá đúng** giao thoa (xem §3.1) |
| Chọn hyperparameter đánh giá cho đẹp | Kết quả không tái lập | Quét fold 2–8, báo cáo **toàn bộ** |
| fit/predict lệch công thức | Số đo vô nghĩa nhưng trông hợp lý | Đây chính là bug đã tìm ra (§3.2) |
| Chạy mẫu nhỏ rồi ngoại suy | Kết luận không bền | Chạy lại trên **toàn bộ** 32,4M tick |
| Đo độ chính xác rồi coi như đã kiếm được tiền | Bỏ qua phí giao dịch — cái thường lớn hơn cả biên lợi nhuận | Tính ngưỡng hoà vốn từ phí **đo được**, không giả định (§3.6) |
| Nhãn dự báo chồng lấp nhau | Khoảng tin cậy hẹp một cách sai lệch | Cửa sổ **không chồng lấp**, chấp nhận mất cỡ mẫu (§3.6) |

---

## 3. Bằng chứng chốt

### 3.1 Pha lượng tử đóng góp bằng 0

Khi `alpha_phase = 0`, coin SU(2) suy biến thành phép quay SO(2) **giao hoán** —
xoá **đúng** số hạng giao thoa, không phải xấp xỉ. Vì vậy toggle này là một
phép cô lập cơ chế hợp lệ về mặt toán học.

| Tài sản | Đóng góp riêng của pha (ΔBrier) | So với ngưỡng 10⁻⁴ |
|---|---:|---|
| BTCUSDT | −0,000000 | bằng 0 |
| ETHUSDT | −1,09 × 10⁻⁵ | nhỏ hơn 9 lần |
| BNBUSDT | +3,15 × 10⁻⁷ | nhỏ hơn 300 lần |

Ép pha lớn hơn làm dự báo **xấu đi đơn điệu**. Model bỏ pha hoàn toàn cho Brier
**giống hệt** model đầy đủ.

📄 [`alpha_phase_ablation_*_postfix.md`](../reports/research/)

### 3.2 Fold-fragility là bug — đã truy nguyên và sửa

Con số cũ đảo dấu khi đổi số fold: dấu hiệu kinh điển của kết quả không thật.
Nguyên nhân: `calibrate_bias()` **fit theo công thức cổ điển** nhưng **dự báo
theo công thức lượng tử** — cùng lớp lỗi mà audit nội bộ đã gắn nhãn C1/C2.

Sau khi sửa, edge ổn định ở **mọi** số fold:

| folds | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| BTC edge | −0,01271 | −0,01287 | −0,01293 | −0,01295 | −0,01283 | −0,01281 |

📄 commit `0a502fd`

### 3.3 Thua baseline cổ điển mạnh — trên cả ba tài sản

Cùng feature causal, cùng train/val/test, cùng grid hyperparameter.

| Tài sản | Hạng của QRW | Model tốt nhất | Brier tốt nhất | Brier QRW | Khoảng cách |
|---|:--:|---|---:|---:|---:|
| BTCUSDT | **4/7** | Logistic L2 + Pairwise | 0,049647 | 0,101923 | +0,052277 |
| ETHUSDT | **7/7** | OrderFlow AR(5) | 0,065707 | 0,100145 | +0,034438 |
| BNBUSDT | **4/7** | OrderFlow AR(5) | 0,146578 | 0,176656 | +0,030078 |

Mọi khoảng cách đều có 95% CI **không chứa 0**. **OrderFlow AR(5) — một hồi quy
tuyến tính 5 hệ số — thắng QRW trên cả ba tài sản.**

📄 [`strong_baseline_*.md`](../reports/research/)

### 3.4 Chạy toàn bộ dữ liệu: thay số cũ bằng số tái lập được

Giới hạn "chưa chạy được full dataset" đã được đóng bằng nạp theo cột + hạ
xuống float32 (32.439.057 dòng, 1070 MB RAM).

| | Con số cũ (trước khi sửa bug) | Con số hiện tại |
|---|---:|---:|
| edge | **+0,049889** *(không tái lập được)* | **−0,013091** (95% CI [−0,01332, −0,01287]) |

📄 [`full_dataset_confirmation.md`](../reports/research/full_dataset_confirmation.md)

### 3.5 Sắc thái trung thực: endpoint CRPS

Ở endpoint **chính đăng-ký-trước** (mean fixed-origin marginal CRPS), bức tranh
khác — và chúng tôi báo cáo nó dù nó không ủng hộ kết luận chung:

| Tài sản | Hạng QRW | Model tốt nhất | Window QRW thắng |
|---|:--:|---|:--:|
| ETHUSDT | **1/6** | QRW Adaptive | 3/5 |
| BTCUSDT | 3/6 | GBM | 0/5 |
| BNBUSDT | 4/6 | CRW Correlated | 1/5 |

Đây là chiều QRW thua **ít dứt khoát nhất** — không phải chiều QRW thắng. Nó
dẫn đầu trên đúng một trong ba tài sản, và thua đậm ở các window biến động
**cao** vì model không mô hình hoá volatility. §3.1 đã cho thấy phần không thua
này **cũng không đến từ pha**.

> **Số BNB đã đổi (rà soát 2026-07-22).** Bản trước ghi BNB hạng 2/6. Rà soát
> repo phát hiện artifact đó trỏ vào một file trong thư mục tạm của phiên làm
> việc, **đã không còn tồn tại** — input không tái lập được. Đã dựng lại dữ liệu
> BNB (31,5 triệu dòng) **bên trong repo**, chạy lại, và kết quả **xấu hơn cho
> QRW**: 4/6 thay vì 2/6. Chúng tôi báo số mới.

📄 [`marginal_crps_*.md`](../reports/research/)

### 3.6 Có giao dịch được không? Kỹ năng và tiền ở hai đầu đối lập

Các phần trên hỏi "mô hình nào dự báo tốt hơn". Phần này hỏi câu quan trọng hơn
với người ngoài ngành: **dự báo tốt hơn thì có kiếm được tiền không?**

Với horizon `h`, một cược hướng đúng `p` phần trăm thu về `(2p−1)·E|biến động|`
trước phí. Ở horizon dự án đang dùng — **1 tick** — biến động trung bình chỉ
bằng 0,0005 (BTC) lần chi phí một vòng giao dịch:

> **Ngưỡng hoà vốn vượt 100%. Một mô hình dự đoán đúng *hoàn hảo* vẫn lỗ.**

Đây là giới hạn của *horizon*, không phải của mô hình — không kỹ thuật nào cứu
được. Chúng tôi đổi nhãn sang lợi suất qua `h` tick và chạy lại trên các cửa sổ
**không chồng lấp**:

| Horizon | BTC | Lớp đa số | Lãi ròng/lệnh |
|---|---:|---:|---:|
| 1.000 (~49 giây) | **65,7%** | 51,2% | **−2,56 bps** |
| 50.000 (~41 phút) | 55,7% | 55,7% *(không hơn hằng số)* | vẫn âm |

**Order flow có sức dự báo thật** — 65,7% là con số đáng kể. Nhưng ở horizon đó
giá chưa dịch đủ để trả phí; kéo dài ra tới khi biên độ đủ lớn thì kỹ năng biến
mất. **Không horizon nào, trên bất kỳ tài sản nào, đạt hoà vốn.**

Con số 65,7% được **kiểm tra chứ không báo cáo thẳng**: nó truy về `tick_direction`
(autocorr 0,965) và tương quan sụp từ +0,329 xuống +0,011 ở cửa sổ tương lai kế
tiếp — dấu hiệu của tác động order flow thật, chứ nếu là rò rỉ dữ liệu thì sẽ
duy trì.

Phần này cũng phát hiện **ba lỗi đo lường** khiến chiến lược demo trông có lãi:
số lệnh bị thổi phồng 15–34×, một t-statistic bị gọi nhầm là "Sharpe", và bộ dò
tham số **quên trừ phí giao dịch**. Sửa xong, profit factor 265 → **0,095**, lãi
ròng +0,04% → **−4,2%**.

📄 [`horizon_feasibility_*.md`](../reports/research/) ·
[`horizon_edge_*.md`](../reports/research/)

---

## 4. Đóng góp kỹ thuật đáng giữ lại

Bốn thành phần dưới đây có giá trị độc lập với kết luận tiêu cực ở trên.

### 4.1 Lévy unitary shift — vá đúng khiếm khuyết vật lý của model

Walk lượng tử thường có marginal **bimodal/ballistic** với giá đỡ compact: sai
lệch **định tính** với thị trường, không phải sai lệch tham số. Chúng tôi tổng
quát hoá toán tử shift trong không gian động lượng thành φ_α(k) = sign(k)·|k|^α.
Mọi trị riêng vẫn nằm trên đường tròn đơn vị ⟹ **unitary theo cấu trúc, không
phải xấp xỉ**; α = 1 tái tạo *đúng* shift ±1 (nên đây là tổng quát hoá chặt);
α < 1 cho hopping lũy thừa |x|^−(1+α). Áp dụng O(N log N) bằng FFT.

Tỉ lệ đuôi q999/q75 (horizon 50 tick, lattice 16.001):

| Tài sản | Thị trường thật | Walk thường | Lévy (α tốt nhất) |
|---|---:|---:|---:|
| BTCUSDT | 5,07 | **1,15** | 7,74 (α = 0,7) |
| ETHUSDT | 3,09 | **1,15** | 3,10 (α = 0,9) |
| BNBUSDT | 4,00 | **1,15** | 3,10 (α = 0,9) |

**Giới hạn ghi rõ:** đây là kiểm định **hình dạng phân phối**, không phải kỹ
năng dự báo. Lévy-stable/Student-t cổ điển cũng khớp đuôi được, và α là tham số
*fit*, khác nhau theo tài sản. Nó đóng khoảng trống **cơ chế**, **không** phải
bằng chứng ưu thế lượng tử.

📄 [`heavy_tail_unitary.py`](../src/models/heavy_tail_unitary.py) · 13 test ·
[`heavy_tail_unitary_*.md`](../reports/research/)

### 4.2 Hạ tầng thu dữ liệu cưỡng chế pre-registration bằng code

[`collect_confirmatory.py`](../scripts/operations/collect_confirmatory.py) không
tin vào kỷ luật của người chạy — nó **ép** protocol:

- `obi_source` hard-code `"lob"` ⟹ **không thể** âm thầm rơi về trade-flow proxy;
- phân đoạn theo **ngày UTC trọn vẹn**, chạy theo chunk nên khởi động lại là *resume*;
- ghi coverage/gap/reconnect thật, chỉ đánh dấu `complete` khi qua ngưỡng tường minh;
- manifest **bất biến** (SHA-256 + git commit) và **từ chối ghi đè** ngày đã xong.

11 test phủ toàn bộ luật trên qua một "fake world" ghép đồng hồ với collector —
kiểm chứng được **không cần mạng**.

### 4.3 Kỷ luật provenance fail-closed

Artifact của **pipeline** (Phase 2–6) ghi protocol version, git commit đầy đủ,
canonical feature path, SHA-256 từng input/output, dependency lock và seed;
trường nào không khớp thì pipeline **hard-fail** thay vì xuất kết quả. Chính kỷ
luật này giúp phát hiện rằng con số +0,049889 không tái lập được.

**Đính chính (rà soát 2026-07-22):** các script nghiên cứu Phase 1–6 mà chúng
tôi viết thêm ban đầu **không** ghi `feature_sha256` — chúng chỉ ghi git commit
và đường dẫn. 16 file JSON trong `reports/research/` sinh trước ngày này vì vậy
thiếu trường đó. Đã sửa cả năm script để ghi SHA-256 và canonical path; các
artifact cũ **không** được backfill hash về sau (làm vậy chính là thứ kỷ luật
này tồn tại để ngăn) — chúng giữ nguyên và được đánh dấu là thiếu trường.

### 4.4 Bộ baseline cổ điển mạnh, dùng lại được

OrderFlow AR(5), Logistic L2 + Pairwise, Marked Hawkes conditional-mark logit,
Nonlinear Calibrated — tất cả trên cùng feature causal. Đây là thứ khiến kết luận
tiêu cực **đáng tin**: chúng tôi không hạ chuẩn đối thủ.

---

## 5. Hạn chế (nêu chủ động)

| # | Hạn chế | Trạng thái |
|---|---|---|
| 1 | OBI là **trade-flow proxy**, không phải L2 LOB thật | Hạ tầng thu đã sẵn sàng và có test; cần **≥20 ngày UTC thời gian thực** — không nén được |
| 2 | Toàn bộ Phase 1–6 là **exploratory**, chưa phải confirmatory | Protocol đã đóng băng dạng văn bản + code; chờ dữ liệu ở #1 |
| 3 | Dữ liệu trải trên khoảng thời gian ngắn | Đã chạy full 32,4M tick, nhưng bề rộng *thời gian* vẫn hạn chế |
| 4 | Windowing CRPS trong-file mỏng hơn chuẩn day-cluster | Đã ghi rõ trong §5d báo cáo cuối |
| 5 | Model không mô hình hoá volatility | Giải thích trực tiếp thất bại ở window biến động cao (§3.5) |
| 6 | Phân tích giao dịch **chưa có adverse selection** | Khi spread thu được lớn hơn phí, công thức hoà vốn kết luận có lãi ở *mọi* độ chính xác — đó là ảo giác. Cần L2 thật để mô hình hoá hàng đợi lệnh (#1) |
| 7 | Khử chồng lấp làm cỡ mẫu tụt còn **86–9.695 cửa sổ** | Khoảng tin cậy rộng; §3.6 không phải kết luận dứt khoát |

Không hạn chế nào ở trên được phát hiện bởi người ngoài — tất cả do chính pipeline
kiểm toán của dự án nêu ra.

---

## 6. Tái lập

```powershell
python -m pytest tests/ -v          # 244 test

python -m scripts.research.alpha_phase_ablation            # §3.1, §3.2
python -m scripts.research.strong_baseline_comparison      # §3.3
python -m scripts.research.full_dataset_confirmation       # §3.4
python -m scripts.research.marginal_crps_comparison        # §3.5
python -m scripts.research.horizon_feasibility             # §3.6
python -m scripts.research.horizon_label_baselines         # §3.6
python -m scripts.research.heavy_tail_unitary_evaluation   # §4.1

python -m scripts.operations.collect_confirmatory --status # §4.2
```

Lịch sử commit của chuỗi nghiên cứu này:

| Commit | Nội dung |
|---|---|
| `08e0865` | Ablation pha — phát hiện fold-fragility |
| `0a502fd` | **Sửa root-cause** `calibrate_bias` |
| `4eb0558` | So sánh baseline mạnh |
| `0198737` | Chạy full dataset, đóng hạn chế #6 |
| `a2d16f1` | Endpoint CRPS |
| `33d8add` | Lévy unitary shift |
| `040f6bb` | Runner thu L2 LOB confirmatory |
| `3a53212` | **Sửa 3 lỗi đo lường** khiến chiến lược demo trông có lãi |
| `6d49274` | Khả thi giao dịch theo horizon |
| `412d6f5` | Nhãn theo horizon — kỹ năng có thật nhưng không sinh lời |

---

## 7. Điều chúng tôi muốn hội đồng đánh giá

Dự án này **không** trình bày một model lượng tử thắng thị trường. Nó trình bày
một quy trình đủ chặt để **phát hiện rằng model của chính mình không thắng**.

Cụ thể, dự án đã **ba lần tự bác bỏ chính mình**, và mỗi lần đều công bố thay vì
giấu đi:

1. Một bug `calibrate_bias` từng tạo ra con số đẹp hơn sự thật gần **4 lần**
   (+0,0499 so với −0,0131 thật).
2. Một artifact BNB không tái lập được; chạy lại đúng cách cho kết quả **xấu
   hơn** cho model (hạng 2/6 → 4/6).
3. Ba lỗi đo lường khiến chiến lược demo trông có lãi; sửa xong nó **lỗ 4,2%**.

Và dự án chấp nhận thua một hồi quy tuyến tính 5 hệ số, rồi còn đi thêm một bước
nữa để chứng minh rằng **ngay cả khi có dự báo tốt cũng chưa giao dịch được** —
vì ở horizon đang dùng, phí giao dịch lớn hơn biên độ giá tới hơn hai nghìn lần.

Trong một lĩnh vực mà kết quả dương tính không tái lập được là vấn đề hệ thống,
chúng tôi cho rằng khả năng tự bác bỏ mới là đóng góp có giá trị.
