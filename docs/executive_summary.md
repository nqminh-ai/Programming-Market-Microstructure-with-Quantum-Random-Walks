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
4. **Ngoại lệ duy nhất:** ở endpoint chính đăng-ký-trước (marginal CRPS), QRW
   *cạnh tranh* với GARCH/GBM — nhưng vẫn không nhất quán, và **vẫn không đến từ
   pha**.

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

### 3.5 Ngoại lệ trung thực: endpoint CRPS

Ở endpoint **chính đăng-ký-trước** (mean fixed-origin marginal CRPS), bức tranh
khác — và chúng tôi báo cáo nó dù nó không ủng hộ kết luận chung:

| Tài sản | Hạng QRW | Model tốt nhất | Window QRW thắng |
|---|:--:|---|:--:|
| ETHUSDT | **1/6** | QRW Adaptive | 3/5 |
| BNBUSDT | 2/6 | GARCH(1,1) | 3/5 |
| BTCUSDT | 3/6 | GBM | 0/5 |

Đây là chiều **duy nhất** QRW không thua dứt khoát. Nhưng: nó thắng ở window
biến động **thấp** và thua đậm ở window biến động **cao** — vì model không mô
hình hoá volatility. Và §3.1 đã cho thấy phần thắng này **cũng không đến từ pha**.

📄 [`marginal_crps_*.md`](../reports/research/)

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

Mọi artifact ghi protocol version, git commit đầy đủ, SHA-256 từng input/output,
dependency lock và seed. Trường nào không khớp thì pipeline **hard-fail** thay vì
xuất kết quả. Chính kỷ luật này giúp phát hiện rằng con số +0,049889 không tái
lập được.

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

Không hạn chế nào ở trên được phát hiện bởi người ngoài — tất cả do chính pipeline
kiểm toán của dự án nêu ra.

---

## 6. Tái lập

```powershell
python -m pytest tests/ -v          # 212 test

python -m scripts.research.alpha_phase_ablation            # §3.1, §3.2
python -m scripts.research.strong_baseline_comparison      # §3.3
python -m scripts.research.full_dataset_confirmation       # §3.4
python -m scripts.research.marginal_crps_comparison        # §3.5
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

---

## 7. Điều chúng tôi muốn hội đồng đánh giá

Dự án này **không** trình bày một model lượng tử thắng thị trường. Nó trình bày
một quy trình đủ chặt để **phát hiện rằng model của chính mình không thắng** —
gồm việc tự tìm ra và công bố một bug đã tạo ra con số đẹp hơn sự thật gần **4
lần**, và việc chấp nhận thua một hồi quy tuyến tính 5 hệ số.

Trong một lĩnh vực mà kết quả dương tính không tái lập được là vấn đề hệ thống,
chúng tôi cho rằng đó mới là đóng góp có giá trị.
