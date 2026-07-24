# BÁO CÁO CHUYÊN SÂU: NGHIÊN CỨU KHOA HỌC & HẠ TẦNG KỸ THUẬT
## Dự án AI Quantum — QRW cho Vi cấu trúc Thị trường

---

## TỔNG QUAN HÀNH CHÍNH

Dự án **AI Quantum — QRW cho Vi cấu trúc Thị trường** được xây dựng nhằm trả lời một câu hỏi cốt lõi: *"Cơ chế lượng tử trong Quantum Random Walk (QRW) có thực sự mang lại lợi thế dự báo (Quantum Advantage) trên dữ liệu tài chính tần suất cao (HFT) hay không?"*

Mặc dù kết quả về mặt giao dịch thực chiến là **kết quả âm (negative result)** — cơ chế pha lượng tử đóng góp bằng 0 và không sinh ra lợi nhuận sau chi phí — giá trị lớn nhất của dự án lại nằm ở **Nghiên cứu Khoa học** và **Hạ tầng Kỹ thuật**. 

Dự án đã giải quyết một bài toán kinh điển trong AI/QML tài chính: **Làm sao để không tự lừa dối chính mình (Anti-Self-Deception Engineering).**

---

## PHẦN I: NHỮNG VẤN ĐỀ NHỨC NHỐI MÀ HẠ TẦNG NÀY KHẮC PHỤC ĐƯỢC

Trong lĩnh vực Tài chính Định lượng (Quantitative Finance) và Machine Learning Lượng tử (QML), hàng loạt bài báo công bố "lợi thế vượt trội" nhưng khi đưa vào thực tế đều thất bại. Hạ tầng của dự án được thiết kế để triệt tiêu 6 cạm bẫy hệ thống sau:

| STT | Cạm bẫy phổ biến trong nghiên cứu ML/QML | Hạ tầng dự án khắc phục như thế nào? |
|---|---|---|
| **1** | **So sánh với Baseline quá yếu** (VD: Random walk đơn giản) làm mô hình mới luôn nhìn như "thắng". | Xây dựng bộ **7 Baseline cổ điển mạnh** (OrderFlow AR(5), Logistic+Pairwise, Marked Hawkes) được đăng ký trước. |
| **2** | **Không cô lập được cơ chế lượng tử**, không biết phần thắng đến từ đâu. | Thiết kế công tắc `alpha_phase = 0` loại bỏ **chính xác** số hạng giao thoa pha về mặt toán học để đo đóng góp riêng của pha. |
| **3** | **Lỗi Fit/Predict lệch công thức (Bias Calibration Bug).** | Xây dựng hệ thống audit mã nguồn tự động, phát hiện và sửa bug `calibrate_bias` (bug này từng làm kết quả ảo đẹp $+0,0499$, sửa xong về số thật $-0,0131$). |
| **4** | **Tráo đổi dữ liệu thử nghiệm (Data Leakage & Proxy Swapping).** | Cưỡng chế quy trình Pre-registration bằng code: khóa nguồn dữ liệu `obi_source = "lob"`, ngăn chặn việc dùng proxy thay thế. |
| **5** | **Bỏ qua Phí giao dịch và Adverse Selection.** | Dựng mô hình chi phí thực tế: dùng ước lượng **Roll (1984)** đo spread thật và tính toán **Realised Half-Spread** thực tế của lệnh chờ (Maker). |
| **6** | **Số liệu báo cáo không tái lập được (Unreproducible Claims).** | Thiết kế hạ tầng Provenance bất biến: lưu SHA-256 mã hash dữ liệu, Git commit, phiên bản Python, khóa exact dependency pins. |

---

## PHẦN II: HẠ TẦNG KỸ THUẬT CHỐNG TỰ LỪA DỐI (ANTI-SELF-DECEPTION INFRASTRUCTURE)

Hạ tầng kỹ thuật của dự án bao gồm 3 trụ cột chính:

### 1. Hạ tầng Thu dữ liệu L2 LOB Pre-registration (`collect_confirmatory.py`)

* **Nhiệm vụ:** Thu thập dữ liệu Sổ lệnh cấp 2 (Level-2 Limit Order Book) và dữ liệu khớp lệnh (Trades) theo thời gian thực từ sàn Binance.
* **Vấn đề khắc phục:** Tránh việc nghiên cứu viên âm thầm đổi sang dữ liệu giả lập hoặc proxy giao dịch dễ làm đẹp kết quả, đồng thời tránh rò rỉ dữ liệu (data leakage).
* **Cơ chế hoạt động:**
  * **Hardcode tham số bất biến:** Mã nguồn quy định cứng `obi_source = "lob"`, không cho phép fallback về proxy trade-flow.
  * **Phân đoạn theo ngày UTC trọn vẹn:** Thu thập dữ liệu theo các khung ngày UTC cố định ($24\text{ giờ}$/file), chia nhỏ dạng chunk ($900\text{ giây}$) giúp tự khôi phục (resume) khi mất kết nối mạng mà không làm hỏng dữ liệu.
  * **Kiểm định chất lượng nghiêm ngặt (Quality Thresholds):** Mỗi ngày dữ liệu chỉ được công nhận khi đạt: Độ phủ sóng (Coverage) $\ge 95\%$, khoảng trống dữ liệu tối đa $< 300\text{ giây}$, tối thiểu $1.000$ trades và $1.000$ LOB snapshots.
  * **Manifest bất biến (Immutable Manifest):** Mỗi ngày dữ liệu hoàn thành sẽ được ghi kèm một file `manifest.json` chứa mã hash SHA-256 của từng file raw, Git commit ID, phiên bản Python và cài đặt phần cứng. Khi một ngày đã gắn nhãn `complete`, hệ thống **từ chối ghi đè**.
* **Công cụ sử dụng:** Python 3.14, WebSocket/REST API client (`LiveMarketCollector`), `hashlib` (SHA-256), `json`, `dataclasses`.

---

### 2. Hạ tầng Tái lập & Truy vết Provenance (`reproduce.py` & Fail-Closed Pipeline)

* **Nhiệm vụ:** Đảm bảo mọi con số xuất hiện trong báo cáo đều có thể truy ngược chính xác về lệnh chạy, mã nguồn và dữ liệu gốc.
* **Vấn đề khắc phục:** Hiện tượng "kết quả chỉ đúng trên máy của tác giả" hoặc số liệu bị biến đổi qua các phiên bản sửa code.
* **Cơ chế hoạt động:**
  * **Fail-Closed Pipeline:** Từ Phase 2 đến Phase 6, nếu mã Git commit bị bẩn (uncommitted changes), hoặc mã SHA-256 của dữ liệu đầu vào không khớp với manifest, pipeline sẽ **lập tức dừng lại (Hard-Fail)** và từ chối xuất kết quả.
  * **Chế độ kiểm tra tự động (`reproduce.py`):** Quét toàn bộ 21 artifact JSON quan trọng trong `reports/research/`, kiểm tra xem chúng có chứa đủ thông tin provenance (Git commit, Python version, Dataset SHA-256) và được gắn nhãn `EXPLORATORY_ONLY_NOT_CONFIRMATORY` hay không.
* **Công cụ sử dụng:** Python `argparse`, `pathlib`, `hashlib`, Makefile, `requirements.lock` (khóa cứng phiên bản mọi thư viện).

---

### 3. Bộ Kiểm toán Tự động & Test Giằng số liệu (`test_report_numbers.py`)

* **Nhiệm vụ:** Đảm bảo văn bản báo cáo (Markdown) trích dẫn **chính xác $100\%$** các con số từ file kết quả JSON.
* **Vấn đề khắc phục:** Người viết báo cáo gõ nhầm số, làm tròn sai, hoặc cố tình làm đẹp con số so với thực tế tính toán.
* **Cơ chế hoạt động:** Pytest tự động đọc từng file báo cáo văn bản (`executive_summary.md`, `final_report.md`), trích xuất các con số được trích dẫn, và so sánh từng chữ số thập phân với dữ liệu trong file JSON do pipeline sinh ra. Nếu lệch $0.00001$, test sẽ thất bại.
* **Công cụ sử dụng:** Pytest, Regular Expressions (Regex), JSON parser.

---

## PHẦN III: ĐÓNG GÓP TOÁN HỌC & MÔ HÌNH — LÉVY UNITARY SHIFT (`heavy_tail_unitary.py`)

### 1. Vấn đề của Quantum Random Walk (QRW) thông thường

Trong cơ học lượng tử chuẩn, toán tử dịch chuyển (Shift Operator) của bước đi lượng tử rời rạc (DTQRW) dịch chuyển vị trí sang trái/phải 1 đơn vị:
$$|x\rangle \otimes |0\rangle \to |x-1\rangle \otimes |0\rangle, \quad |x\rangle \otimes |1\rangle \to |x+1\rangle \otimes |1\rangle$$

Phân phối xác suất của QRW thông thường có tính chất **Bimodal/Ballistic** (tập trung hai bên biên) và **Compact Support** (không có đuôi dài). Trong khi đó, dữ liệu giá tài chính thực tế lại có phân phối **Đuôi nặng (Fat Tails/Heavy Tails)** và bước nhảy Lévy. 

Các nghiên cứu trước đây cố gắng tạo bước nhảy đuôi nặng bằng cách lấy mẫu ngẫu nhiên cổ điển (Bernoulli/Pareto sampling) bên ngoài vòng lặp lượng tử. Điều này **phá hủy tính thuần lượng tử** và làm mất tính bảo toàn xác suất (Unitary).

---

### 2. Giải pháp: Toán tử Lévy Unitary Shift trong Không gian Động lượng

Dự án đã phát minh và cài đặt thành công toán tử **Lévy Unitary Shift** thuần túy trong không gian động lượng (Momentum Space).

#### Công thức Toán học:
Trên một mạng lưới vòng (periodic cycle) gồm $N$ vị trí, toán tử shift chuẩn có dạng nhân pha trong không gian động lượng $k$: $e^{-i k}$ (cho coin-up) và $e^{+i k}$ (cho coin-down), với $k \in [-\pi, \pi)$.

Dự án tổng quát hóa góc pha thành hàm:
$$\phi_\alpha(k) = \text{sign}(k) \cdot |k|^\alpha, \quad \text{với } \alpha \in (0, 2]$$

#### Tính chất Vật lý & Toán học nổi bật:
1. **Bảo toàn tính Unitary cấu trúc (Exact Structural Unitarity):** Vì $\phi_\alpha(k)$ là một số thực với mọi $k$, trị riêng $e^{-i \phi_\alpha(k)}$ luôn nằm trên đường tròn đơn vị ($|e^{-i \phi_\alpha(k)}| = 1$). Do đó, toán tử này **Unitary chính xác tuyệt đối** tới độ chính xác máy tính ($10^{-16}$), không phải xấp xỉ.
2. **Tổng quát hóa chặt chẽ (Strict Generalization):** 
   * Khi $\alpha = 1 \implies \phi_1(k) = k$, tái tạo **chính xác** bước nhảy $\pm 1$ của Quantum Walk chuẩn.
3. **Tạo ra bước nhảy Lévy thực sự (Genuine Lévy Flights):**
   * Khi $\alpha < 1$, hàm $\phi_\alpha(k)$ có điểm gián đoạn đạo hàm (cusp) $|k|^\alpha$ tại $k = 0$.
   * Biến đổi Fourier ngược của một hàm không trơn tạo ra các hệ số phân rã theo hàm mũ lũy thừa:
     $$c(x) \sim |x|^{-(1+\alpha)}$$
   * Điều này sinh ra các bước nhảy xa (Lévy flights) trực tiếp từ tiến hóa lượng tử thuần túy, không hề dùng sampling cổ điển.

---

### 3. Tối ưu hóa Thuật toán & Độ phức tạp

* **Thực thi qua Biến đổi Fourier Nhanh (FFT):**
  Thay vì nhân ma trận mật kích thước $N \times N$ với độ phức tạp $O(N^2)$, lớp `LevyUnitaryQRW` áp dụng toán tử dịch chuyển thông qua FFT:
  1. Biến đổi trạng thái sang không gian động lượng: $\tilde{\Psi}(k) = \text{FFT}(\Psi(x))$ — $O(N \log N)$
  2. Nhân trực tiếp với mảng pha $\exp(-i \phi_\alpha(k))$ — $O(N)$
  3. Biến đổi ngược về không gian vị trí: $\Psi(x) = \text{IFFT}(\tilde{\Psi}(k))$ — $O(N \log N)$
  
  $\implies$ **Tổng độ phức tạp tính toán: $O(N \log N)$**, cho phép chạy với mạng lưới $N = 16.001$ vị trí trong vài mili-giây.

* **Kiểm soát Khối lượng Tràn Biên (`wraparound_mass`):**
  Do mạng lưới có tính tuần hoàn, các bước nhảy Lévy quá xa có thể vượt qua biên và tràn về phía đối diện. Lớp `LevyUnitaryQRW` cung cấp hàm `wraparound_mass()` đo tỷ lệ xác suất ở $10\%$ vùng biên để cảnh báo nghiên cứu viên mở rộng kích thước $N$.

---

## PHẦN IV: BỘ BASELINE CỔ ĐIỂN & MÔ HÌNH CHI PHÍ VI CẤU TRÚC

### 1. Bộ Baseline Cổ điển Đăng ký Trước

Để đánh giá trung thực QRW, hạ tầng cung cấp bộ 7 mô hình đối chứng cổ điển mạnh:
1. **OrderFlow AR(5):** Mô hình Hồi quy Tuyến tính/Autoregressive 5 bậc trên luồng lệnh.
2. **Logistic L2 + Pairwise:** Hồi quy Logistic có chuẩn hóa L2 và tương quan cặp đặc trưng.
3. **Marked Hawkes Process:** Mô hình điểm tự kích hoạt mô phỏng khoảng thời gian giữa các giao dịch.
4. **GBM (Gradient Boosting Machine):** Mô hình cây quyết định phân cấp cho dữ liệu phi tuyến.
5. **CRW (Correlated Random Walk):** Bước đi ngẫu nhiên cổ điển có tương quan thời gian.

### 2. Mô hình Chi phí Vi cấu trúc Thực tế (Microstructure Cost Model)

Hạ tầng phân tích giao dịch khắc phục các giả định sai lầm về chi phí HFT:
* **Khắc phục lỗi đo Spread:** Bản cũ dùng $VWAP_{100}$ trượt làm mid-price dẫn đến đo spread sai gấp $22\text{--}33$ lần. Hạ tầng mới áp dụng ước lượng **Roll (1984)** dựa trên tự tương quan chuỗi giá:
  $$\text{Spread}_{\text{Roll}} = 2 \sqrt{-\text{Cov}(\Delta P_t, \Delta P_{t-1})}$$
* **Đo lường Adverse Selection thực tế:** Hạ tầng đo lường **Realised Half-Spread** của lệnh chờ (Passive Limit Order). Kết quả cho thấy lệnh chờ của Maker bị lỗ ròng $\sim -1,2\text{ bps}$ do hiện tượng "bị nhặt" (adverse selection) bởi các lệnh Market thông minh.

---

## PHẦN V: HƯỚNG DẪN VẬN HÀNH & SỬ DỤNG HẠ TẦNG

### 1. Cài đặt Môi trường
Yêu cầu Python 3.14. Sử dụng virtual environment và cài đặt gói thư viện đã khóa cứng phiên bản:

```powershell
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
.venv\Scripts\Activate.ps1

# Cài đặt chính xác các thư viện từ requirements.lock
python -m pip install --requirement requirements.lock
```

### 2. Chạy Kiểm thử Tự động (Automated Test Suite)
Chạy toàn bộ 244 test case kiểm tra tính đúng đắn của toán học và hạ tầng:

```powershell
python -m pytest tests/ -v
```

### 3. Xác minh Tái lập Số liệu Báo cáo (Provenance Verification)
Chạy lệnh kiểm tra tính toàn vẹn của tất cả các file kết quả nghiên cứu JSON:

```powershell
# Kiểm tra nhanh tính hợp lệ và provenance của 21 artifact chính
python -m scripts.operations.reproduce

# In ra toàn bộ câu lệnh để tái tạo lại các file kết quả nghiên cứu
python -m scripts.operations.reproduce --commands
```

### 4. Vận hành Thu thập Dữ liệu L2 LOB Confirmatory

```powershell
# Kiểm tra tiến độ thu thập dữ liệu confirmatory của BTC, ETH, BNB
python -m scripts.operations.collect_confirmatory --status

# Thu thập dữ liệu L2 LOB ngày UTC hiện tại cho BTCUSDT
python -m scripts.operations.collect_confirmatory --symbols BTCUSDT
```

---

## TỔNG KẾT BẢN BÁO CÁO

| Hạng mục | Giá trị đóng góp cốt lõi |
|---|---|
| **Về Toán học / QML** | Phát minh thành công **Lévy Unitary Shift** ($\phi_\alpha(k) = \text{sign}(k)|k|^\alpha$) thực thi bằng FFT $O(N \log N)$, bảo toàn tính Unitary tuyệt đối mà vẫn tạo ra bước nhảy đuôi nặng. |
| **Về Hạ tầng Kỹ thuật** | Dựng thành công **Hệ thống Kiểm toán & Thu thập Dữ liệu Pre-registration** bất biến (SHA-256 manifest, Git commit locking, fail-closed pipeline). |
| **Về Phương pháp luận** | Xây dựng chuẩn mực nghiên cứu **Tài chính Định lượng Phản biện (Falsifiable Quant Research)**, tự bác bỏ 6 lỗi hệ thống và cung cấp công cụ giúp các nhà nghiên cứu khác không rơi vào cạm bẫy "lợi thế lượng tử ảo". |
