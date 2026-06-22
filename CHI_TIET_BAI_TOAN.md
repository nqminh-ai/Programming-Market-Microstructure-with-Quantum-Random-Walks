# 🔬 Phân Tích Chuyên Sâu: Bài Toán, Dữ Liệu và Phương Pháp Nghiên Cứu

Tài liệu này cung cấp cái nhìn chi tiết, mang tính học thuật nhưng được diễn giải rõ ràng về cốt lõi của dự án **"Lập trình Vi cấu trúc Thị trường bằng Quantum Random Walks"**. Đây là tài liệu lý tưởng để đưa vào Báo cáo đề tài, Khóa luận hoặc Proposal.

---

## 1. Bài Toán Nghiên Cứu (Problem Statement)

### 1.1 Bối cảnh & Tầm quan trọng của Vi cấu trúc thị trường (Market Microstructure)
Khi phân tích thị trường tài chính, đa số mọi người tập trung vào kinh tế vĩ mô hoặc phân tích kỹ thuật (biểu đồ ngày/giờ). Tuy nhiên, mọi sự dịch chuyển giá, mọi cú sập (crash) hay bùng nổ (breakout) đều bắt nguồn từ một thế giới vi mô: **Sự khớp lệnh giữa hàng nghìn lệnh Mua và Bán trong từng mili-giây**. 

Vi cấu trúc thị trường chính là ngành khoa học nghiên cứu cơ chế "khớp bánh răng" này. Nắm bắt được vi cấu trúc là chìa khóa để xây dựng các thuật toán giao dịch tần suất cao (HFT) và quản trị rủi ro thanh khoản.

### 1.2 Những điểm nghẽn của các mô hình truyền thống
Nhiều thập kỷ qua, tài chính định lượng dựa vào **Bước ngẫu nhiên cổ điển (Classical Random Walk - CRW)** và các biến thể của nó (như GARCH, ARIMA). Mô hình này giả định:
- Giá ngày mai tăng hay giảm là ngẫu nhiên, giống như tung một đồng xu.
- Các bước nhảy của giá là độc lập với nhau.

**Vấn đề:** Thị trường thực tế không như vậy. Tâm lý đám đông khiến thị trường có quán tính. Việc sử dụng xác suất cổ điển (Cộng các xác suất lại luôn bằng 1) không thể mô hình hóa được những pha biến động cực đoan, hay hiện tượng "hút giá" khi sổ lệnh bị mất cân bằng trầm trọng.

### 1.3 Lời giải từ Lượng tử (Quantum Random Walks - QRW)
Vật lý lượng tử cung cấp một công cụ toán học vượt trội: **Bước ngẫu nhiên lượng tử (QRW)**. 
- Thay vì cộng xác suất, QRW cộng các **Biên độ xác suất (Probability Amplitudes)** – vốn là các số phức.
- Điều này tạo ra hiện tượng **Giao thoa (Interference)**. Các sóng xác suất có thể triệt tiêu lẫn nhau (khiến giá đứng im) hoặc cộng hưởng với nhau (tạo ra xác suất cực lớn ở một hướng, dự báo sự bùng nổ giá).
- **Tuyên bố bài toán:** Dự án này nhằm mục đích mã hóa toán học QRW thành các thuật toán máy tính, áp dụng vào dữ liệu sổ lệnh để dự báo hướng đi của giá ở cấp độ vi mô, và chứng minh xem nó có vượt qua được giới hạn của các mô hình cổ điển hay không.

---

## 2. Dữ Liệu Nghiên Cứu (The Data)

Để giải quyết bài toán vi cấu trúc, chúng ta không thể dùng giá đóng cửa (Close price) cuối ngày. Chúng ta cần dữ liệu ở độ phân giải cao nhất có thể: **High-Frequency Data**.

### 2.1 Cấu trúc Dữ liệu
Dự án sử dụng kết hợp hai luồng dữ liệu chính:
1. **Dữ liệu Sổ lệnh (Limit Order Book - LOB):** 
   - Là một "bức ảnh chụp" toàn bộ các lệnh đang chờ trên thị trường tại một thời điểm.
   - Bao gồm: `Bid Price` (Giá chào mua tốt nhất), `Ask Price` (Giá chào bán tốt nhất), và `Volume` (Khối lượng đang chờ ở từng mức giá).
2. **Dữ liệu Tick (Tick-by-tick Data):**
   - Là bản ghi mọi giao dịch đã khớp thành công.
   - Bao gồm: `Timestamp` (chính xác đến mili-giây), `Trade Price` (Giá khớp), `Trade Size` (Khối lượng khớp), `Direction` (Lệnh chủ động Mua hay Bán).

### 2.2 Đặc trưng quan trọng nhất (Feature Engineering)
Dữ liệu thô rất nhiễu. Dự án phải trích xuất ra một chỉ báo cốt lõi gọi là **Sự mất cân bằng sổ lệnh (Order Book Imbalance - OBI)**.
- *Công thức:* `OBI = (Volume_Bid - Volume_Ask) / (Volume_Bid + Volume_Ask)`
- *Ý nghĩa:* Nếu OBI tiến gần về 1, áp lực mua đang áp đảo hoàn toàn. Nếu OBI tiến về -1, phe bán đang hoảng loạn tháo chạy. OBI chính là "lực đẩy" để nạp vào mô hình Lượng tử.

### 2.3 Thách thức của dữ liệu
- Kích thước khổng lồ (hàng triệu dòng cho chỉ vài ngày giao dịch).
- Chứa nhiều dữ liệu lỗi (outliers) do lỗi đường truyền mạng của sàn giao dịch. Cần có quy trình làm sạch (Data Cleaning) cực kỳ chuẩn xác trước khi chạy mô hình.

---

## 3. Phương Pháp Dự Kiến (Proposed Methodology)

Dự án được thiết kế theo một **Pipeline (Đường ống) khoa học dữ liệu** hoàn chỉnh và khép kín, bao gồm 4 bước:

### Bước 1: Kỹ thuật Dữ liệu (Data Engineering Pipeline)
- Tải dữ liệu thô và làm sạch các điểm dị thường (loại bỏ spread âm, giá trị rỗng).
- Tính toán `Micro-returns` (Tỷ suất sinh lợi vi mô) và `OBI`.
- Đồng bộ hóa thời gian (Resampling) dữ liệu tick rời rạc thành các lưới thời gian đều nhau để máy tính có thể xử lý.

### Bước 2: Thiết kế Hệ thống Động lực học Lượng tử (Quantum Modeling)
Xây dựng 3 lớp mô hình để đối chiếu:
1. **Classical Random Walk (CRW):** Làm baseline (thước đo cơ sở). Chạy mô phỏng Monte Carlo bằng xác suất cổ điển.
2. **Standard QRW (Lượng tử tiêu chuẩn):** 
   - Khởi tạo *Trạng thái đồng xu (Coin state)* và *Trạng thái vị trí (Position state)*.
   - Dùng toán tử tiến hóa Unitary Operator. Tín hiệu OBI từ sổ lệnh sẽ được dùng để bẻ cong "Góc lượng tử" (Theta), khiến xác suất di chuyển của giá ngả về bên Mua hoặc Bán.
3. **Adaptive QRW (Lượng tử thích ứng):** Nâng cấp của Standard QRW, mô hình tự động "cảm nhận" độ biến động của thị trường (Volatility) để điều chỉnh tốc độ thay đổi góc lượng tử.

### Bước 3: Huấn luyện & Chống rò rỉ dữ liệu (Anti-Leakage Architecture)
Trong tài chính, việc mô hình "nhìn lén" tương lai để dự đoán hiện tại (Look-ahead bias) là lỗi chí mạng.
- **Walk-forward Validation:** Đào tạo mô hình theo phương pháp cuốn chiếu. (Ví dụ: Học dữ liệu ngày 1, dự đoán ngày 2; sau đó học ngày 1+2, dự đoán ngày 3). Đảm bảo mô hình luôn đối mặt với tương lai mù mịt.
- Áp dụng các bài Audit tự động để block (chặn) mọi dòng code có dấu hiệu sử dụng dữ liệu tương lai.

### Bước 4: Đánh giá & Giám sát (Evaluation & Dashboarding)
- **Tiêu chí đánh giá:** Sử dụng RMSE (Sai số toàn phương trung bình), MAE (Sai số tuyệt đối trung bình) và Directional Accuracy (Tỷ lệ đoán đúng hướng).
- **Trực quan hóa:** Xây dựng một Dashboard bằng Streamlit để vẽ biểu đồ phân bố xác suất lượng tử (Quantum Probability Distribution) theo thời gian thực, cho phép so sánh trực quan hiệu năng của QRW và các mô hình truyền thống bằng mắt thường.
