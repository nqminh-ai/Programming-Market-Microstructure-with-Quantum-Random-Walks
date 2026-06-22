# 📖 Hướng Dẫn Nhanh: Hiểu Về Dự Án "Lập trình Vi cấu trúc Thị trường bằng Quantum Random Walks"

Chào bạn! Nếu bạn là một người mới bắt đầu làm quen với dự án này, tài liệu này được thiết kế dành riêng cho bạn. Hãy cùng bóc tách những khái niệm phức tạp thành những điều dễ hiểu nhất nhé!

---

## 1. Mục Tiêu Của Dự Án Này Là Gì?

Hãy tưởng tượng bạn đang nhìn vào bảng điện tử của thị trường chứng khoán. Bạn sẽ thấy giá thay đổi liên tục từng giây, từng mili-giây.
- **Vi cấu trúc thị trường (Market Microstructure):** Là việc nghiên cứu sự thay đổi giá cả ở mức độ cực kỳ nhỏ này. Nó quan tâm đến từng lệnh mua, lệnh bán và cách chúng tương tác với nhau để tạo ra giá.
- **Quantum Random Walks (Bước ngẫu nhiên lượng tử - QRW):** Nếu bạn từng nghe tới việc "tung đồng xu" để đoán giá sẽ lên hay xuống (đó là cách truyền thống - Classical Random Walk), thì "Lượng tử" ở đây giống như việc tung một đồng xu đặc biệt. Nó không chỉ là "sấp" hay "ngửa", mà nó có thể xem xét nhiều trạng thái cùng lúc, giúp mô hình hóa và dự đoán các mẫu hình phức tạp của giá cả tốt hơn.

**💡 Tóm lại:** Dự án này sử dụng một công cụ toán học siêu việt (QRW) để dự đoán cách giá chứng khoán/tiền điện tử dịch chuyển trong từng giây, và so sánh xem nó có thực sự giỏi hơn cách dự đoán truyền thống hay không.

---

## 2. Giải Mã Các Thư Mục & Tệp Tin (File) Trong Dự Án

Dự án này được sắp xếp rất khoa học. Dưới đây là ý nghĩa của từng "ngôi nhà" trong mã nguồn:

### 📁 Các Thư Mục Chính (Folder)

*   **`config/` (Cấu hình):** Chứa các cài đặt cho dự án. Giống như việc bạn setup một trò chơi trước khi chơi vậy (VD: Chạy trên dữ liệu nào, thông số cơ bản ra sao).
*   **`data/` (Dữ liệu):** Nơi chứa nguyên liệu thô để nấu ăn.
    *   `raw/`: Dữ liệu gốc lấy từ thị trường về (chưa qua xử lý).
    *   `processed/`: Dữ liệu đã được làm sạch, nhặt bỏ "sạn".
    *   `features/`: Dữ liệu đã được "tẩm ướp", rút trích ra các đặc trưng quan trọng nhất để đưa cho AI học.
*   **`docs/` (Tài liệu):** Nơi chứa toàn bộ giấy tờ, kế hoạch, lý thuyết toán học, các báo cáo, và các slide thuyết trình của nhóm.
*   **`figures/` (Biểu đồ):** Nơi lưu lại các hình ảnh, biểu đồ đẹp mắt mà hệ thống tự động vẽ ra sau khi chạy xong.
*   **`notebooks/` (Sổ tay thử nghiệm):** Chứa các file nháp (Jupyter Notebook). Các lập trình viên dùng nó để viết code thử nghiệm, xem kết quả ngay lập tức trước khi đưa vào hệ thống chính.
*   **`reports/` (Báo cáo tự động):** Chứa kết quả đánh giá chi tiết (như dữ liệu có sạch không, mô hình có bị học vẹt/overfitting không, tiến độ các giai đoạn).
*   **`results/` (Kết quả):** Nơi xuất ra bảng điểm cuối cùng, so sánh xem mô hình Quantum hay mô hình Thường chiến thắng.
*   **`scripts/` (Kịch bản tự động):** Chứa các tệp lệnh. Chỉ cần gõ một dòng lệnh, máy tính sẽ tự động chạy theo thứ tự từ làm sạch dữ liệu -> tính toán -> vẽ biểu đồ -> in báo cáo.
*   **`tests/` (Kiểm thử):** Chứa các bài "kiểm tra bài cũ" cho máy tính. Mỗi khi ai đó viết thêm code mới, thư mục này sẽ tự động kiểm tra xem code mới có làm hỏng code cũ không (hiện có 61 bài test tự động).

### 📁 `src/` (Mã nguồn lõi)
Đây là trái tim của dự án, nơi chứa các đoạn code chính:
*   `models/`: Chứa bộ não toán học (Mô hình Quantum, Mô hình cổ điển, v.v.).
*   `data/`: Code dùng để tải, dọn dẹp và nhào nặn dữ liệu.
*   `evaluation/`: Bộ phận "chấm điểm", xem mô hình nào dự đoán chuẩn xác hơn.
*   `visualization/`: Họa sĩ của dự án, chuyên dùng số liệu để vẽ nên các biểu đồ.
*   `reporting/`: Máy in báo cáo tự động.
*   **`dashboard/`: Giao diện ứng dụng (Sẽ nói rõ ở phần sau).**

### 📄 Các File Đáng Chú Ý
*   **`README.md`**: Tấm bản đồ của dự án, tóm tắt tình trạng hiện tại và cách cài đặt.
*   **`requirements.txt` / `pyproject.toml`**: Danh sách các "đồ nghề" (thư viện lập trình) cần tải về để chạy được dự án này.
*   **`start.bat`**: Nút bấm "khởi động nhanh" dành cho người dùng hệ điều hành Windows.

---

## 3. Dashboard Của Dự Án Là Gì Và Tại Sao Nó Quan Trọng?

**Nằm tại:** `src/dashboard/app.py`

### 🖥️ Dashboard là gì?
Nếu những thư mục trên là "bếp" và "kho nguyên liệu", thì Dashboard chính là **"Phòng ăn sang trọng"**.
Thay vì phải ngồi đọc những dòng chữ code nhàm chán hay nhìn vào các bảng số liệu Excel khô khan, Dashboard biến mọi thứ thành một **Trang Web Trực Quan** (sử dụng công nghệ Streamlit).

### 🎯 Ý nghĩa của Dashboard:
1.  **Dễ sử dụng (User-Friendly):** Bất kỳ ai, dù không biết viết code, đều có thể mở Dashboard lên. Nó có các nút bấm, thanh trượt để bạn dễ dàng tương tác.
2.  **Trực quan hóa thời gian thực:** Bạn có thể chọn một ngày cụ thể trên lịch, chọn loại mô hình (Quantum hay Cổ điển), và bùm! Biểu đồ giá cả và biểu đồ dự đoán sẽ hiện ra ngay lập tức.
3.  **Công cụ đánh giá nhanh:** Nhìn vào Dashboard, bạn có thể lập tức thấy được mô hình Quantum đang dự đoán tốt hay kém hơn so với thực tế, các chỉ số độ chính xác được hiển thị to và rõ ràng.

### Lời kết
Dự án này là một cỗ máy phân tích toán học cực kỳ phức tạp. Tuy nhiên, nhờ có cách sắp xếp thư mục rõ ràng khoa học và một giao diện **Dashboard** thân thiện, bất kỳ ai (dù không rành về code hay toán học lượng tử) cũng có thể tiếp cận, xem kết quả và đưa ra các đánh giá về tài chính!
