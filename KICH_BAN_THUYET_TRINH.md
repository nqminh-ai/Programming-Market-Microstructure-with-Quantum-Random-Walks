# 🎤 Kịch Bản Thuyết Trình Chi Tiết: "Lập trình Vi cấu trúc Thị trường bằng Quantum Random Walks"

*(Ghi chú cho Presenter: Kịch bản này được viết theo phong cách của một giảng viên đang thuyết trình khoa học: dẫn dắt từ từ, giải thích cặn kẽ từng khái niệm khó bằng ví dụ thực tế (case study), để đảm bảo dù Hội đồng có người không chuyên về lượng tử hay tài chính định lượng cũng có thể hiểu và bị thuyết phục hoàn toàn.)*

---

## [Slide 1: Tiêu đề & Lời mở đầu]

**🗣️ Lời nói:**
"Kính chào quý vị trong Hội đồng phản biện, các thầy cô và các bạn sinh viên. 

Hôm nay, nhóm chúng em xin trình bày một đề tài mang tính giao thoa giữa Vật lý lượng tử, Khoa học Máy tính và Tài chính: **'Nghiên cứu và Lập trình Vi cấu trúc Thị trường bằng mô hình Quantum Random Walks (Bước ngẫu nhiên lượng tử).'** 

Nói một cách đơn giản, chúng em đang dùng toán học của thế giới vi mô (lượng tử) để giải mã sự hỗn loạn của thị trường chứng khoán trong từng mili-giây."

---

## [Slide 2: Đặt vấn đề - Vi cấu trúc thị trường (Market Microstructure) là gì?]

**🗣️ Lời nói:**
"Trước tiên, chúng ta cùng làm rõ: *Vi cấu trúc thị trường là gì?*

Hãy tưởng tượng quý vị đang nhìn vào biểu đồ giá chứng khoán. Đa số nhà đầu tư thường nhìn vào 'nến ngày' hay 'nến tuần' – giống như việc đứng trên cao nhìn xuống một dòng sông, thấy dòng nước trôi rất êm đềm. 

Nhưng dự án của chúng em không nhìn như vậy. Chúng em soi dòng sông đó dưới **kính hiển vi**. Ở cấp độ mili-giây (tick data) và cấu trúc Sổ lệnh (Limit Order Book), thị trường không hề êm đềm. Nó là sự va đập điên cuồng của hàng triệu lệnh Mua (Bid) và Bán (Ask). Khối lượng chào mua, tốc độ hủy lệnh, dòng tiền vào ra... tất cả những chi tiết siêu nhỏ đó tương tác với nhau để tạo ra mức giá tiếp theo. Đó chính là Vi cấu trúc thị trường.

Để dự đoán sự va đập này, giới tài chính truyền thống bao lâu nay vẫn dùng một mô hình gọi là **Classical Random Walk (Bước ngẫu nhiên cổ điển)**."

---

## [Slide 3: Vấn đề của mô hình cũ (Classical Random Walk - CRW)]

**🗣️ Lời nói:**
"Hãy nói về mô hình truyền thống (CRW) một chút. Các thầy cô có thể hình dung CRW giống hệt trò chơi **'Tung đồng xu'**.

Tại mỗi tích tắc, mô hình CRW cho rằng giá cổ phiếu có 50% cơ hội tăng (nếu tung ra ngửa) và 50% cơ hội giảm (nếu tung ra sấp). Mỗi bước đi hoàn toàn độc lập, không nhớ gì về quá khứ. Trong toán học, người ta hay gọi vui đây là mô hình 'Gã say rượu đi bộ' (Drunkard's walk) – bước đi loạng choạng, không có định hướng.

**Nhược điểm chí mạng là gì?** Thị trường tài chính KHÔNG ngẫu nhiên như tung đồng xu. Thị trường có tâm lý đám đông, có sự hốt hoảng (panic sell) và sự hưng phấn (FOMO). Khi một tin tức xấu ra đời, giá không đi lên đi xuống 50-50 nữa, mà nó sẽ 'quán tính' rơi thẳng đứng. Mô hình tung đồng xu cổ điển hoàn toàn thất bại trong việc dự đoán những cú sốc hoặc những chuỗi tăng/giảm liên tiếp này."

---

## [Slide 4: Giải pháp đột phá - Quantum Random Walk (QRW) & Case Study]

**🗣️ Lời nói:**
"Và đó là lúc chúng em đưa **Vật lý lượng tử (Quantum Random Walks)** vào giải quyết vấn đề. 

Khác với đồng xu cổ điển chỉ có 'Sấp' hoặc 'Ngửa', trong thế giới lượng tử, một hạt có thể tồn tại ở cả trạng thái sấp và ngửa cùng một lúc (Superposition - Chồng chập). Hơn thế nữa, các xác suất này có thể **'giao thoa' (Interference)** với nhau. 
- *Giao thoa triệt tiêu:* Hai luồng sóng triệt tiêu nhau, tạo ra xác suất bằng 0.
- *Giao thoa cộng gộp:* Hai luồng sóng cộng hưởng, tạo ra đỉnh sóng vọt lên cực cao.

**📊 Để dễ hiểu, em xin phép trình bày một Case Study (Ví dụ thực tế) từ dữ liệu chạy mô hình của nhóm:**
Hãy xét một khoảnh khắc thị trường chuẩn bị có tin tức lớn (ví dụ: công bố lãi suất). 
- *Mô hình Cổ điển (CRW)* sẽ vẽ ra một biểu đồ hình chuông phân phối chuẩn (Bell curve). Nó dự đoán giá sẽ dao động nhẹ lân cận mức hiện tại.
- *Mô hình Lượng tử (QRW)* lại đọc được sự mất cân bằng trong Sổ lệnh (Limit Order Book). Nhờ hiện tượng 'Giao thoa cộng gộp' trong toán học lượng tử, mô hình QRW tự động dồn phân bố xác suất về hai phía cực đoan. Nó phát tín hiệu: *"Giá sẽ không đứng im, giá sẽ quét mạnh lên trên hoặc giật mạnh xuống dưới"*. 
Kết quả thực tế? Giá đã xảy ra một cú bứt phá (breakout) mạnh. QRW đã dự đoán chính xác hành vi bùng nổ này nhờ đặc tính giao thoa mà mô hình thường không có."

---

## [Slide 5: Kiến trúc Hệ thống & Kiểm định khắt khe (Rigorous Testing)]

**🗣️ Lời nói:**
"Nhưng thưa Hội đồng, một lý thuyết hay trên giấy là chưa đủ. Để chứng minh, nhóm đã xây dựng một **Quy trình Phần mềm (Pipeline)** khép kín từ A-Z:

1. **Xử lý Dữ liệu thô (Data Engineering):** Hút dữ liệu tick cực lớn, làm sạch và trích xuất các tính năng đặc thù của Vi cấu trúc (như Order Book Imbalance - Sự mất cân bằng sổ lệnh).
2. **Triển khai Thuật toán:** Code thuật toán QRW gốc, QRW Thích ứng (Adaptive QRW) và đặt lên bàn cân cùng các mô hình kinh điển như GARCH, GBM.
3. **Kiểm tra độ tin cậy (Chống gian lận dữ liệu):** Trong tài chính có một lỗi rất phổ biến là 'Look-ahead bias' (Nhìn lén dữ liệu tương lai để dự đoán hiện tại). Nhóm em đã viết **61 kịch bản kiểm thử tự động (automated tests)** cực kỳ nghiêm ngặt. Báo cáo Audit của dự án cam kết: Tuyệt đối không rò rỉ dữ liệu tương lai (forecast leakage) và không học vẹt (overfitting). Mọi đánh giá đều hoàn toàn công bằng."

---

## [Slide 6: Kết quả Thực nghiệm - Báo cáo Khoa học Trung thực]

**🗣️ Lời nói:**
"Tiếp theo, xin báo cáo kết quả thực nghiệm. Quan điểm của nhóm là làm khoa học phải trung thực tuyệt đối. Vì vậy, kết quả thu được là **Đa chiều (Mixed Results)**, cụ thể như sau:

1. **Ở tập dữ liệu Holdout (Tập dữ liệu độc lập, biến động mạnh):** Mô hình QRW chiến thắng rõ rệt. Giống như Case Study em vừa kể, QRW cực kỳ xuất sắc trong việc phán đoán các pha thị trường mất cân bằng.
2. **Tuy nhiên, ở bài test Walk-forward (Kiểm thử cuốn chiếu liên tục trong dài hạn):** QRW lại tỏ ra kém hiệu quả hơn một chút so với mô hình Fair Affine Baseline (một mô hình cổ điển rất đơn giản). 

**Tại sao lại như vậy? (Giảng giải nguyên nhân):**
Đây là một phát hiện khoa học rất thú vị của nhóm. Mô hình QRW quá nhạy cảm. Khi thị trường bình lặng trong thời gian dài, sự nhạy cảm của 'giao thoa lượng tử' vô tình phản ứng quá mức với những biến động nhiễu nhỏ, dẫn đến sai số cao hơn so với một mô hình tuyến tính đơn giản. 

*Bài học rút ra:* QRW không phải là 'chén thánh' dùng ở đâu cũng thắng, mà nó là một vũ khí hạng nặng. Nó phát huy tối đa sức mạnh khi cấu trúc thị trường đang ở trạng thái nhiễu động hoặc mất cân bằng cao."

---

## [Slide 7: Sản phẩm thực tiễn - Hệ thống Dashboard]

**🗣️ Lời nói:**
"Để chứng minh dự án không chỉ nằm trên những dòng code khó hiểu, nhóm đã đóng gói toàn bộ nghiên cứu thành một phần mềm thực tế: **Interactive Dashboard** (viết bằng Python Streamlit - nằm trong thư mục `src/dashboard/`).

Thay vì phải chạy code bằng dòng lệnh, bây giờ bất kỳ ai cũng có thể mở trình duyệt web lên. 
- Mọi người có thể chọn một ngày giao dịch cụ thể trên thanh menu.
- Chọn mô hình muốn đối chiếu (Quantum vs Classical).
- Hệ thống sẽ tự động gọi dữ liệu, chạy thuật toán và vẽ biểu đồ so sánh trực tiếp ngay trên màn hình. 
Bảng điểm (Scorecard) sẽ hiện ra rõ ràng để phân định thắng thua một cách trực quan, sinh động nhất."

---

## [Slide 8: Tổng kết & Lời cảm ơn]

**🗣️ Lời nói:**
"Kính thưa Hội đồng, để tóm tắt lại, dự án của chúng em đã thành công trong 3 việc:
1. Đưa thành công một lý thuyết Toán Lý phức tạp (QRW) vào ứng dụng thực tiễn của dữ liệu tài chính (Market Microstructure).
2. Xây dựng một quy trình công nghệ chuẩn mực, vượt qua 61 bài test chống rò rỉ dữ liệu.
3. Cung cấp một góc nhìn đánh giá khoa học, đa chiều và trung thực về ưu/nhược điểm của lượng tử so với các thuật toán truyền thống.

Phần trình bày của nhóm đến đây là kết thúc. Chúng em rất mong nhận được những câu hỏi, những màn 'chất vấn' từ Hội đồng để nhóm có cơ hội giải thích sâu hơn, cũng như nhận được những góp ý quý báu để hoàn thiện nghiên cứu. 
Xin chân thành cảm ơn!"
