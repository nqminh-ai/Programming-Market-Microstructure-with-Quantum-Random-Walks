# Hướng phát triển

Tài liệu này vạch các bước tiếp theo của dự án. Nó giữ đúng tính cách đã định
hình toàn bộ công trình: mỗi hướng **bám vào một kết quả đã xác lập** và **một
giới hạn cụ thể**, nêu rõ hạ tầng nào đã có, và **kết cục kỳ vọng trung thực** —
kể cả khi kết cục đó là một kết quả "không" nữa. Dự án này không hứa ưu thế lượng
tử; nó hứa đo đạc chặt chẽ. Roadmap cũng vậy.

Thứ tự ưu tiên đi từ **gần và chắc** (hạ tầng sẵn sàng) tới **xa và mở** (câu hỏi
khoa học chưa ngã ngũ).

---

## Tầng 0 — Đóng vòng confirmatory *(ưu tiên cao nhất, hạ tầng đã sẵn sàng)*

**Vì sao:** Toàn bộ Phase 1–6 mang nhãn `EXPLORATORY_ONLY_NOT_CONFIRMATORY`. Ba
giới hạn nền tảng (#1 OBI là proxy, #2 chưa có gì confirmatory, #6 chưa mô hình
xác suất khớp) đều chờ **một thứ duy nhất không rút ngắn được bằng tính toán**:
≥20 ngày UTC thu L2 LOB thời gian thực.

**Đã có:** protocol confirmatory viết xong, đóng băng, pre-register
([data_collection_todo.md](data_collection_todo.md)); runner
[`collect_confirmatory.py`](../scripts/operations/collect_confirmatory.py) hard-code
`obi_source="lob"` nên không thể âm thầm rơi về proxy; 11 test cưỡng chế điều
khoản; chunked + tự resume; manifest bất biến SHA-256/ngày.

**Việc cần làm:** khởi động thu 20+ ngày → chạy đúng protocol đã đóng băng **một
lần**, không nhìn test trước → công bố kết quả **dù nó ra chiều nào**.

**Kết cục kỳ vọng:** đây là lần đầu dự án có quyền gỡ nhãn exploratory. Kết quả
nhiều khả năng vẫn là "QRW không có lợi thế bền vững" (mọi bằng chứng exploratory
đều chỉ hướng đó), nhưng lúc đó nó là **kết luận confirmatory**, không phải thăm
dò. Số ngày đã thu hiện tại: **0**.

---

## Tầng 1 — Từ proxy sang vi cấu trúc thật

**Vì sao:** OBI hiện là **trade-flow proxy** dựng từ hướng lệnh khớp, không phải
mất cân bằng sổ lệnh L2 thật (giới hạn #1). Và phân tích giao dịch §5e cho thấy
realised half-spread **âm** (~1,2 bps) — lệnh chờ *trả tiền* chứ không ăn spread —
nhưng con số đó mới là **cận dưới** của chi phí maker vì chưa có mô hình xác suất
khớp và vị trí hàng đợi (giới hạn #6).

**Việc cần làm:**
- Thay OBI proxy bằng **order-book imbalance thật** từ L2 depth (dữ liệu Tầng 0
  cung cấp): mất cân bằng theo mức giá, độ sâu, áp lực sổ lệnh.
- Dựng **mô hình xác suất khớp + vị trí hàng đợi**: một lệnh chờ ở mức giá `k`
  thực sự được khớp với xác suất bao nhiêu trước khi giá bỏ đi? Điều này biến chi
  phí maker từ cận dưới thành **con số thật**.

**Kết cục kỳ vọng:** biết chính xác chi phí maker thực. Nếu adverse selection +
xác suất khớp thấp khiến maker vẫn lỗ, §5e được củng cố bằng số cứng thay vì cận
dưới. Nếu ngược lại, đó là ô đầu tiên đáng xem lại — và sẽ được báo cáo nguyên
trạng.

---

## Tầng 2 — Bản đồ horizon × chi phí

**Vì sao:** §5e chứng minh ở horizon **1 tick**, phí một vòng lớn hơn biên độ giá
tới **1.610 lần** (BTC) — ngưỡng hòa vốn vượt 100%, dự đoán hoàn hảo vẫn lỗ.
Nhưng dự án cũng thấy **edge co lại khi tăng số fold/horizon**, và tỷ lệ
chi-phí/biến-động thay đổi theo horizon. Câu hỏi chưa trả lời hệ thống: **có
horizon nào (giây–phút) mà một dự báo thật đủ vượt chi phí không?**

**Việc cần làm:** quét toàn bộ mặt phẳng **horizon × kịch bản phí**, mỗi ô tính
ngưỡng hòa vốn `p* = 0,5 + chi_phí/(2·E|r_h|)` và độ chính xác đạt được, trên cả
ba tài sản với cỡ mẫu khử chồng lấp. Hạ tầng đã có phần lớn:
[`horizon_feasibility.py`](../scripts/research/horizon_feasibility.py) và
[`horizon_label_baselines.py`](../scripts/research/horizon_label_baselines.py).

**Kết cục kỳ vọng:** hoặc tìm ra một **vùng horizon khả thi** (và khi đó câu hỏi
chuyển thành "mô hình nào đạt `p*` ở đó"), hoặc **chứng minh không tồn tại vùng
nào** — một kết quả "không" mạnh hơn và tổng quát hơn ô 1-tick hiện tại.

---

## Tầng 3 — Câu hỏi lượng tử thật sự

**Vì sao:** đây là dự án *AI Quantum*, và phát hiện trung tâm là **pha
`alpha_phase` ≈ 0 trên cả ba tài sản** — cơ chế giao thoa lượng tử đóng góp bằng
0. Trước khi kết luận điều này là **phổ quát**, cần loại hai khả năng nó chỉ là
artifact của thiết lập hiện tại:

- **Toán tử heavy-tail phải thật sự unitary.** Prototype `qrw_heavy_tail.py` hiện
  dùng bước nhảy Bernoulli/Pareto **cổ điển** — đã bị loại khỏi mọi tuyên bố cơ
  chế QRW. Dựng một **heavy-tailed unitary shift** đúng nghĩa là con đường **duy
  nhất** để kiểm xem cơ chế lượng tử có thể đóng góp gì khi phân phối đuôi nặng
  được đưa vào một cách coherent hay không.
- **Tìm chế độ thị trường mà pha không tầm thường.** Nếu tồn tại (sụp đổ, halt,
  luồng cực đoan) một chế độ nơi giao thoa pha khác 0 có ý nghĩa, đó là ngoại lệ
  đáng giá. Nếu không, kết quả "pha = 0" được nâng từ quan sát lên **kết luận
  tổng quát**.

**Kết cục kỳ vọng:** nhiều khả năng khẳng định lại "pha không giúp gì" — nhưng
lần này sau khi đã **chủ động tấn công** giả thuyết ngược bằng đúng toán tử
unitary và đúng chế độ, thay vì mặc định. Đây là khác biệt giữa "chưa tìm thấy"
và "đã tìm và không có".

---

## Tầng 4 — Mở rộng chế độ thị trường & benchmark hiện đại

**Vì sao:** 69 ngày (2026-05-13 → 07-20) là **~2,5 tháng, một chế độ thị trường**.
Kết luận "không có edge" có thể phụ thuộc chế độ. Và baseline mạnh nhất đang là
OrderFlow AR(5) — một logistic 5 hệ số; chưa đấu với mô hình chuỗi hiện đại.

**Việc cần làm:**
- **Phủ nhiều chế độ:** kéo dữ liệu qua các giai đoạn bull/bear/biến-động-cao để
  kiểm "không có edge" có phổ quát không.
- **Thêm baseline hiện đại:** transformer/mô hình chuỗi, neural SDE, rough
  volatility. Nếu QRW đã thua một hồi quy tuyến tính, benchmark trung thực đòi so
  cả với phương pháp mạnh — **không** phải để QRW trông thắng bằng cách chọn đối
  thủ yếu.

**Kết cục kỳ vọng:** biết kết luận hiện tại **bền tới đâu** qua chế độ và qua độ
mạnh của đối thủ. Cả hai chiều đều là thông tin, kể cả khi làm QRW trông tệ hơn.

---

## Tầng 5 — Đóng gói phương pháp *(đóng góp có thể công bố)*

**Vì sao:** đóng góp thật của dự án **không** phải "một mô hình lượng tử", mà là
**bộ máy tự kiểm khiến việc tự lừa mình trở nên khó**: nó đã tự bác bỏ chính mình
**sáu lần**, mỗi lần đều công bố. Trong một lĩnh vực mà kết quả dương tính không
tái lập được là vấn đề hệ thống, chính cái máy đó mới đáng tổng quát hóa.

**Đã có:** manifest tái lập [`reproduce.py`](../scripts/operations/reproduce.py)
(21 artifact), kỷ luật provenance (SHA-256 + git commit + protocol version),
chuỗi `lệnh → artifact → prose` ép bằng test
([`test_report_numbers.py`](../tests/test_report_numbers.py)), pre-registration
cưỡng chế bằng code, [REPRODUCE.md](../REPRODUCE.md).

**Việc cần làm:** tách bộ máy này thành một **framework tái dùng** để đánh giá
tuyên bố ML/lượng tử trên dữ liệu tài chính — "pipeline kết quả âm nghiêm ngặt".
Đây là một đóng góp **methods paper**, độc lập với việc QRW thắng hay thua.

**Kết cục kỳ vọng:** giá trị của dự án sống lâu hơn kết luận cụ thể về QRW, dưới
dạng một quy trình người khác dùng lại được.

---

## Điều roadmap này **không** làm

Nhất quán với phần "điều dự án không làm" trong [tóm tắt điều hành](executive_summary.md):

- **Không** gắn nhãn confirmatory lên dữ liệu exploratory để đi nhanh. Tầng 0 làm
  đúng cách hoặc không làm.
- **Không** làm QRW trông thắng bằng cách chọn đối thủ yếu, chọn window có lợi,
  hay quét endpoint sau khi xem kết quả.
- **Không** giấu kết quả "không". Mỗi tầng ở trên đều có thể ra thêm một kết quả
  âm, và nếu vậy nó sẽ được công bố y như sáu lần trước.

## Bảng tóm tắt ưu tiên

| Tầng | Hướng | Mở khóa | Hạ tầng | Kết cục có thể là "không"? |
|---|---|---|---|:--:|
| 0 | Đóng vòng confirmatory | #1, #2, #6 | **Sẵn sàng** | Có |
| 1 | Vi cấu trúc L2 thật + xác suất khớp | #1, #6 | Một phần (cần dữ liệu Tầng 0) | Có |
| 2 | Bản đồ horizon × chi phí | Vùng khả thi giao dịch | **Phần lớn đã có** | Có |
| 3 | Toán tử unitary + chế độ pha | Câu hỏi lượng tử | Prototype cần thay | Có (khả năng cao) |
| 4 | Đa chế độ + benchmark hiện đại | Độ bền kết luận | Cần dữ liệu + model mới | Có |
| 5 | Đóng gói phương pháp | Đóng góp publishable | **Sẵn sàng** | Không áp dụng |
