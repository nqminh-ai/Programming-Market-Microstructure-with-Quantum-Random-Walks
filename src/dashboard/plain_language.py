"""Plain-language layer for the dashboard.

The platform was written for readers who already know what OBI, GARCH and a
Brier score are. This module holds the Vietnamese, jargon-free copy and the
components that wrap every panel in "what am I looking at / how do I read it /
what must I not conclude" so a non-technical visitor -- a competition judge, a
teacher, a parent -- can use the dashboard without a finance background.

Accessibility and honesty are handled together here on purpose. Making the
numbers easier to read also makes them easier to *mis*read: the demo artifacts
are computed from 1,000 rows and show a Sharpe near 5, which a lay reader would
reasonably take as "this makes money". Every explainer therefore carries a
`caveat` field, and it is not optional.
"""

from __future__ import annotations

from src.dashboard.design_system import COLORS


# ---------------------------------------------------------------------------
# What the project actually is and found
# ---------------------------------------------------------------------------

PROJECT_ONE_LINER = (
    "Một thí nghiệm khoa học: thử xem ý tưởng từ vật lý lượng tử có giúp dự "
    "đoán giá thị trường tốt hơn các phương pháp thông thường hay không."
)

PROJECT_ANSWER = (
    "Câu trả lời là **không**. Và chúng tôi coi việc chứng minh được điều đó "
    "một cách chặt chẽ mới là kết quả của dự án."
)

WHAT_IS_QRW = [
    (
        "Bước đi ngẫu nhiên là gì?",
        "Hình dung một người say đứng ở cột đèn, mỗi bước tung đồng xu để quyết "
        "định đi trái hay phải. Sau 100 bước, người đó ở đâu? Ta không biết chắc, "
        "nhưng biết được **xác suất**. Giá cổ phiếu được mô hình hoá gần giống vậy: "
        "mỗi giao dịch đẩy giá lên hoặc xuống một chút.",
    ),
    (
        "Vậy 'lượng tử' thêm gì vào?",
        "Trong thế giới lượng tử, hạt không chọn trái *hoặc* phải — nó đi **cả hai "
        "hướng cùng lúc**, rồi các khả năng đó **giao thoa** với nhau, giống hai gợn "
        "sóng nước gặp nhau: chỗ thì cộng hưởng mạnh lên, chỗ thì triệt tiêu nhau. "
        "Câu hỏi của dự án: sự giao thoa đó có bắt được điều gì về thị trường mà "
        "đồng xu thường không bắt được không?",
    ),
    (
        "Chúng tôi đã kiểm tra thế nào?",
        "Có một công tắc trong mô hình bật/tắt đúng phần 'giao thoa lượng tử' đó. "
        "Chúng tôi chạy mô hình hai lần — một lần bật, một lần tắt — trên hàng chục "
        "triệu giao dịch thật của Bitcoin, Ethereum và BNB, rồi so kết quả.",
    ),
    (
        "Kết quả?",
        "**Giống hệt nhau.** Bật hay tắt phần lượng tử không làm đổi kết quả tới "
        "5 chữ số thập phân. Nói cách khác, phần 'lượng tử' trong mô hình này "
        "**không đóng góp gì cả**. Ép nó mạnh lên thì dự đoán còn *tệ đi*.",
    ),
    (
        "Thế mô hình có thắng được gì không?",
        "Chúng tôi cho nó đấu với các phương pháp cổ điển mạnh. Nó **thua** — trên "
        "cả ba loại tiền. Một phép hồi quy tuyến tính đơn giản 5 hệ số đánh bại nó "
        "ở mọi trường hợp.",
    ),
    (
        "Giả sử có ai đó dự đoán ĐÚNG 100% thì sao?",
        "Vẫn **lỗ**. Đây là phần bất ngờ nhất của dự án. Mô hình dự đoán giá của "
        "giao dịch *kế tiếp*, mà khoảng thời gian đó giá chỉ nhúc nhích rất ít — "
        "trong khi mỗi lần mua bán đều mất phí cho sàn. Với Bitcoin, phí một vòng "
        "mua–bán lớn gấp **1.610 lần** biên độ giá mà ta đang cố đoán.\n\n"
        "Ví dụ cho dễ hình dung: giống như đoán đúng con xúc xắc để ăn 1 đồng, "
        "nhưng mỗi lần chơi phải trả 1.610 đồng tiền vé. Đoán đúng bao nhiêu lần "
        "cũng không cứu được. **Đây là giới hạn của khoảng thời gian dự báo, không "
        "phải của mô hình** — không kỹ thuật nào sửa được.",
    ),
    (
        "Vậy dự báo xa hơn để giá kịp chạy thì sao?",
        "Chúng tôi thử đúng điều đó, ở nhiều khoảng thời gian từ vài giây tới vài "
        "giờ. Kết quả là **kỹ năng và tiền nằm ở hai đầu đối lập**: ở khoảng ngắn "
        "(~26 giây) thì đoán đúng tới 64% — con số thật sự tốt — nhưng giá chưa "
        "kịp chạy đủ để trả phí. Kéo dài ra tới khi giá chạy đủ thì **khả năng "
        "đoán biến mất**, chỉ còn ngang mức tung đồng xu.\n\n"
        "Không một khoảng thời gian nào, trên bất kỳ đồng tiền nào, có lãi.",
    ),
    (
        "Đặt lệnh chờ để khỏi trả phí cao thì sao?",
        "Đây là mẹo tiêu chuẩn: thay vì mua ngay theo giá thị trường, ta đặt lệnh "
        "chờ ở giá tốt hơn và đợi người khác khớp vào. Sách vở nói cách này *được* "
        "hưởng chênh lệch giá thay vì phải trả.\n\n"
        "Chúng tôi **đo thử trên dữ liệu thật** thay vì tin sách. Kết quả ngược "
        "lại: người đặt lệnh chờ **mất** khoảng 1,2 phần vạn mỗi lệnh. Lý do rất "
        "đời: lệnh chờ của bạn chỉ được khớp khi có người *muốn* giao dịch ngược "
        "lại với bạn — và họ thường có lý do. Bạn được khớp đúng lúc thị trường "
        "sắp đi ngược hướng mình.",
    ),
]

WHY_NEGATIVE_MATTERS = [
    "Trong khoa học, một kết quả 'không' được kiểm chứng kỹ **có giá trị ngang** "
    "một kết quả 'có'. Nó ngăn người khác tốn công đi lại con đường cụt.",
    "Dự án đã **năm lần tự bác bỏ chính mình**, và lần nào cũng công bố thay vì "
    "giấu: một lỗi lập trình từng biến −0,0131 thành +0,0499; một kết quả không "
    "chạy lại được, chạy lại đúng cách thì *xấu hơn*; ba lỗi khiến chiến lược demo "
    "trông có lãi, sửa xong hoá ra lỗ 4,2%; một con số gọi là 'chênh lệch giá' hoá "
    "ra đo nhầm đại lượng khác, sai 22–33 lần; và giả định 'lệnh chờ được hưởng "
    "chênh lệch' hoá ra **sai cả dấu**.",
    "Điểm chung của cả năm: **không lỗi nào do người ngoài chỉ ra**. Tất cả đều do "
    "chính hệ thống kiểm toán của dự án tìm được — và cả năm đều làm kết quả xấu "
    "đi, chứ không phải đẹp lên.",
    "Chúng tôi **chấp nhận thua** một mô hình đơn giản hơn nhiều, và ghi rõ điều đó "
    "trong báo cáo thay vì chỉ so với đối thủ yếu để trông có vẻ thắng.",
    "Toàn bộ kết quả đều **chạy lại được**: mỗi con số gắn với mã nguồn, mã băm "
    "dữ liệu và hạt giống ngẫu nhiên cụ thể.",
    "Và **không con số nào** ở đây được gắn nhãn 'đã xác nhận'. Quy trình xác nhận "
    "cần 20 ngày thu dữ liệu thời gian thực mà dự án chưa thu — nên chúng tôi để "
    "nguyên nhãn 'thăm dò', kể cả với những kết quả **ủng hộ** kết luận của mình.",
]


# ---------------------------------------------------------------------------
# The boundary a lay reader most easily crosses by accident
# ---------------------------------------------------------------------------

DEMO_WARNING_TITLE = "Đọc kỹ trước khi xem các con số bên dưới"

DEMO_WARNING_BODY = [
    "Bảng điều khiển này là **bản trình diễn kỹ thuật** — nó cho thấy mô hình có "
    "thể được lắp vào những công cụ nào, **không phải** bằng chứng là mô hình kiếm "
    "được tiền.",
    "**Chiến lược demo này đang LỖ.** Profit Factor **0,095** (dưới 1 nghĩa là "
    "lỗ), lợi nhuận ròng **−4,2%**, Sharpe quy đổi năm **−48,6**. Chúng tôi hiển "
    "thị đúng như vậy thay vì giấu đi.",
    "Trước ngày 22/07/2026 bảng này từng hiện Sharpe **4,9** và Profit Factor "
    "**265** — trông như một chiến lược sinh lời. Đó là do **lỗi đo lường**: mã "
    "đếm mỗi *dòng dữ liệu đang giữ vị thế* thành một lệnh (651 thay vì 19), gọi "
    "một đại lượng thống kê khác là 'Sharpe', và phần dò tham số **quên trừ phí "
    "giao dịch**. Sửa xong thì dấu đảo ngược.",
    "Bài học đáng nhớ hơn cả con số: **một backtest có lỗi luôn có xu hướng trông "
    "đẹp hơn sự thật**, vì lỗi làm đẹp thì ít ai đi tìm, còn lỗi làm xấu thì bị "
    "phát hiện ngay.",
    "Số liệu demo chỉ dựa trên **500 dòng dữ liệu** và **34 lệnh**. Dù dấu có "
    "dương đi nữa thì mẫu này cũng quá nhỏ để kết luận bất cứ điều gì.",
    "Kết luận khoa học thật của dự án nằm ở tab **Bắt đầu ở đây**, và nó là một "
    "kết luận **phủ định**.",
]


# ---------------------------------------------------------------------------
# Per-tab guides
# ---------------------------------------------------------------------------

TAB_GUIDES: dict[str, dict[str, object]] = {
    "volatility": {
        "title": "Đo mức độ biến động",
        "what": "Biến động = giá đang 'nhảy' mạnh hay êm. Biến động cao nghĩa là "
                "giá dao động dữ dội, rủi ro lớn hơn. Bảng này so sánh vài cách "
                "ước lượng mức dao động sắp tới.",
        "how": [
            "Đường càng đi lên = thị trường càng động, càng rủi ro.",
            "Các đường màu khác nhau là các phương pháp dự đoán khác nhau. Chúng "
            "bám nhau càng sát thì càng đáng tin.",
            "Không có đường nào là 'đúng'. Đường trắng là mức dao động **đã thực "
            "sự xảy ra**; các đường còn lại là dự đoán, để bạn xem ai đoán sát hơn.",
        ],
        "caveat": "Đây là dữ liệu trình diễn. Đừng dùng để ra quyết định đầu tư.",
    },
    "risk": {
        "title": "Ước lượng rủi ro thua lỗ",
        "what": "Trả lời câu hỏi: 'Trong trường hợp xấu, tôi có thể mất bao nhiêu?' "
                "Máy tính mô phỏng hàng nghìn kịch bản tương lai rồi xem các kịch "
                "bản tệ nhất trông ra sao.",
        "how": [
            "Mỗi đường mảnh là **một kịch bản tương lai có thể xảy ra**, không phải "
            "một dự đoán. Cả chùm đường mới là câu trả lời.",
            "Chùm càng loe rộng = tương lai càng khó đoán.",
            "'Mức lỗ tối đa 95%' nghĩa là: trong 95 trên 100 trường hợp, mức lỗ "
            "không vượt quá con số đó. Vẫn còn 5 trường hợp tệ hơn.",
        ],
        "caveat": "Mọi mô hình rủi ro đều giả định tương lai giống quá khứ. Khủng "
                  "hoảng thật thường phá vỡ giả định đó.",
    },
    "signal": {
        "title": "Tín hiệu mua / bán thử nghiệm",
        "what": "Mô hình tự động gợi ý MUA, BÁN hoặc ĐỨNG NGOÀI dựa trên áp lực "
                "mua-bán đang diễn ra trên sổ lệnh.",
        "how": [
            "Nhãn xanh = gợi ý mua, đỏ = gợi ý bán, xám = không hành động.",
            "'Độ tin cậy' là mức chắc chắn của mô hình, **không phải** xác suất "
            "bạn sẽ có lãi.",
            "'Tỷ lệ đúng' quanh 14% nghe có vẻ tệ, nhưng chỉ số này một mình không "
            "nói lên điều gì — thắng ít lần nhưng thắng lớn vẫn có thể tốt.",
        ],
        "caveat": "ĐÂY KHÔNG PHẢI KHUYẾN NGHỊ ĐẦU TƯ. Tín hiệu chạy trên dữ liệu "
                  "trình diễn 1.000 dòng và chưa từng được kiểm chứng bằng tiền thật.",
    },
    "optimizer": {
        "title": "Dò tìm tham số tốt nhất",
        "what": "Mô hình có vài 'núm vặn'. Bảng này thử rất nhiều tổ hợp trên dữ "
                "liệu quá khứ để xem tổ hợp nào cho kết quả tốt nhất.",
        "how": [
            "Vùng sáng trên bản đồ nhiệt = tổ hợp tham số cho kết quả tốt hơn.",
            "Điều đáng lo là khi chỉ có **một điểm sáng nhỏ cô lập** — nghĩa là kết "
            "quả tốt chỉ do may mắn ở đúng một cấu hình.",
            "Vùng sáng **rộng và liền mạch** mới là dấu hiệu đáng tin.",
            "Xem ô **Deflated Sharpe** phía dưới bảng: nó trả lời thẳng câu hỏi "
            "'kết quả này có thật hay chỉ do thử nhiều lần rồi vớ được?'.",
        ],
        "caveat": "Dò càng nhiều tổ hợp thì càng dễ tìm được thứ 'có vẻ tốt' hoàn "
                  "toàn do ngẫu nhiên. Đây là cái bẫy phổ biến nhất trong ngành.",
    },
    "anomaly": {
        "title": "Phát hiện bất thường",
        "what": "Máy học xem thị trường lúc 'bình thường' trông thế nào, rồi báo "
                "động khi có gì đó lệch khỏi mức bình thường đó.",
        "how": [
            "Sigma (σ) là thước đo độ lệch. 1σ là dao động thường ngày; 3σ là hiếm; "
            "trên 4σ là rất hiếm.",
            "Cảnh báo **không** có nghĩa là 'sắp sập'. Nó chỉ nghĩa là 'chỗ này khác "
            "thường, đáng nhìn kỹ'.",
            "Nhiều cảnh báo cùng lúc thường phản ánh tin tức hoặc một lệnh lớn, chứ "
            "không phải sự cố.",
        ],
        "caveat": "Ngưỡng cảnh báo do con người chọn. Hạ ngưỡng xuống thì cảnh báo "
                  "nào cũng 'đúng' — vì nó báo liên tục.",
    },
}


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------

GLOSSARY: list[tuple[str, str]] = [
    ("QRW (Quantum Random Walk)",
     "Bước đi ngẫu nhiên lượng tử — mô hình toán học *lấy cảm hứng* từ vật lý "
     "lượng tử. Dự án KHÔNG cho rằng thị trường vận hành theo cơ học lượng tử, "
     "và KHÔNG dùng máy tính lượng tử."),
    ("Biến động (Volatility)",
     "Mức độ giá nhảy lên xuống. Cao = rủi ro lớn, không có nghĩa là giá sẽ giảm."),
    ("OBI (Order Book Imbalance)",
     "Chênh lệch giữa lượng người muốn mua và lượng người muốn bán đang chờ trên "
     "sổ lệnh. Nhiều người chờ mua hơn ⟹ giá có xu hướng nhích lên."),
    ("Tick",
     "Một giao dịch đơn lẻ. Bitcoin có thể có hàng triệu tick mỗi ngày."),
    ("Sharpe",
     "Lợi nhuận thu được trên mỗi đơn vị rủi ro chấp nhận, đã quy đổi về mức "
     "một năm. Quỹ đầu tư giỏi ở đời thực thường đạt 1–2. Âm nghĩa là lỗ."),
    ("t-statistic",
     "**Không phải Sharpe**, dù dễ bị nhầm. Nó đo mức chắc chắn rằng lợi nhuận "
     "trung bình khác 0, và **tăng lên khi bạn có nhiều dữ liệu hơn** — nên "
     "không so sánh được giữa hai backtest dài ngắn khác nhau. Sharpe thật thì "
     "không có tính chất đó. Bảng này từng hiển thị t-statistic dưới tên Sharpe."),
    ("Profit Factor",
     "Tổng tiền thắng chia tổng tiền thua. Trên 1 là lãi, dưới 1 là lỗ. "
     "Chiến lược demo hiện đạt 0,095 — tức thua gấp hơn 10 lần thắng."),
    ("Lệnh (round trip)",
     "Một lần mở vị thế rồi đóng lại — dù giữ qua bao nhiêu giao dịch đi nữa vẫn "
     "tính là **một** lệnh. Đếm nhầm mỗi khoảnh khắc đang giữ vị thế thành một "
     "lệnh sẽ thổi phồng số lệnh lên hàng chục lần và làm sai mọi tỷ lệ tính theo "
     "lệnh."),
    ("Phí giao dịch",
     "Khoản sàn thu mỗi lần mua/bán, ở đây là 0,05% mỗi chiều. Nghe nhỏ nhưng "
     "với chiến lược giao dịch dày, nó thường lớn hơn cả lợi nhuận gộp."),
    ("Lệnh chờ (maker) và lệnh khớp ngay (taker)",
     "**Taker** mua bán ngay theo giá thị trường — chắc chắn được, nhưng trả phí "
     "cao. **Maker** đặt lệnh chờ ở giá tốt hơn và đợi người khác khớp vào — phí "
     "thấp hơn, nhưng không chắc có được khớp hay không."),
    ("Adverse selection (bị chọn ngược)",
     "Lệnh chờ của bạn chỉ được khớp khi có người *muốn* giao dịch ngược lại — và "
     "họ thường có lý do. Kết quả: bạn hay được khớp đúng lúc thị trường sắp đi "
     "ngược hướng mình. Dự án đo được khoản này là **1,2 phần vạn mỗi lệnh**, "
     "đủ để xoá sạch lợi thế phí thấp của lệnh chờ."),
    ("Ngưỡng hoà vốn",
     "Tỉ lệ đoán đúng tối thiểu để không lỗ sau khi trả phí. Nếu ngưỡng này **vượt "
     "100%** thì kể cả đoán đúng mọi lần vẫn lỗ — đúng tình huống của dự án ở "
     "khoảng thời gian đang dùng."),
    ("Drawdown",
     "Mức sụt giảm sâu nhất từ đỉnh xuống đáy. Cho biết bạn phải chịu đau đến đâu."),
    ("Brier score",
     "Cách chấm điểm dự đoán xác suất. **Càng thấp càng tốt** — ngược với điểm thi."),
    ("CRPS",
     "Giống Brier nhưng chấm cả một phân phối dự đoán thay vì một xác suất. "
     "Cũng càng thấp càng tốt."),
    ("GARCH / GBM",
     "Hai mô hình cổ điển kinh điển trong tài chính, dùng làm đối thủ để so sánh. "
     "Chúng đã có từ nhiều thập kỷ và rất khó đánh bại."),
    ("Baseline (mô hình đối chứng)",
     "Đối thủ để so sánh. Chọn đối thủ yếu thì mô hình nào cũng 'thắng' — nên "
     "chúng tôi cố tình chọn đối thủ mạnh."),
    ("Exploratory (thăm dò)",
     "Kết quả tìm tòi, chưa phải kết luận chính thức. Muốn thành chính thức phải "
     "đăng ký phương pháp trước rồi mới thu dữ liệu mới."),
    ("Deflated Sharpe Ratio",
     "Xác suất chiến lược thật sự có kỹ năng, **sau khi trừ đi lợi thế của việc "
     "đã thử rất nhiều cấu hình**. Thử 500 tổ hợp thì kiểu gì cũng có một tổ hợp "
     "trông đẹp do may mắn; chỉ số này hỏi 'đẹp hơn mức may mắn thường thấy "
     "chưa?'. Trên 0,95 mới đáng tin. Chiến lược demo ở đây đạt **0,000**."),
    ("Overfitting (quá khớp)",
     "Mô hình học thuộc lòng dữ liệu cũ thay vì hiểu quy luật — nên đúng với quá "
     "khứ nhưng sai với tương lai. Kẻ thù số một của lĩnh vực này."),
]


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tab_explainer(key: str) -> None:
    """Render the "what / how to read / caveat" box at the top of a tab."""
    import streamlit as st

    guide = TAB_GUIDES[key]
    how_html = "".join(
        f'<div style="margin-top:0.35rem;">• {_escape(str(line))}</div>'
        for line in guide["how"]  # type: ignore[union-attr]
    )
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left:3px solid {COLORS['accent_cyan']};
             margin-bottom:1rem;">
            <div style="font-family:'Inter',sans-serif; font-size:1.05rem;
                 font-weight:600; color:{COLORS['text_primary']};">
                {_escape(str(guide['title']))}
            </div>
            <div style="margin-top:0.5rem; color:#C7D3E0; font-size:0.9rem;
                 line-height:1.6;">
                {_escape(str(guide['what']))}
            </div>
            <div class="kpi-label" style="margin-top:0.9rem;">Cách đọc biểu đồ</div>
            <div style="color:#C7D3E0; font-size:0.86rem; line-height:1.55;">
                {how_html}
            </div>
            <div style="margin-top:0.8rem; padding-top:0.6rem;
                 border-top:1px solid {COLORS['border_subtle']};
                 color:{COLORS['accent_yellow']}; font-size:0.84rem;">
                ⚠ {_escape(str(guide['caveat']))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def demo_disclaimer() -> None:
    """The banner every visitor must see before reading any performance number."""
    import streamlit as st

    body = "".join(
        f'<div style="margin-top:0.45rem;">• {line}</div>'
        for line in DEMO_WARNING_BODY
    )
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left:3px solid {COLORS['accent_red']};
             background:rgba(255,68,68,0.05);">
            <div style="font-family:'Inter',sans-serif; font-size:1rem;
                 font-weight:600; color:{COLORS['accent_red']};">
                {DEMO_WARNING_TITLE}
            </div>
            <div style="margin-top:0.5rem; color:#D8E2EE; font-size:0.88rem;
                 line-height:1.6;">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def glossary_expander() -> None:
    """A jargon lookup available on every tab."""
    import streamlit as st

    with st.expander("Từ điển thuật ngữ — bấm để mở"):
        for term, meaning in GLOSSARY:
            st.markdown(f"**{term}** — {meaning}")


def render_start_here() -> None:
    """The landing tab: what this project is, in language anyone can follow."""
    import streamlit as st

    st.markdown(
        f"""
        <div style="padding:1.6rem 0 0.6rem 0;">
            <div style="font-family:'Inter',sans-serif; font-size:2rem;
                 font-weight:600; color:{COLORS['text_primary']}; line-height:1.25;">
                Bước đi ngẫu nhiên lượng tử có dự đoán được thị trường không?
            </div>
            <div style="margin-top:0.8rem; color:#C7D3E0; font-size:1.02rem;
                 line-height:1.65; max-width:52rem;">
                {PROJECT_ONE_LINER}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="kpi-card" style="border-left:3px solid {COLORS['accent_cyan']};
             margin:1rem 0 1.5rem 0;">
            <div class="kpi-label">Kết luận của dự án</div>
            <div style="color:{COLORS['text_primary']}; font-size:1.05rem;
                 line-height:1.6;">
                Câu trả lời là <b>không</b>. Và chúng tôi coi việc chứng minh được
                điều đó một cách chặt chẽ mới chính là kết quả của dự án.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    demo_disclaimer()

    st.markdown("### Giải thích cho người chưa biết gì")
    for question, answer in WHAT_IS_QRW:
        with st.expander(question, expanded=False):
            st.markdown(answer)

    st.markdown("### Vì sao một kết quả “không” lại đáng giá?")
    for line in WHY_NEGATIVE_MATTERS:
        st.markdown(f"- {line}")

    st.markdown("### Các tab còn lại là gì?")
    st.markdown(
        "Năm tab sau là **bản trình diễn ứng dụng**: nếu có một mô hình dự báo "
        "thị trường, người ta sẽ lắp nó vào những công cụ nào. Chúng minh hoạ "
        "phần kỹ thuật, **không** minh hoạ kết luận khoa học ở trên."
    )
    for key, guide in TAB_GUIDES.items():
        st.markdown(f"- **{guide['title']}** — {guide['what']}")

    st.markdown("---")
    glossary_expander()
