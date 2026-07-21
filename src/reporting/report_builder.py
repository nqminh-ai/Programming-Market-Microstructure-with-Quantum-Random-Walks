"""Build the Phase 6 technical report and seminar slides."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


@dataclass(frozen=True)
class ReportContext:
    """Quantitative values embedded consistently across Phase 6 documents."""

    feature_path: str
    train_rows: int
    holdout_rows: int
    n_steps: int
    n_paths: int
    random_seed: int
    top_model: str
    top_mean_marginal_crps: float
    qrw_rank: int
    qrw_mean_marginal_crps: float
    qrw_mean_direction_log_loss: float
    empirical_beta: float
    qrw_beta: float
    qrw_beta_ci_low: float
    qrw_beta_ci_high: float
    walk_forward_folds: int
    walk_forward_folds_qrw_better: int
    walk_forward_brier_difference: float
    walk_forward_ci_low: float
    walk_forward_ci_high: float
    walk_forward_verdict: str
    top_directional_model: str
    qrw_directional_rank: int
    qrw_directional_log_loss: float
    directional_test_events: int

    def __post_init__(self) -> None:
        """Reject NaN/inf/negative-count values before they reach a report.

        A malformed upstream artifact could otherwise produce a "complete"
        report or PDF built from nonsensical numbers without any error.
        """
        non_negative_counts = {
            "train_rows": self.train_rows,
            "holdout_rows": self.holdout_rows,
            "n_steps": self.n_steps,
            "n_paths": self.n_paths,
            "directional_test_events": self.directional_test_events,
            "walk_forward_folds": self.walk_forward_folds,
        }
        for name, value in non_negative_counts.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        for name in ("qrw_rank", "qrw_directional_rank"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")

        if not 0 <= self.walk_forward_folds_qrw_better <= self.walk_forward_folds:
            raise ValueError(
                "walk_forward_folds_qrw_better must be between 0 and "
                f"walk_forward_folds ({self.walk_forward_folds}), got "
                f"{self.walk_forward_folds_qrw_better}"
            )

        finite_fields = (
            "top_mean_marginal_crps",
            "qrw_mean_marginal_crps",
            "qrw_mean_direction_log_loss",
            "empirical_beta",
            "qrw_beta",
            "qrw_beta_ci_low",
            "qrw_beta_ci_high",
            "walk_forward_brier_difference",
            "walk_forward_ci_low",
            "walk_forward_ci_high",
            "qrw_directional_log_loss",
        )
        for name in finite_fields:
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")

        non_negative_fields = (
            "top_mean_marginal_crps",
            "qrw_mean_marginal_crps",
            "qrw_mean_direction_log_loss",
            "qrw_directional_log_loss",
        )
        for name in non_negative_fields:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        if self.qrw_beta_ci_low > self.qrw_beta_ci_high:
            raise ValueError(
                "qrw_beta_ci_low must be <= qrw_beta_ci_high: "
                f"{self.qrw_beta_ci_low} > {self.qrw_beta_ci_high}"
            )
        if self.walk_forward_ci_low > self.walk_forward_ci_high:
            raise ValueError(
                "walk_forward_ci_low must be <= walk_forward_ci_high: "
                f"{self.walk_forward_ci_low} > {self.walk_forward_ci_high}"
            )


REFERENCES = (
    "Aharonov, Davidovich, and Zagury (1993). Quantum random walks.",
    "Ambainis et al. (2001). One-dimensional quantum walks.",
    "Konno (2002). Quantum random walks in one dimension.",
    "Kempe (2003). Quantum random walks: an introductory overview.",
    "Kendon (2007). Decoherence in quantum walks.",
    "Cont and de Larrard (2013). Price dynamics in a Markovian limit order market.",
    "Cont, Kukanov, and Stoikov (2014). The price impact of order book events.",
    "Engle (1982). Autoregressive conditional heteroscedasticity.",
    "Bollerslev (1986). Generalized autoregressive conditional heteroskedasticity.",
    "Black and Scholes (1973). The pricing of options and corporate liabilities.",
    "Mandelbrot (1963). The variation of certain speculative prices.",
    "Hill (1975). A simple general approach to inference about the tail.",
    "Ljung and Box (1978). On a measure of lack of fit in time series models.",
    "Benjamini and Hochberg (1995). Controlling the false discovery rate.",
)


def _format_probability(value: float) -> str:
    return f"{value:.3e}" if value < 0.001 else f"{value:.4f}"


def build_final_report_markdown(context: ReportContext) -> str:
    """Tạo báo cáo UTF-8 tiếng Việt với diễn giải khớp dấu của kết quả walk-forward.

    walk_forward_brier_difference âm nghĩa là QRW thắng baseline; các câu văn
    tường thuật phải phản ánh đúng dấu này thay vì hardcode một kết luận cố
    định, để báo cáo không tự mâu thuẫn với chính con số nó vừa in ra.
    """
    qrw_beats_baseline = context.walk_forward_brier_difference < 0.0
    summary_evidence_line = (
        "Kết quả này không thiết lập bằng chứng về ưu thế dự báo của QRW."
        if not qrw_beats_baseline
        else "Walk-forward cho thấy QRW có lợi thế Brier score so với baseline "
        "trên dữ liệu đã kiểm toán; đây chưa phải kết luận confirmatory (xem §5)."
    )
    walk_forward_limitation_line = (
        "Walk-forward hiện không ủng hộ lợi thế QRW."
        if not qrw_beats_baseline
        else "Walk-forward hiện ủng hộ lợi thế QRW trên dữ liệu đã kiểm toán, "
        "nhưng chưa được xác nhận trên toàn bộ dataset gốc."
    )
    conclusion_evidence_line = (
        "kết quả khoa học hiện tại vẫn âm: chưa có bằng chứng QRW vượt baseline "
        "cổ điển."
        if not qrw_beats_baseline
        else "kết quả khoa học hiện tại là dương cho QRW trên dữ liệu đã kiểm "
        "toán, nhưng vẫn cần xác nhận trên toàn bộ dataset gốc trước khi coi là "
        "confirmatory."
    )
    return f"""# Báo cáo cuối: Quantum Random Walk cho vi cấu trúc thị trường

## Tóm tắt

Nghiên cứu đánh giá quantum random walk (QRW) density-matrix với coin thích
nghi và decoherence. Tập đánh giá có {context.train_rows:,} dòng train,
{context.holdout_rows:,} dòng holdout, {context.n_steps:,} horizon và
{context.n_paths:,} mẫu biên cho mỗi mô hình/horizon. Endpoint chính là CRPS
biên trung bình; log loss hướng chỉ dùng để phá hòa.

{context.top_model} có CRPS thấp nhất ({context.top_mean_marginal_crps:.6g}).
QRW xếp hạng {context.qrw_rank}, CRPS {context.qrw_mean_marginal_crps:.6g} và
log loss hướng {context.qrw_mean_direction_log_loss:.6g}. {summary_evidence_line}

## 1. Ngữ nghĩa đánh giá

QRW chỉ tạo phân phối biên fixed-origin tại từng horizon. Các mẫu độc lập giữa
các horizon không tạo thành trajectory. Vì vậy QRW không được đưa vào thống kê
sai phân lợi suất, ACF, tail index hoặc Diebold–Mariano dựa trên đường đi.

Diebold–Mariano chỉ dùng loss rolling-origin một bước, căn chỉnh trên cùng
timestamp. ACF và tail là chẩn đoán `trajectory_only` cho baseline có đường đi
thật.

## 2. Dữ liệu và giao thức

Artifact feature: `{context.feature_path}`.

| Thành phần | Giá trị |
|---|---:|
| Train | {context.train_rows:,} dòng |
| Holdout | {context.holdout_rows:,} dòng |
| Horizon | {context.n_steps:,} |
| Mẫu biên/mô hình/horizon | {context.n_paths:,} |
| Seed | {context.random_seed} |

OBI trong dữ liệu hiện tại là proxy mất cân bằng luồng giao dịch, không phải
sổ lệnh giới hạn L2 đồng bộ. Calibration và mô hình AR(1) cho OBI tương lai chỉ
dùng train.

## 3. Kết quả định lượng

| Đại lượng | Giá trị quan sát |
|---|---:|
| Mô hình đứng đầu | {context.top_model} |
| CRPS đứng đầu | {context.top_mean_marginal_crps:.6g} |
| Hạng tổng thể của QRW | {context.qrw_rank} |
| CRPS biên trung bình của QRW | {context.qrw_mean_marginal_crps:.6g} |
| Log loss hướng của QRW | {context.qrw_mean_direction_log_loss:.6g} |
| Beta phương sai thực nghiệm | {context.empirical_beta:.4f} |
| Beta phương sai QRW | {context.qrw_beta:.4f} |
| Khoảng tin cậy 95% của beta QRW | [{context.qrw_beta_ci_low:.4f}; {context.qrw_beta_ci_high:.4f}] |
| Thống kê phụ thuộc đường đi của QRW | không áp dụng |
| Baseline hướng đứng đầu | {context.top_directional_model} |
| Hạng hướng của QRW | {context.qrw_directional_rank} |
| Log loss hướng QRW trong benchmark mạnh | {context.qrw_directional_log_loss:.6g} |
| Số sự kiện test dùng chung | {context.directional_test_events:,} |

![Tiến hóa xác suất](../figures/prob_evolution.gif)

![Quy luật co giãn phương sai](../figures/variance_scaling.png)

![Dải biên QRW và đường đi baseline](../figures/sample_paths.png)

![Bảng điểm](../figures/scorecard.png)

## 4. Giới hạn

1. Dữ liệu hoạt động chỉ gồm một cửa sổ rất ngắn.
2. OBI là proxy luồng giao dịch, không phải sổ lệnh L2 đồng bộ.
3. Kết quả cross-asset cũ chưa đánh giá đầy đủ mọi ngày có sẵn.
4. {walk_forward_limitation_line}
5. Chưa có holdout đa ngày untouched để đưa ra kết luận confirmatory.

## 5. Kết quả Walk-forward bắt buộc

QRW thắng **{context.walk_forward_folds_qrw_better}/{context.walk_forward_folds}**
fold. Chênh lệch Brier gộp `QRW - baseline` là
**{context.walk_forward_brier_difference:+.6f}**, khoảng tin cậy 95%
**[{context.walk_forward_ci_low:+.6f}; {context.walk_forward_ci_high:+.6f}]**.
Giá trị dương nghĩa là QRW kém hơn baseline.

Kết luận từ artifact audit có provenance:

> {context.walk_forward_verdict}

## 6. Kết luận

Pipeline có thể kiểm toán ngữ nghĩa phân phối biên, và {conclusion_evidence_line}
Kết luận chỉ được xem xét lại sau khi protocol và provenance được đóng băng,
đồng thời có ít nhất 20 ngày UTC mới cho mỗi tài sản với sổ lệnh L2 đồng bộ.
"""


def build_presentation_markdown(context: ReportContext) -> str:
    """Return a 16-slide Marp-compatible Markdown presentation."""
    slides = (
        ("Quantum Random Walks for Market Microstructure", "Phase 6 final presentation"),
        ("Research Question", "Can a causally calibrated QRW reproduce short-horizon market properties better than classical baselines?"),
        ("Why QRW?", "Interference, ballistic spreading, and a tunable coherent-to-diffusive transition."),
        ("QRW vs CRW", "Coherent QRW variance grows near t^2; symmetric CRW variance grows as t."),
        ("Active Data", f"June 12, 2026 BTCUSDT: {context.train_rows:,} train rows and {context.holdout_rows:,} later holdout rows."),
        ("Causal Pipeline", "Trailing cleaner, chronological split, disjoint validation, fixed seed, and matched samples."),
        ("Market Mapping", "OBI and direction enter a unitary phase-adaptive coin; intensity controls dephasing."),
        ("Benchmark Models", "QRW Adaptive, three CRWs, GARCH(1,1), and GBM."),
        ("Evaluation", "Marginal CRPS, direction log loss, marginal variance scaling, and aligned rolling one-step DM."),
        ("Probability Evolution", "../figures/prob_evolution.gif"),
        ("Variance Scaling", "../figures/variance_scaling.png"),
        ("Distribution Shape", "../figures/return_distributions.png"),
        ("Dependence and Paths", "../figures/acf_comparison.png"),
        ("Scorecard", f"{context.top_model} ranks first; QRW ranks {context.qrw_rank}."),
        (
            "Walk-forward Primary Result",
            f"QRW wins {context.walk_forward_folds_qrw_better}/"
            f"{context.walk_forward_folds} folds; pooled Brier difference "
            f"{context.walk_forward_brier_difference:+.6f} with 95% CI "
            f"[{context.walk_forward_ci_low:+.6f}, "
            f"{context.walk_forward_ci_high:+.6f}].",
        ),
        ("What We Can Claim", "The software pipeline passes. Current evidence does not establish QRW superiority."),
        ("Next Study", "Freeze protocol, collect 20+ fresh UTC days with synchronized LOB, then run one confirmatory evaluation."),
    )
    return "\n\n---\n\n".join(
        f"# {title}\n\n{body}" for title, body in slides
    )


def _wrapped_lines(
    paragraphs: Iterable[str],
    *,
    width: int = 92,
) -> list[str]:
    lines: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width))
        lines.append("")
    return lines


def _text_page(
    pdf: PdfPages,
    title: str,
    paragraphs: Sequence[str],
    *,
    page_number: int,
    subtitle: str | None = None,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.08, 0.94, title, fontsize=19, weight="bold", color="#16324F")
    if subtitle:
        fig.text(0.08, 0.905, subtitle, fontsize=10, color="#52677A")
    lines = _wrapped_lines(paragraphs)
    font_size = 10.2 if len(lines) < 48 else 8.8
    fig.text(
        0.08,
        0.87,
        "\n".join(lines),
        fontsize=font_size,
        va="top",
        ha="left",
        linespacing=1.35,
        color="#202A33",
    )
    fig.text(
        0.5,
        0.035,
        f"QRW cho vi cấu trúc thị trường | {page_number}",
        ha="center",
        fontsize=8,
        color="#687785",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _image_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.seek(0)
        return np.asarray(image.convert("RGB"))


def _figure_page(
    pdf: PdfPages,
    title: str,
    figure_path: Path,
    caption: str,
    *,
    page_number: int,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold", color="#16324F")
    axis = fig.add_axes((0.07, 0.22, 0.86, 0.64))
    axis.imshow(_image_array(figure_path))
    axis.axis("off")
    fig.text(
        0.09,
        0.16,
        "\n".join(textwrap.wrap(caption, width=100)),
        fontsize=9.5,
        va="top",
        color="#303A43",
    )
    fig.text(
        0.5,
        0.035,
        f"QRW cho vi cấu trúc thị trường | {page_number}",
        ha="center",
        fontsize=8,
        color="#687785",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _quantitative_page(
    pdf: PdfPages,
    context: ReportContext,
    *,
    page_number: int,
) -> None:
    fig, axis = plt.subplots(figsize=(8.27, 11.69))
    axis.axis("off")
    fig.text(
        0.08,
        0.94,
        "Bảng kết quả định lượng",
        fontsize=19,
        weight="bold",
        color="#16324F",
    )
    rows = (
        ("Mô hình đứng đầu", context.top_model),
        ("CRPS biên trung bình tốt nhất", f"{context.top_mean_marginal_crps:.6g}"),
        ("Hạng tổng thể của QRW", str(context.qrw_rank)),
        ("CRPS biên trung bình của QRW", f"{context.qrw_mean_marginal_crps:.6g}"),
        (
            "Log loss hướng của QRW",
            f"{context.qrw_mean_direction_log_loss:.6g}",
        ),
        ("Beta phương sai thực nghiệm", f"{context.empirical_beta:.4f}"),
        ("Beta phương sai QRW", f"{context.qrw_beta:.4f}"),
        (
            "Khoảng tin cậy 95% của beta QRW",
            f"[{context.qrw_beta_ci_low:.4f}, {context.qrw_beta_ci_high:.4f}]",
        ),
        ("Chỉ số phụ thuộc đường đi của QRW", "không áp dụng"),
        (
            "Số fold walk-forward QRW thắng",
            f"{context.walk_forward_folds_qrw_better}/{context.walk_forward_folds}",
        ),
        (
            "Brier gộp QRW - baseline",
            f"{context.walk_forward_brier_difference:+.6f}",
        ),
        (
            "Khoảng tin cậy 95% Brier walk-forward",
            f"[{context.walk_forward_ci_low:+.6f}, "
            f"{context.walk_forward_ci_high:+.6f}]",
        ),
        ("Baseline hướng đứng đầu", context.top_directional_model),
        ("Hạng hướng của QRW", str(context.qrw_directional_rank)),
        (
            "Log loss hướng của QRW",
            f"{context.qrw_directional_log_loss:.6g}",
        ),
    )
    table = axis.table(
        cellText=rows,
        colLabels=("Đại lượng", "Giá trị quan sát"),
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=(0.55, 0.32),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#B8C3CC")
        if row == 0:
            cell.set_facecolor("#DCEAF5")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F7F9")
    fig.text(
        0.08,
        0.12,
        "Mọi giá trị đều mang tính thăm dò và đến từ một cửa sổ ngày 12/06/2026.",
        fontsize=9,
        color="#4D5D6A",
    )
    fig.text(
        0.5,
        0.035,
        f"QRW cho vi cấu trúc thị trường | {page_number}",
        ha="center",
        fontsize=8,
        color="#687785",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_final_report_pdf(
    context: ReportContext,
    figures_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Render a self-contained 20-page PDF without external LaTeX tools."""
    figures = Path(figures_dir)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure_pages = (
        (
            "Tiến hóa xác suất",
            figures / "prob_evolution.gif",
            "Giao thoa và độ phân tán ballistic của QRW coherent tương phản với "
            "phân phối nhị thức khuếch tán cổ điển.",
        ),
        (
            "Quy luật co giãn phương sai",
            figures / "variance_scaling.png",
            "Độ dốc hồi quy tóm tắt cách phương sai biến động giá tăng theo "
            "horizon trên dữ liệu thực nghiệm và từng mô hình mô phỏng.",
        ),
        (
            "So sánh phân phối lợi suất",
            figures / "return_distributions.png",
            "Lợi suất được chuẩn hóa riêng cho từng chuỗi, nên phép so sánh tập "
            "trung vào hình dạng phân phối thay vì thang đo biến động.",
        ),
        (
            "So sánh tự tương quan",
            figures / "acf_comparison.png",
            "Chuỗi thực nghiệm có cấu trúc phụ thuộc mà không baseline mô phỏng "
            "đơn lẻ nào tái tạo hoàn toàn.",
        ),
        (
            "Các đường mẫu",
            figures / "sample_paths.png",
            "Việc chồng trực tiếp các đường đi cho thấy khác biệt về thang đo và "
            "hướng vốn bị nén trong các điểm tổng hợp.",
        ),
        (
            "Toán tử coin thích nghi",
            figures / "coin_operator_heatmap.png",
            "Toán tử giữ độ lớn cân bằng trong khi thông tin thị trường làm thay "
            "đổi các pha phức.",
        ),
        (
            "Bảng điểm benchmark",
            figures / "scorecard.png",
            f"{context.top_model} đứng đầu tổng thể. QRW xếp hạng "
            f"{context.qrw_rank} theo endpoint CRPS chính.",
        ),
    )
    for _title, path, _caption in figure_pages:
        if not path.exists():
            raise FileNotFoundError(path)

    with PdfPages(destination) as pdf:
        page = 1
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#F4F7FB")
        fig.text(
            0.08,
            0.72,
            "Quantum Random Walk\ncho vi cấu trúc thị trường",
            fontsize=28,
            weight="bold",
            color="#16324F",
            linespacing=1.25,
        )
        fig.text(
            0.08,
            0.60,
            "Báo cáo kỹ thuật Giai đoạn 6",
            fontsize=16,
            color="#2A6F97",
        )
        fig.text(
            0.08,
            0.50,
            "Dữ liệu sự kiện BTCUSDT | 12/06/2026\n"
            "Benchmark thăm dò có thể tái lập",
            fontsize=12,
            color="#475B6B",
            linespacing=1.6,
        )
        fig.text(
            0.08,
            0.12,
            "Kết luận kỹ thuật: pipeline hoàn chỉnh; chưa xác nhận QRW vượt trội.",
            fontsize=10,
            color="#7A2E2E",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        text_pages = (
            (
                "Tóm tắt",
                (
                    "Báo cáo đánh giá quantum random walk thời gian rời rạc được "
                    "hiệu chỉnh nhân quả cho vi cấu trúc thị trường ở horizon "
                    "ngắn. QRW Adaptive được so sánh với ba random walk cổ điển, "
                    "GARCH(1,1) và GBM trong cùng một benchmark theo thời gian.",
                    f"Mẫu đang dùng có {context.train_rows:,} hàng train và "
                    f"{context.holdout_rows:,} hàng holdout về sau. "
                    f"{context.top_model} đứng đầu, còn QRW xếp hạng "
                    f"{context.qrw_rank}.",
                    "Checkpoint phần mềm và artifact đạt yêu cầu, nhưng bằng "
                    "chứng từ một cửa sổ chưa xác lập ưu thế dự báo hay phân phối.",
                ),
            ),
            (
                "1. Động cơ và câu hỏi nghiên cứu",
                (
                    "Dữ liệu thị trường đến dưới dạng sự kiện rời rạc, phụ thuộc, "
                    "với thanh khoản thay đổi và đuôi nặng. QRW cung cấp giao thoa "
                    "và chuyển tiếp coherent-khuếch tán có kiểm soát mà random "
                    "walk cổ điển không có.",
                    "Câu hỏi thực nghiệm là các cơ chế bổ sung đó có cải thiện độ "
                    "khớp trong một so sánh thời gian công bằng hay không. Dự án "
                    "không tuyên bố cơ chế thị trường mang bản chất lượng tử.",
                ),
            ),
            (
                "2. Hình thức toán học QRW",
                (
                    "Phép toán coin tác động lên không gian hướng hai trạng thái, "
                    "sau đó là dịch chuyển có điều kiện trên lattice vị trí một "
                    "chiều. Xác suất vị trí được lấy bằng cách biên hóa bình "
                    "phương biên độ qua trạng thái coin.",
                    "Hadamard walk coherent phân tán theo ballistic. Khử pha theo "
                    "cơ sở làm giảm các phần tử ngoài đường chéo của density "
                    "matrix nhưng giữ trace và population, tạo chuyển tiếp về "
                    "khuếch tán cổ điển.",
                ),
            ),
            (
                "3. Ánh xạ sang thị trường",
                (
                    "Mất cân bằng sổ lệnh, hướng tick, thay đổi mất cân bằng và "
                    "trị tuyệt đối của mất cân bằng tạo thành tín hiệu hướng "
                    "thích nghi. Cường độ giao dịch điều biến khử pha theo sự kiện.",
                    "Toán tử thích nghi vẫn unitary. Tín hiệu hướng được mã hóa "
                    "trong pha, vì vậy bình phương độ lớn các phần tử coin vẫn cân bằng.",
                ),
            ),
            (
                "4. Dữ liệu và kiểm soát nhân quả",
                (
                    f"Artifact đang dùng: {context.feature_path}. Nghiên cứu cuối "
                    "chỉ dùng cửa sổ ngày 12/06/2026 đã làm sạch theo nhân quả.",
                    "Các artifact dẫn xuất từ ngày 01-07/06 trước đây đã bị vô "
                    "hiệu sau khi phát hiện look-ahead một tick trong bộ lọc "
                    "outlier. Chúng không được hỗ trợ kết luận cuối cho đến khi tái tạo.",
                    "Hiệu chỉnh tuân theo thời gian. Validation cấu trúc không được "
                    "đưa vào refit và cập nhật bias về sau không tái dùng warmup cấu trúc.",
                ),
            ),
            (
                "5. Benchmark và giao thức thống kê",
                (
                    f"Benchmark dùng {context.n_paths:,} mẫu biên cho mỗi mô hình "
                    f"và horizon, {context.n_steps} bước mô phỏng cùng seed "
                    f"{context.random_seed}.",
                    "Endpoint chính là CRPS biên fixed-origin; log loss hướng chỉ "
                    "dùng để phá hòa. Co giãn phương sai biên là chỉ số chẩn đoán.",
                    "ACF và kết quả tail chỉ áp dụng cho trajectory và loại QRW. "
                    "Diebold-Mariano dùng loss rolling một bước đã căn chỉnh. "
                    "AIC và BIC chỉ so sánh trong cùng họ likelihood.",
                ),
            ),
            (
                "6. Hiện thực các mô hình",
                (
                    "QRW Adaptive dùng coin unitary thích nghi theo pha và khử pha "
                    "theo cơ sở phụ thuộc cường độ. Baseline cổ điển gồm random "
                    "walk đơn giản, thiên lệch và tương quan.",
                    "GARCH(1,1) cung cấp cụm biến động có điều kiện, còn GBM là "
                    "baseline khuếch tán lognormal liên tục. Mọi mô hình dùng cùng "
                    "mốc cắt train và cùng horizon test.",
                ),
            ),
        )
        for title, paragraphs in text_pages:
            page += 1
            _text_page(pdf, title, paragraphs, page_number=page)

        for title, path, caption in figure_pages:
            page += 1
            _figure_page(
                pdf,
                title,
                path,
                caption,
                page_number=page,
            )

        page += 1
        _quantitative_page(pdf, context, page_number=page)

        closing_pages = (
            (
                "7. Thảo luận",
                (
                    f"{context.top_model} có CRPS biên chính thấp nhất. QRW xếp "
                    f"hạng {context.qrw_rank}; QRW không được gán điểm phụ thuộc "
                    "đường đi.",
                    "Giới hạn này là có chủ đích: các marginal độc lập theo horizon "
                    "không xác định lợi suất, ACF hay một mẫu tail.",
                    f"Kết quả Giai đoạn 3 là âm: QRW thắng "
                    f"{context.walk_forward_folds_qrw_better}/"
                    f"{context.walk_forward_folds} fold; chênh lệch Brier gộp là "
                    f"{context.walk_forward_brier_difference:+.6f}.",
                ),
            ),
            (
                "8. Hạn chế",
                (
                    "Nghiên cứu dùng một cửa sổ sự kiện ngắn và có ít dữ liệu sổ "
                    "lệnh thực được đồng bộ.",
                    "Hiệu chỉnh thích nghi bị giới hạn bởi dữ liệu. Scorecard dùng "
                    "CRPS làm endpoint chính duy nhất và không lấy trung bình hạng "
                    "qua các chỉ số chẩn đoán phụ thuộc nhau.",
                    "Chưa có artifact nào là holdout confirmatory mới và chưa từng "
                    "đụng tới. Dashboard cùng trực quan hóa chỉ là giao diện thăm "
                    "dò, không phải bằng chứng bổ sung.",
                ),
            ),
            (
                "9. Kết luận và hướng phát triển",
                (
                    "Giai đoạn 6 hoàn thiện lớp trực quan hóa và báo cáo có thể tái "
                    "lập. Kết luận có thể bảo vệ là phần kỹ thuật đạt, nhưng chưa "
                    "xác nhận QRW vượt trội.",
                    "Nghiên cứu tiếp theo cần đóng băng protocol trước khi quan sát "
                    "nhãn mới, thu thập ít nhất 20 ngày UTC mới với dữ liệu LOB đồng "
                    "bộ và chỉ chạy một phân tích confirmatory.",
                    "Hướng mô hình tiếp theo có thể nghiên cứu learned coin được "
                    "regularize, QRW đa tài sản hai chiều và walk thời gian liên tục "
                    "so với baseline point-process cổ điển linh hoạt tương đương.",
                ),
            ),
            (
                "Tài liệu tham khảo",
                REFERENCES,
            ),
        )
        for title, paragraphs in closing_pages:
            page += 1
            _text_page(pdf, title, paragraphs, page_number=page)
    return destination


def render_presentation_pdf(
    context: ReportContext,
    figures_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Render a 16-slide seminar presentation."""
    figures = Path(figures_dir)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    slides: tuple[tuple[str, str, str | None], ...] = (
        (
            "Quantum Random Walks for Market Microstructure",
            "Phase 6 final presentation\nBTCUSDT, June 12, 2026",
            None,
        ),
        (
            "Research Question",
            "Can a causally calibrated QRW reproduce short-horizon market "
            "properties better than classical baselines?",
            None,
        ),
        (
            "Why QRW?",
            "Interference, ballistic spreading, and a tunable "
            "coherent-to-diffusive transition.",
            None,
        ),
        (
            "QRW vs CRW",
            "The coherent walk spreads with interference peaks; the classical "
            "walk follows a diffusive binomial envelope.",
            "prob_evolution.gif",
        ),
        (
            "Active Data",
            f"{context.train_rows:,} training rows\n"
            f"{context.holdout_rows:,} later holdout rows\n"
            "Historical June 1-7 derived artifacts remain invalidated.",
            None,
        ),
        (
            "Causal Pipeline",
            "Trailing cleaner -> chronological split -> disjoint validation "
            "-> fixed seed -> matched statistical samples",
            None,
        ),
        (
            "Market Mapping",
            "OBI and direction enter a unitary phase-adaptive coin. Trade "
            "intensity controls basis dephasing.",
            "coin_operator_heatmap.png",
        ),
        (
            "Benchmark Models",
            "QRW Adaptive\nCRW Simple / Biased / Correlated\nGARCH(1,1)\nGBM",
            None,
        ),
        (
            "Evaluation",
            "Marginal CRPS\nDirection log loss\nMarginal variance scaling\n"
            "Rolling one-step DM",
            None,
        ),
        (
            "Variance Scaling",
            f"QRW beta = {context.qrw_beta:.4f} "
            f"[{context.qrw_beta_ci_low:.4f}, "
            f"{context.qrw_beta_ci_high:.4f}]",
            "variance_scaling.png",
        ),
        (
            "Distribution Shape",
            "Trajectory-only diagnostic; QRW marginal draws are excluded.",
            "return_distributions.png",
        ),
        (
            "Autocorrelation",
            "Trajectory-only diagnostic; QRW marginal draws are excluded.",
            "acf_comparison.png",
        ),
        (
            "Sample Paths",
            "QRW is shown as a marginal interval; only classical trajectory "
            "models are drawn as paths.",
            "sample_paths.png",
        ),
        (
            "Scorecard",
            f"{context.top_model} ranks first.\n"
            f"QRW Adaptive ranks {context.qrw_rank}.",
            "scorecard.png",
        ),
        (
            "Walk-Forward Result",
            f"QRW wins {context.walk_forward_folds_qrw_better}/"
            f"{context.walk_forward_folds} folds.\n"
            f"Brier difference = {context.walk_forward_brier_difference:+.6f}\n"
            f"95% CI [{context.walk_forward_ci_low:+.6f}, "
            f"{context.walk_forward_ci_high:+.6f}]",
            None,
        ),
        (
            "What We Can Claim",
            "The engineering and reporting pipeline passes.\n\n"
            "Current evidence does not establish QRW predictive or "
            "distributional superiority.",
            None,
        ),
        (
            "Next Confirmatory Study",
            "Freeze the protocol, collect at least 20 fresh UTC days with "
            "synchronized LOB data, then execute one untouched evaluation.",
            None,
        ),
    )
    with PdfPages(destination) as pdf:
        for index, (title, body, image_name) in enumerate(slides, start=1):
            fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
            fig.text(
                0.055,
                0.89,
                title,
                fontsize=24,
                weight="bold",
                color="#16324F",
            )
            if image_name:
                path = figures / image_name
                if not path.exists():
                    raise FileNotFoundError(path)
                axis = fig.add_axes((0.07, 0.15, 0.58, 0.65))
                axis.imshow(_image_array(path))
                axis.axis("off")
                fig.text(
                    0.70,
                    0.70,
                    "\n".join(textwrap.wrap(body, width=35)),
                    fontsize=16,
                    va="top",
                    linespacing=1.5,
                    color="#263744",
                )
            else:
                fig.text(
                    0.09,
                    0.67,
                    "\n".join(textwrap.wrap(body, width=64)),
                    fontsize=21,
                    va="top",
                    linespacing=1.55,
                    color="#263744",
                )
            fig.text(
                0.94,
                0.045,
                str(index),
                ha="right",
                fontsize=9,
                color="#71808C",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return destination


def count_pdf_pages(path: str | Path) -> int:
    """Count page objects in PDFs emitted by Matplotlib."""
    content = Path(path).read_bytes()
    return len(re.findall(rb"/Type\s*/Page(?!s)\b", content))
