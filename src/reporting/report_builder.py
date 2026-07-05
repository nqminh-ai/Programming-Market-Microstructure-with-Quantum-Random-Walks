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


def _build_final_report_markdown_legacy(context: ReportContext) -> str:
    """Retain the prior layout for comparison; it is not release output."""
    references = "\n".join(
        f"{index}. {reference}"
        for index, reference in enumerate(REFERENCES, start=1)
    )
    return f"""# Quantum Random Walks for Market Microstructure

## Abstract

This project evaluates whether a discrete-time quantum random walk (QRW) can
serve as a useful engineering model for short-horizon market microstructure.
The implementation maps order-book imbalance, tick direction, imbalance
changes, and trade intensity into a unitary adaptive coin and an
intensity-dependent dephasing channel. It compares the resulting QRW against
simple, biased, and correlated classical random walks, GARCH(1,1), and
geometric Brownian motion under one chronological train/holdout protocol.
The active dataset contains {context.train_rows + context.holdout_rows:,}
causally processed BTCUSDT observations from June 12, 2026. Phase 5 uses
{context.train_rows:,} training rows, {context.holdout_rows:,} later holdout
rows, {context.n_paths:,} marginal draws per model and horizon, and a fixed
random seed of {context.random_seed}. The pre-registered primary endpoint is
mean fixed-origin marginal CRPS; directional log loss is used only as a
tie-break. {context.top_model} has the lowest primary score
({context.top_mean_marginal_crps:.6g}); QRW Adaptive ranks {context.qrw_rank}
with CRPS {context.qrw_mean_marginal_crps:.6g} and directional log loss
{context.qrw_mean_direction_log_loss:.6g}. QRW horizon samples are independent
marginals, so no return differences, ACF, tail index, or path-based
Diebold-Mariano statistic is computed from them. These results validate a
reproducible software pipeline but do not establish QRW predictive
superiority. Confirmation requires a frozen protocol and fresh multi-day,
synchronized limit-order-book holdout data.

## 1. Introduction

Market microstructure is driven by discrete event arrivals, persistent order
flow, changing liquidity, and heavy-tailed price changes. Classical random
walks provide a transparent baseline but cannot express interference or a
controlled transition from coherent to diffusive dynamics. QRWs supply those
mechanisms through a coin state, conditional shift, and decoherence channel.

The research question is deliberately modest: can a causally calibrated QRW
reproduce selected empirical distributional, scaling, dependence, and tail
properties better than standard baselines? The project treats this as an
engineering and falsification exercise, not as evidence that markets are
quantum mechanical.

## 2. Theoretical Framework

A one-dimensional coined QRW evolves on
`H_coin tensor H_position`. One step applies a unitary coin `C_t`, followed by
a conditional shift `S`, so `|psi_(t+1)> = S(C_t tensor I)|psi_t>`. The
position probability is obtained by summing squared coin amplitudes. For a
symmetric coherent Hadamard walk, variance grows ballistically, approximately
as `t^2`; a symmetric classical walk grows diffusively as `t`.

The mixed-state implementation evolves `rho` and applies basis dephasing after
each unitary step. Off-diagonal entries are multiplied by `exp(-gamma_t)`,
preserving trace and populations. As coherence is reduced, the walk approaches
classical diffusion.

## 3. Market Mapping

The adaptive coin uses a bounded nonlinear signal from current order-book
imbalance, tick direction, imbalance change, and absolute imbalance. Market
activity modifies the event-level dephasing rate. The operator remains unitary;
directional information is phase encoded, so the magnitude-squared coin matrix
alone is intentionally symmetric.

This mapping is causal at each event. Calibration is chronological, structural
selection excludes its validation segment from refitting, and the later bias
update does not reuse structural warmup observations.

## 4. Data and Methodology

Active feature artifact: `{context.feature_path}`.

| Protocol item | Value |
|---|---:|
| Chronological training rows | {context.train_rows:,} |
| Later holdout rows | {context.holdout_rows:,} |
| Simulation steps | {context.n_steps:,} |
| Marginal draws per model/horizon | {context.n_paths:,} |
| Random seed | {context.random_seed} |

The active June 12 dataset is the only window used for the final benchmark.
Historical June 1-7 processed and feature artifacts were rebuilt with the
causal pipeline, but remain development-only rather than confirmatory data.
QRW is evaluated with proper fixed-origin marginal scores and marginal
variance scaling. ACF and tail diagnostics are explicitly trajectory-only and
exclude QRW. Diebold-Mariano comparisons use aligned rolling-origin one-step
losses rather than columns from a fixed-origin forecast.

## 5. Model Implementations

The benchmark includes QRW Adaptive, CRW Simple, CRW Biased, CRW Correlated,
GARCH(1,1), and GBM. All models receive the same chronological split and
simulation horizon. Directional likelihoods are compared only within the
Bernoulli family; continuous Gaussian likelihood AIC/BIC values are not ranked
against directional likelihoods.

## 6. Results

### 6.1 Probability Evolution

![QRW and CRW probability evolution](../figures/prob_evolution.gif)

The coherent QRW spreads ballistically with interference peaks, whereas the
symmetric CRW remains concentrated around a diffusive binomial envelope.

### 6.2 Variance Scaling

![Variance scaling](../figures/variance_scaling.png)

The empirical fitted exponent is {context.empirical_beta:.4f}. QRW Adaptive
has beta {context.qrw_beta:.4f} with 95% interval
[{context.qrw_beta_ci_low:.4f}, {context.qrw_beta_ci_high:.4f}].

### 6.3 Trajectory-only Return Distributions

![Return distributions](../figures/return_distributions.png)

This diagnostic excludes QRW because its independently sampled horizon
marginals are not a return trajectory.

### 6.4 Autocorrelation

![ACF comparison](../figures/acf_comparison.png)

This diagnostic includes only empirical data and classical models with jointly
sampled trajectories. It is not part of the QRW scorecard.

### 6.5 Sample Paths

![Sample paths](../figures/sample_paths.png)

The QRW layer is a 90% marginal interval and median at each horizon, not a set
of joined paths. Classical models retain genuine sample trajectories.

### 6.6 Adaptive Coin

![Adaptive coin heatmap](../figures/coin_operator_heatmap.png)

The magnitude-squared entries remain balanced while the complex phase changes
with the market signal. This distinction is required to interpret the
adaptive operator correctly.

### 6.7 Scorecard

![Benchmark scorecard](../figures/scorecard.png)

| Quantity | Observed value |
|---|---:|
| Top-ranked model | {context.top_model} |
| Top mean marginal CRPS | {context.top_mean_marginal_crps:.6g} |
| QRW overall rank | {context.qrw_rank} |
| QRW mean marginal CRPS | {context.qrw_mean_marginal_crps:.6g} |
| QRW direction log loss | {context.qrw_mean_direction_log_loss:.6g} |
| QRW variance beta | {context.qrw_beta:.4f} |
| QRW beta 95% interval | [{context.qrw_beta_ci_low:.4f}, {context.qrw_beta_ci_high:.4f}] |

## 7. Discussion

The primary CRPS scorecard and marginal variance scaling answer narrower
questions than path diagnostics. QRW is not assigned an ACF or tail score,
because doing so would require inventing temporal coupling between independent
horizon draws. The current primary endpoint does not support a general QRW
advantage.

The Phase 3 predictive audit is mixed after the causal rebuild: QRW beats the
fair affine baseline on the final holdout but loses in pooled walk-forward
evaluation. That instability and the Phase 5 scorecard answer different
questions, but neither supports a current general superiority claim.

## 8. Limitations

1. The benchmark uses one short June 12, 2026 event window.
2. Structural QRW calibration has a low observations-per-parameter ratio.
3. Rebuilt June 1-7 data remain development-only after prior inspection.
4. Real synchronized limit-order-book coverage is limited.
5. The scorecard uses CRPS as its primary endpoint; other metrics are diagnostic.
6. Fixed tick moves cannot reproduce empirical jump size or heavy tails.
7. The optional dashboard is exploratory and does not alter inference.

## 9. Conclusion and Future Work

The project completes a reproducible QRW market-microstructure prototype,
classical benchmarks, statistical tests, visualization suite, and reporting
pipeline. The defensible conclusion is an engineering pass with no validated
QRW predictive or distributional superiority.

Future work should: freeze a confirmatory protocol before new labels are
observed; collect at least 20 fresh UTC days with synchronized LOB data; add a
learned but regularized coin benchmarked against equally flexible classical
links; study two-dimensional QRWs for multi-asset state spaces; and compare
continuous-time quantum walks with marked point-process baselines.

## References

{references}
"""


def build_final_report_markdown(context: ReportContext) -> str:
    """Trả về báo cáo tiếng Việt, giữ nguyên kết luận khoa học âm."""
    return f"""# Báo cáo cuối: Quantum Random Walk cho vi cấu trúc thị trường

## Tóm tắt

Nghiên cứu đánh giá một quantum random walk (QRW) density-matrix với coin
thích nghi và decoherence. Tập đánh giá có {context.train_rows:,} dòng train,
{context.holdout_rows:,} dòng holdout, {context.n_steps:,} horizon và
{context.n_paths:,} draw biên cho mỗi model/horizon. Endpoint chính là mean
marginal CRPS; directional log loss chỉ dùng làm tie-break.

{context.top_model} có CRPS thấp nhất ({context.top_mean_marginal_crps:.6g}).
QRW xếp hạng {context.qrw_rank}, CRPS {context.qrw_mean_marginal_crps:.6g} và
directional log loss {context.qrw_mean_direction_log_loss:.6g}. Kết quả này
không thiết lập bằng chứng về ưu thế dự báo của QRW.

## 1. Ngữ nghĩa đánh giá

QRW chỉ tạo phân phối biên fixed-origin tại từng horizon. Các draw độc lập giữa
các horizon không tạo thành trajectory. Vì vậy QRW không được đưa vào thống kê
return difference, ACF, tail index hoặc path-based Diebold–Mariano.

Diebold–Mariano chỉ dùng loss rolling-origin một bước, căn chỉnh trên cùng
timestamp. ACF và tail là chẩn đoán `trajectory_only` cho baseline có path
thật.

## 2. Dữ liệu và giao thức

Artifact feature: `{context.feature_path}`.

| Thành phần | Giá trị |
|---|---:|
| Train | {context.train_rows:,} dòng |
| Holdout | {context.holdout_rows:,} dòng |
| Horizon | {context.n_steps:,} |
| Draw biên/model/horizon | {context.n_paths:,} |
| Seed | {context.random_seed} |

OBI trong dữ liệu hiện tại là proxy trade-flow imbalance, không phải L2 LOB.
Calibration và mô hình AR(1) cho OBI tương lai chỉ dùng train.

## 3. Kết quả định lượng

| Đại lượng | Giá trị quan sát |
|---|---:|
| Model đứng đầu | {context.top_model} |
| CRPS đứng đầu | {context.top_mean_marginal_crps:.6g} |
| QRW overall rank | {context.qrw_rank} |
| QRW mean marginal CRPS | {context.qrw_mean_marginal_crps:.6g} |
| QRW direction log loss | {context.qrw_mean_direction_log_loss:.6g} |
| Empirical variance beta | {context.empirical_beta:.4f} |
| QRW variance beta | {context.qrw_beta:.4f} |
| QRW beta 95% CI | [{context.qrw_beta_ci_low:.4f}; {context.qrw_beta_ci_high:.4f}] |
| Thống kê path-dependent của QRW | không áp dụng |

![Tiến hóa xác suất](../figures/prob_evolution.gif)

![Variance scaling](../figures/variance_scaling.png)

![Dải marginal QRW và path baseline](../figures/sample_paths.png)

![Scorecard](../figures/scorecard.png)

## 4. Giới hạn

1. Dữ liệu hoạt động chỉ gồm một cửa sổ rất ngắn.
2. OBI là proxy luồng giao dịch, không phải order book L2 đồng bộ.
3. Kết quả cross-asset cũ chưa đánh giá đầy đủ mọi ngày có sẵn.
4. Walk-forward hiện không ủng hộ lợi thế QRW.
5. Chưa có holdout multi-day untouched để đưa ra kết luận confirmatory.

## 5. Kết luận

Pipeline có thể kiểm toán ngữ nghĩa marginal, nhưng kết quả khoa học hiện tại
vẫn âm: chưa có bằng chứng QRW vượt baseline cổ điển. Kết luận chỉ được xem xét
lại sau khi protocol và provenance được đóng băng, đồng thời có ít nhất 20 ngày
UTC mới cho mỗi tài sản với L2 LOB đồng bộ.
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
        f"QRW Market Microstructure | {page_number}",
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
        f"QRW Market Microstructure | {page_number}",
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
        "Quantitative Result Table",
        fontsize=19,
        weight="bold",
        color="#16324F",
    )
    rows = (
        ("Top model", context.top_model),
        ("Top mean marginal CRPS", f"{context.top_mean_marginal_crps:.6g}"),
        ("QRW overall rank", str(context.qrw_rank)),
        ("QRW mean marginal CRPS", f"{context.qrw_mean_marginal_crps:.6g}"),
        (
            "QRW direction log loss",
            f"{context.qrw_mean_direction_log_loss:.6g}",
        ),
        ("Empirical variance beta", f"{context.empirical_beta:.4f}"),
        ("QRW variance beta", f"{context.qrw_beta:.4f}"),
        (
            "QRW beta 95% CI",
            f"[{context.qrw_beta_ci_low:.4f}, {context.qrw_beta_ci_high:.4f}]",
        ),
        ("QRW path-dependent metrics", "not applicable"),
    )
    table = axis.table(
        cellText=rows,
        colLabels=("Quantity", "Observed value"),
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
        "All values are exploratory and come from one June 12, 2026 window.",
        fontsize=9,
        color="#4D5D6A",
    )
    fig.text(
        0.5,
        0.035,
        f"QRW Market Microstructure | {page_number}",
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
            "Probability Evolution",
            figures / "prob_evolution.gif",
            "Coherent QRW interference and ballistic spread contrast with the "
            "diffusive classical binomial distribution.",
        ),
        (
            "Variance Scaling",
            figures / "variance_scaling.png",
            "Regression slopes summarize how price-change variance grows with "
            "horizon for the empirical path and every simulated model.",
        ),
        (
            "Return Distribution Comparison",
            figures / "return_distributions.png",
            "Returns are standardized per series, so the comparison concerns "
            "distributional shape rather than volatility scale.",
        ),
        (
            "Autocorrelation Comparison",
            figures / "acf_comparison.png",
            "The empirical series has dependence that no single simulated "
            "baseline reproduces completely.",
        ),
        (
            "Sample Paths",
            figures / "sample_paths.png",
            "Direct path overlays reveal scale and directional differences "
            "that are compressed by aggregate scores.",
        ),
        (
            "Adaptive Coin Operator",
            figures / "coin_operator_heatmap.png",
            "The operator keeps balanced magnitudes while market information "
            "changes complex phases.",
        ),
        (
            "Benchmark Scorecard",
            figures / "scorecard.png",
            f"{context.top_model} is first overall. QRW is rank "
            f"{context.qrw_rank} under the primary CRPS endpoint.",
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
            "Quantum Random Walks\nfor Market Microstructure",
            fontsize=28,
            weight="bold",
            color="#16324F",
            linespacing=1.25,
        )
        fig.text(
            0.08,
            0.60,
            "Phase 6 Technical Report",
            fontsize=16,
            color="#2A6F97",
        )
        fig.text(
            0.08,
            0.50,
            "BTCUSDT event data | June 12, 2026\n"
            "Reproducible exploratory benchmark",
            fontsize=12,
            color="#475B6B",
            linespacing=1.6,
        )
        fig.text(
            0.08,
            0.12,
            "Engineering conclusion: pipeline complete; no validated QRW "
            "superiority.",
            fontsize=10,
            color="#7A2E2E",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        text_pages = (
            (
                "Abstract",
                (
                    "This report evaluates a causally calibrated discrete-time "
                    "quantum random walk for short-horizon market "
                    "microstructure. It compares QRW Adaptive with three "
                    "classical random walks, GARCH(1,1), and GBM under one "
                    "chronological benchmark.",
                    f"The active sample has {context.train_rows:,} training "
                    f"rows and {context.holdout_rows:,} later holdout rows. "
                    f"{context.top_model} ranks first, while QRW ranks "
                    f"{context.qrw_rank}.",
                    "The software and artifact checkpoint passes, but the "
                    "single-window evidence does not establish predictive or "
                    "distributional superiority.",
                ),
            ),
            (
                "1. Motivation and Research Question",
                (
                    "Market data arrive as discrete, dependent events with "
                    "changing liquidity and heavy tails. A QRW offers "
                    "interference and a controlled coherent-to-diffusive "
                    "transition that classical random walks do not possess.",
                    "The empirical question is whether those extra mechanisms "
                    "improve fit under a fair chronological comparison. The "
                    "project does not claim that market mechanics are quantum.",
                ),
            ),
            (
                "2. QRW Formalism",
                (
                    "A coin operation acts on a two-state direction space, "
                    "followed by a conditional shift on a one-dimensional "
                    "position lattice. Position probabilities are obtained by "
                    "marginalizing squared amplitudes over the coin state.",
                    "The coherent Hadamard walk spreads ballistically. Basis "
                    "dephasing damps off-diagonal density-matrix entries while "
                    "preserving trace and populations, producing a transition "
                    "toward classical diffusion.",
                ),
            ),
            (
                "3. Market Mapping",
                (
                    "Order-book imbalance, tick direction, imbalance change, "
                    "and absolute imbalance form the adaptive directional "
                    "signal. Trade intensity modulates event-level dephasing.",
                    "The adaptive operator remains unitary. Its directional "
                    "signal is encoded in phase, which is why magnitude-squared "
                    "coin entries remain balanced.",
                ),
            ),
            (
                "4. Data and Causal Controls",
                (
                    f"Active artifact: {context.feature_path}. The final study "
                    "uses only the causal-cleaned June 12, 2026 window.",
                    "Historical June 1-7 derived artifacts were invalidated "
                    "after detection of a one-tick look-ahead in the outlier "
                    "filter. They may not support final conclusions until "
                    "rebuilt.",
                    "Calibration is chronological. Structural validation is "
                    "not included in refitting, and the later bias update does "
                    "not reuse the structural warmup.",
                ),
            ),
            (
                "5. Benchmark and Statistical Protocol",
                (
                    f"The benchmark uses {context.n_paths:,} marginal draws "
                    "per model and horizon, "
                    f"{context.n_steps} simulation steps, and random seed "
                    f"{context.random_seed}.",
                    "The primary endpoint is fixed-origin marginal CRPS; "
                    "directional log loss is a tie-break. Marginal variance "
                    "scaling is diagnostic.",
                    "ACF and tail outputs are trajectory-only and exclude QRW. "
                    "Diebold-Mariano uses aligned rolling one-step losses. "
                    "AIC and BIC remain within matching likelihood families.",
                ),
            ),
            (
                "6. Model Implementations",
                (
                    "QRW Adaptive uses a unitary phase-adaptive coin and "
                    "intensity-dependent basis dephasing. Classical baselines "
                    "include simple, biased, and correlated random walks.",
                    "GARCH(1,1) supplies conditional volatility clustering and "
                    "GBM supplies a continuous lognormal diffusion baseline. "
                    "All models use the same training cutoff and test horizon.",
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
                "7. Discussion",
                (
                    f"{context.top_model} has the lowest primary marginal CRPS. "
                    f"QRW ranks {context.qrw_rank}; no path-dependent score is "
                    "assigned to QRW.",
                    "This restriction is intentional: independent horizon "
                    "marginals do not define returns, ACF, or a tail sample.",
                    "Phase 3 is mixed: QRW wins the final holdout but loses the "
                    "pooled walk-forward comparison against a fair affine "
                    "baseline.",
                ),
            ),
            (
                "8. Limitations",
                (
                    "The study uses one short event window and limited real "
                    "synchronized order-book coverage.",
                    "The adaptive calibration is data constrained, and the "
                    "rank scorecard compresses dependent metrics to an equal "
                    "weight average.",
                    "No current artifact is a fresh untouched confirmatory "
                    "holdout. The dashboard and visualizations are exploratory "
                    "interfaces, not additional evidence.",
                ),
            ),
            (
                "9. Conclusion and Future Work",
                (
                    "Phase 6 completes the reproducible visualization and "
                    "reporting layer. The defensible conclusion is an "
                    "engineering pass with no validated QRW superiority.",
                    "Next work should freeze the protocol before observing new "
                    "labels, collect at least 20 fresh UTC days with "
                    "synchronized LOB data, and run one confirmatory analysis.",
                    "Further model work may study regularized learned coins, "
                    "two-dimensional multi-asset QRWs, and continuous-time "
                    "walks against equally flexible classical point-process "
                    "baselines.",
                ),
            ),
            (
                "References",
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
