"""The research result, as a picture.

The dashboard had five tabs of demo tooling and one tab of prose. The finding
the project actually spent its time on -- that no forecast horizon can pay its
own trading costs -- had no visual anywhere, and it is the part a visitor
without a finance background can grasp fastest: two dots per row, and they
never meet.

Everything here is read out of ``reports/research/horizon_edge_*.json`` rather
than restated. Those files are regenerated whenever a study is re-run; prose is
not, and this dashboard has already carried a figure that a rerun had moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.dashboard.design_system import COLORS, PLOTLY_TEMPLATE

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "research"

ASSETS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
FEE_SCENARIO = "maker_futures_2bps"

# Two categorical slots, validated against this dashboard's plot surface
# (#0D1420) with the dataviz palette checker: lightness band, chroma floor,
# CVD separation (worst adjacent dE 26.8 protan), normal-vision floor (31.8)
# and contrast all pass. Not the accent red/green -- those are status tokens
# here and mean good/bad, which is not what these two series are.
ACHIEVED_COLOR = "#3987e5"
REQUIRED_COLOR = "#d95926"


@dataclass(frozen=True)
class HorizonRow:
    """One asset at one horizon: what was reached against what was needed."""

    asset: str
    horizon: int
    seconds: float | None
    achieved: float
    required: float
    n_test: int
    net_edge_bps: float

    @property
    def shortfall(self) -> float:
        """Percentage points still missing. Positive means not viable."""
        return self.required - self.achieved

    @property
    def label(self) -> str:
        return f"{self.asset[:3]} · {self.horizon:,}".replace(",", ".")

    @property
    def duration(self) -> str:
        if self.seconds is None:
            return "—"
        if self.seconds < 90:
            return f"{self.seconds:.0f} giây"
        if self.seconds < 5400:
            return f"{self.seconds / 60:.0f} phút"
        return f"{self.seconds / 3600:.1f} giờ"


def load_horizon_rows(reports_dir: Path | None = None) -> list[HorizonRow]:
    """Read the edge studies. Returns [] when they have not been generated."""
    directory = reports_dir or REPORTS
    rows: list[HorizonRow] = []
    for asset in ASSETS:
        path = directory / f"horizon_edge_{asset}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in payload.get("analysis", {}).get("horizons", []):
            if "skipped" in entry:
                continue
            required = entry.get("breakeven_accuracy", {}).get(FEE_SCENARIO)
            if required is None:
                continue
            rows.append(
                HorizonRow(
                    asset=asset,
                    horizon=int(entry["horizon_ticks"]),
                    seconds=entry.get("seconds"),
                    achieved=float(entry["best_accuracy"]) * 100.0,
                    required=float(required) * 100.0,
                    n_test=int(entry.get("n_test", 0)),
                    net_edge_bps=float(
                        entry.get("net_edge_per_trade", {}).get(FEE_SCENARIO, 0.0)
                    )
                    * 1e4,
                )
            )
    return rows


def build_gap_figure(rows: list[HorizonRow]):
    """A dumbbell per asset-horizon: reached, needed, and the gap between.

    Horizontal so the 164% BTC row does not compress everything else, which a
    shared vertical scale would. One axis, one unit -- both series are hit
    rates in percent.
    """
    import plotly.graph_objects as go

    # Reading order, top to bottom. The y axis is reversed below so that index
    # 0 lands at the top rather than at the origin.
    ordered = sorted(rows, key=lambda r: (ASSETS.index(r.asset), r.horizon))
    labels = [row.label for row in ordered]

    figure = go.Figure()

    # Connectors first so the markers sit on top of them.
    for index, row in enumerate(ordered):
        figure.add_shape(
            type="line",
            x0=row.achieved,
            x1=row.required,
            y0=index,
            y1=index,
            line=dict(color=COLORS["border_active"], width=2),
            layer="below",
        )

    figure.add_trace(
        go.Scatter(
            x=[row.achieved for row in ordered],
            y=list(range(len(ordered))),
            mode="markers",
            name="Đạt được",
            marker=dict(
                size=11,
                color=ACHIEVED_COLOR,
                # 2px surface ring, so overlapping marks stay separable.
                line=dict(color=COLORS["bg_secondary"], width=2),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Đạt được: %{x:.1f}%<br>"
                "Cửa sổ kiểm định: %{customdata[1]}<extra></extra>"
            ),
            customdata=[[row.label, f"{row.n_test:,}".replace(",", ".")] for row in ordered],
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[row.required for row in ordered],
            y=list(range(len(ordered))),
            mode="markers",
            name="Cần có để hoà vốn",
            marker=dict(
                size=11,
                color=REQUIRED_COLOR,
                symbol="diamond",
                line=dict(color=COLORS["bg_secondary"], width=2),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Cần có: %{x:.1f}%<br>"
                "Còn thiếu: %{customdata[1]:.1f} điểm<extra></extra>"
            ),
            customdata=[[row.label, row.shortfall] for row in ordered],
        )
    )

    # Two references a lay reader already understands. Labelled along the top;
    # the legend is moved below the plot so the two cannot collide, and the
    # bottom band is left to the axis ticks alone.
    for value, text in ((50.0, "Tung đồng xu"), (100.0, "Đúng mọi lần")):
        figure.add_vline(
            x=value,
            line=dict(color=COLORS["text_muted"], width=1),
            annotation_text=text,
            annotation_position="top",
            annotation_font=dict(size=10, color=COLORS["text_muted"]),
        )

    # Direct-label the single most extreme row, not every point. Placed below
    # its marker rather than beside it: at 164% the marker is near the right
    # edge, and a label to its right would overflow a narrow container.
    worst = max(ordered, key=lambda r: r.shortfall)
    figure.add_annotation(
        x=worst.required,
        y=ordered.index(worst),
        text=f"thiếu {worst.shortfall:.0f} điểm",
        showarrow=False,
        xanchor="right",
        yshift=-17,
        font=dict(size=11, color=COLORS["text_secondary"]),
    )

    # Room for the row labels on the left, the reference labels on top, and
    # the axis title plus legend stacked underneath.
    layout = dict(PLOTLY_TEMPLATE)
    layout["margin"] = dict(l=118, r=44, t=44, b=78)

    figure.update_layout(
        **layout,
        height=118 + 34 * len(ordered),
        xaxis=dict(
            title="Tỉ lệ đoán đúng (%)",
            gridcolor=COLORS["border_subtle"],
            zeroline=False,
            griddash="solid",
            # Room to the left of the 50% rule so it reads as a reference
            # inside the plot rather than as the axis itself.
            range=[44, max(row.required for row in ordered) + 12],
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(ordered))),
            ticktext=labels,
            gridcolor="rgba(0,0,0,0)",
            zeroline=False,
            autorange="reversed",
        ),
        legend=dict(orientation="h", y=-0.16, x=0, yanchor="top"),
        hovermode="closest",
    )
    return figure


FINDING_HEADLINE = (
    "Không một khoảng thời gian dự báo nào, trên bất kỳ đồng tiền nào, "
    "kiếm đủ để trả phí giao dịch của chính nó."
)

FINDING_READING = [
    "Mỗi dòng là một **khoảng thời gian dự báo** trên một đồng tiền — ví dụ "
    "`BTC · 1.000` nghĩa là dự báo giá sau 1.000 giao dịch của Bitcoin.",
    "Chấm **xanh** là tỉ lệ đoán đúng mô hình tốt nhất **thực sự đạt được**.",
    "Hình thoi **cam** là tỉ lệ đoán đúng **cần có** để hoà vốn sau phí.",
    "Khoảng cách giữa hai điểm là phần còn thiếu. **Chưa dòng nào khép lại "
    "được khoảng đó**, và ở vài dòng thì mốc cần có còn nằm bên phải vạch "
    "“đúng mọi lần” — tức đoán đúng 100% vẫn lỗ.",
]


def render_findings_tab() -> None:
    """The one tab that shows a result rather than a demo."""
    import streamlit as st

    rows = load_horizon_rows()
    if not rows:
        st.info(
            "Chưa có kết quả nghiên cứu để hiển thị. Chạy "
            "`python -m scripts.research.horizon_label_baselines` để sinh ra."
        )
        return

    st.markdown(
        f"""
        <div class="kpi-card" style="border-left:3px solid {COLORS['accent_cyan']};
             margin-bottom:1.2rem;">
            <div class="kpi-label">Kết quả nghiên cứu</div>
            <div style="color:{COLORS['text_primary']}; font-size:1.05rem;
                 line-height:1.6;">{FINDING_HEADLINE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(build_gap_figure(rows), use_container_width=True)

    st.markdown("**Đọc biểu đồ này thế nào**")
    for line in FINDING_READING:
        st.markdown(f"- {line}")

    # The table view: every value reachable without reading a tooltip or a hue.
    with st.expander("Xem số dạng bảng"):
        import pandas as pd

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Đồng tiền": row.asset,
                        "Horizon (giao dịch)": row.horizon,
                        "Thời gian": row.duration,
                        "Đạt được (%)": round(row.achieved, 1),
                        "Cần có (%)": round(row.required, 1),
                        "Còn thiếu (điểm)": round(row.shortfall, 1),
                        "Lãi ròng (bps/lệnh)": round(row.net_edge_bps, 2),
                        "Cửa sổ kiểm định": row.n_test,
                    }
                    for row in sorted(
                        rows, key=lambda r: (ASSETS.index(r.asset), r.horizon)
                    )
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Nguồn: `reports/research/horizon_edge_*.json`, kịch bản phí maker "
        "2bps/chiều — mức phí **rẻ nhất** trong bốn kịch bản được xét. "
        "Đã tính adverse selection. Nhãn: EXPLORATORY_ONLY_NOT_CONFIRMATORY."
    )
