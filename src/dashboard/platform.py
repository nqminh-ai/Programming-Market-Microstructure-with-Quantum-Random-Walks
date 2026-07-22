"""QRW Financial Platform — Multi-panel Streamlit Dashboard.

Bloomberg-style dark terminal UI with 5 module tabs:
   Volatility (A1) |  Risk (A2) |  Signal (A3)
   Optimizer (A4)  |  Anomaly (A5)

Run:
    streamlit run src/dashboard/platform.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.design_system import COLORS, GLOBAL_CSS, PLOTLY_TEMPLATE
from src.dashboard.plain_language import (
    glossary_expander,
    render_start_here,
    tab_explainer,
)
from src.data.paths import asset_data_dir

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"


# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------

def _configure_page() -> None:
    import streamlit as st
    st.set_page_config(
        page_title="QRW Financial Platform",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def format_percent(value: float, decimals: int = 1) -> str:
    """Format a fraction as a percentage, falling back to scientific
    notation when the value would otherwise round to "0.0%" and silently
    look identical to an actual zero (e.g. a GARCH forecast of 0.00024%)."""
    percent = value * 100
    formatted = f"{percent:.{decimals}f}%"
    if percent != 0.0 and float(formatted[:-1]) == 0.0:
        return f"{percent:.2e}%"
    return formatted


def kpi_card(
    col,
    label: str,
    value: str,
    delta: float | None = None,
    color: str = "#00D4FF",
    delta_label: str = "vs realized",
) -> None:
    """Render a KPI metric card with optional delta."""
    import streamlit as st

    delta_html = ""
    if delta is not None:
        cls = "pos" if delta >= 0 else "neg"
        sign = "+" if delta >= 0 else ""
        delta_html = (
            f'<div class="kpi-delta {cls}">'
            f'{sign}{delta * 100:.2f}% {delta_label}'
            f"</div>"
        )

    col.markdown(
        f"""
        <div class="kpi-card" style="border-left: 3px solid {color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color}">{value}</div>
            {delta_html}
        </div>""",
        unsafe_allow_html=True,
    )


def insight_panel(observations: list[str], recommendation: str, *, tone: str = "neutral") -> None:
    """Render a plain-language interpretation of the tab's own numbers plus
    one concrete recommendation.

    Every observation must be derived from the values already computed in
    that tab (not a static blurb), so this reads as "what these numbers
    mean" rather than a generic disclaimer. `tone` picks the accent color:
    good/caution/bad/neutral.
    """
    import streamlit as st

    color = {
        "good": COLORS["accent_green"],
        "caution": COLORS["accent_yellow"],
        "bad": COLORS["accent_red"],
        "neutral": COLORS["accent_cyan"],
    }[tone]
    observations_html = "".join(
        f'<div style="margin-top:0.3rem;">• {line}</div>' for line in observations
    )
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left:3px solid {color}; margin-top:0.75rem;">
            <div class="kpi-label">📌 Insight &amp; Khuyến nghị</div>
            <div style="margin-top:0.4rem; color:#C7D3E0; font-size:0.85rem; line-height:1.5;">
                {observations_html}
            </div>
            <div style="margin-top:0.6rem; padding-top:0.6rem; border-top:1px solid #1C2A3D; color:{color}; font-weight:600; font-size:0.85rem;">
                → {recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    import streamlit as st
    st.markdown(
        f'<div class="section-header">{text}</div>', unsafe_allow_html=True
    )


def _artifacts_ready(*paths: Path) -> bool:
    """True when DEMO_MODE is on and every given artifact file exists.

    Shared by each tab's `_load_*` helper so the same
    "DEMO_MODE and <file>.exists()" check isn't repeated with slightly
    different shapes across five tabs.
    """
    return DEMO_MODE and all(path.exists() for path in paths)


def synthetic_data_banner(command_hint: str) -> None:
    """Warn that the panel below is fabricated demo data, not a real result.

    Every tab must call this before rendering a random/hardcoded fallback so
    numbers with the visual confidence of a real backtest are never shown
    without a visible caveat.
    """
    import streamlit as st
    st.warning(
        "⚠ **Phần dưới đây là số liệu giả do máy tự sinh**, không phải kết quả "
        "tính từ dữ liệu thị trường. Nó chỉ để bạn thấy giao diện trông ra sao. "
        "Đừng diễn giải các con số này.\n\n"
        "Người phụ trách kỹ thuật có thể tạo dữ liệu thật bằng lệnh:\n\n"
        f"```bash\n{command_hint}\n```"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sync_live_collector(enabled: bool, asset: str):
    """Keep one SSI worker per Streamlit session and selected asset."""
    import streamlit as st
    from src.data.ssi_live_collector import SSIClientConfig, SSILiveCollector

    state_key = "_ssi_live_collector"
    existing = st.session_state.get(state_key)
    if not enabled or asset == "Synthetic":
        if existing is not None:
            existing.stop()
            st.session_state.pop(state_key, None)
        return None

    if existing is not None and existing.symbol != asset:
        existing.stop()
        st.session_state.pop(state_key, None)
        existing = None

    runtime_ssi_config = SSIClientConfig.from_environment()
    if (
        existing is not None
        and not existing.config.has_credentials
        and runtime_ssi_config.has_credentials
    ):
        existing.stop()
        st.session_state.pop(state_key, None)
        existing = None

    if existing is None:
        existing = SSILiveCollector(
            symbol=asset,
            max_events=1000,
            config=runtime_ssi_config,
        )
        st.session_state[state_key] = existing
    existing.start()
    return existing


def render_sidebar() -> dict:
    import streamlit as st

    with st.sidebar:
        st.markdown(
            '<div class="section-header">QRW PLATFORM</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Bạn có thể để nguyên mọi thiết lập bên dưới — chúng đã được đặt "
            "sẵn ở mức hợp lý. Di chuột vào dấu **?** để xem giải thích."
        )

        date = st.date_input(
            "Ngày phân tích",
            value=datetime(2026, 6, 12).date(),
            key="sidebar_date",
            help="Ngày dữ liệu thị trường được đem ra phân tích.",
        )
        asset = st.selectbox(
            "Tài sản theo dõi",
            ["VN30F1M", "VNINDEX", "Synthetic"],
            key="sidebar_asset",
            help="VN30F1M và VNINDEX là chỉ số chứng khoán Việt Nam. "
                 "'Synthetic' là dữ liệu giả lập do máy tạo ra để thử nghiệm.",
        )

        st.markdown("---")
        st.markdown(
            '<div class="kpi-label">Dữ liệu trực tiếp</div>',
            unsafe_allow_html=True,
        )
        requested_live = st.toggle(
            "Nối dữ liệu thị trường thật",
            value=False,
            disabled=asset == "Synthetic",
            key="ssi_live_mode",
            help="Bật để lấy giá trực tiếp từ sàn (cần tài khoản dữ liệu SSI). "
                 "Tắt thì bảng dùng dữ liệu đã lưu sẵn.",
        )
        live_mode = bool(requested_live and asset != "Synthetic")
        refresh_seconds = st.slider(
            "Làm mới sau mỗi (giây)",
            1,
            10,
            1,
            disabled=not live_mode,
            key="ssi_refresh_seconds",
            help="Bảng tự cập nhật lại sau mỗi khoảng thời gian này.",
        )
        collector = _sync_live_collector(live_mode, asset)
        if collector is not None:
            live_info = collector.status_snapshot()
            st.caption(
                f"SSI: {live_info['status']} · "
                f"{live_info['events']}/{collector.max_events} events"
            )

        st.markdown("---")
        st.markdown(
            '<div class="kpi-label">Thiết lập dự báo</div>',
            unsafe_allow_html=True,
        )
        window = st.slider(
            "Số giao dịch dùng để tính",
            5,
            30,
            15,
            5,
            key="vol_window",
            help="Mô hình nhìn lại bao nhiêu giao dịch gần nhất để ước lượng "
                 "mức biến động. Số nhỏ = phản ứng nhanh nhưng nhiễu; "
                 "số lớn = mượt hơn nhưng chậm.",
        )
        show_garch = st.checkbox(
            "Hiện mô hình đối chứng GARCH",
            True,
            key="show_garch",
            help="GARCH là mô hình cổ điển đã dùng nhiều thập kỷ trong tài chính, "
                 "để đây làm mốc so sánh.",
        )
        show_real = st.checkbox(
            "Hiện mức biến động đã thực sự xảy ra",
            True,
            key="show_real",
            help="Đường sự thật: mức dao động đo được sau khi mọi việc đã diễn ra. "
                 "Dùng để chấm xem dự báo sát đến đâu.",
        )
        theta_buy = st.slider(
            "Ngưỡng tự tin để gợi ý MUA",
            0.50,
            0.80,
            0.60,
            0.01,
            key="theta_buy",
            help="Mô hình chỉ gợi ý MUA khi tự tin vượt mức này. Kéo lên cao "
                 "⟹ ít tín hiệu hơn nhưng chọn lọc hơn.",
        )
        theta_sell = st.slider(
            "Ngưỡng tự tin để gợi ý BÁN",
            0.50,
            0.80,
            0.60,
            0.01,
            key="theta_sell",
            help="Tương tự nhưng cho chiều bán. Kéo lên cao ⟹ ít tín hiệu bán hơn.",
        )

        st.markdown("---")
        st.markdown(
            '<div class="kpi-label">Tình trạng các mục</div>',
            unsafe_allow_html=True,
        )
        results_dir = ROOT / "results" / "track_a"
        modules = {
            "Mức biến động": (results_dir / "vol_metrics.json").exists(),
            "Rủi ro thua lỗ": (results_dir / "risk_paths.parquet").exists(),
            "Tín hiệu mua/bán": (results_dir / "signal_log.parquet").exists(),
            "Dò tham số": (results_dir / "optimizer_params.json").exists(),
            "Bất thường": (results_dir / "anomaly_log.parquet").exists(),
        }
        status_html = '<div style="display:flex; flex-direction:column; gap:0.4rem; margin-top:0.5rem;">'
        for name, ready in modules.items():
            color = "#00E676" if ready else "#4A6080"
            status = "✓ có dữ liệu" if ready else "○ chưa có dữ liệu"
            status_html += (
                f'<div style="font-family:\'Inter\',sans-serif; font-size:0.78rem; color:{color};">'
                f"{name}: {status}"
                f"</div>"
            )
        status_html += "</div>"
        st.markdown(status_html, unsafe_allow_html=True)

        st.markdown("---")
        ready_count = sum(modules.values())
        st.caption(
            f"{ready_count}/5 mục đã có dữ liệu để hiển thị.  \n"
            + (
                "Đang ở **chế độ trình diễn** — số liệu chỉ để minh hoạ cách "
                "dùng, không phải kết quả đầu tư."
                if DEMO_MODE
                else "Đang chạy trên dữ liệu thật."
            )
        )

    return {
        "date": str(date),
        "asset": asset,
        "live_mode": live_mode,
        "refresh_seconds": refresh_seconds,
        "collector": collector,
        "window": window,
        "show_garch": show_garch,
        "show_real": show_real,
        "theta_buy": theta_buy,
        "theta_sell": theta_sell,
        "modules_status": modules,
    }


# ---------------------------------------------------------------------------
# Global header
# ---------------------------------------------------------------------------

def render_header(config: dict) -> None:
    import streamlit as st

    # Computed once by render_sidebar() and threaded through config so the
    # module-readiness check isn't duplicated (and can't drift) here.
    modules_status = config["modules_status"]

    pills_html = ""
    for name, ready in modules_status.items():
        border_color = "#00E676" if ready else "#4A6080"
        text_color = "#00E676" if ready else "#4A6080"
        status = "✓" if ready else "○"
        pills_html += (
            f'<span style="background:#0D1420; border:1px solid {border_color}; '
            f'color:{text_color}; padding:0.2rem 0.6rem; border-radius:3px; '
            f'font-size:0.72rem; font-family:\'Inter\',sans-serif;">'
            f"{status} {name}</span>"
        )

    live_info = config.get("live_info")
    if config.get("live_mode") and live_info:
        live_status = str(live_info["status"])
        live_color = {
            "LIVE": COLORS["accent_green"],
            "REST FALLBACK": COLORS["accent_yellow"],
            "MARKET CLOSED": COLORS["text_muted"],
            "CONNECTING": COLORS["accent_cyan"],
        }.get(live_status, COLORS["accent_red"])
        pills_html += (
            f'<span style="background:#0D1420; border:1px solid {live_color}; '
            f'color:{live_color}; padding:0.2rem 0.6rem; border-radius:3px; '
            f'font-size:0.7rem; font-family:\'JetBrains Mono\';">'
            f"SSI {live_status}</span>"
        )

    # `now` is the page-render wall-clock time, not the analysis date's
    # timestamp -- label both explicitly so the pair doesn't read as one
    # live "as-of" moment when most tabs actually serve static demo data.
    rendered_at = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:0.5rem 0; border-bottom:1px solid #1C2A3D; margin-bottom:1rem;">
            <div style="font-family:'JetBrains Mono'; color:#00D4FF; font-weight:600; font-size:1.1rem;">
                QRW FINANCIAL PLATFORM
            </div>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                {pills_html}
            </div>
            <div style="font-family:'JetBrains Mono'; color:#4A6080; font-size:0.7rem;">
                Analysis date {config['date']} · rendered {rendered_at}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab A1 — Volatility Forecaster
# ---------------------------------------------------------------------------

def _build_live_volatility_frame(
    ticks: pd.DataFrame,
    window: int,
    *,
    include_garch: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a live chart frame and next-tick forecast from memory."""
    from src.models.qrw_core import QuantumRandomWalk
    from src.models.volatility_forecaster import QRWVolatilityForecaster

    market = ticks.replace([np.inf, -np.inf], np.nan).dropna(subset=["price"])
    market = market[market["price"] > 0].reset_index(drop=True)
    positions = max(63, 2 * int(window) + 3)
    if positions % 2 == 0:
        positions += 1
    forecaster = QRWVolatilityForecaster(QuantumRandomWalk(n_positions=positions))
    latest = forecaster.latest_vol_forecast(market, window=window)

    parts = [
        forecaster.rolling_vol_forecast(market, window=window),
        forecaster._realized_trailing_vol(market, window=window),
    ]
    if include_garch and len(market) >= 25:
        try:
            parts.append(forecaster._fixed_origin_garch_vol(market, window=window))
        except (RuntimeError, ValueError):
            pass
    frame = pd.concat(parts, axis=1).dropna(subset=["vol_qrw", "vol_realized"])
    if frame.empty:
        frame = pd.DataFrame(
            {
                "vol_qrw": [latest["vol_qrw"]],
                "vol_realized": [latest["vol_realized"]],
            },
            index=[market.index[-1]],
        )
    frame["timestamp"] = market.loc[frame.index, "timestamp"].to_numpy()
    return frame, latest


def tab_volatility(config: dict) -> None:
    import streamlit as st

    window = int(config["window"])
    show_garch = bool(config["show_garch"])
    show_real = bool(config["show_real"])

    section_header("VOLATILITY FORECASTER — MODULE A1")

    # Load pre-computed data
    @st.cache_data(ttl=60)
    def _load_vol_data(date: str, window_size: int):
        vol_parquet = ROOT / "results" / "track_a" / "vol_metrics.parquet"
        metrics_json = ROOT / "results" / "track_a" / "vol_metrics.json"
        df, met = None, {}
        if _artifacts_ready(vol_parquet):
            try:
                df = pd.read_parquet(vol_parquet)
            except Exception as error:
                logger.warning("Found {} but failed to read it: {}", vol_parquet, error)
        if _artifacts_ready(metrics_json):
            try:
                with open(metrics_json) as f:
                    met = json.load(f)
            except Exception as error:
                logger.warning("Found {} but failed to read it: {}", metrics_json, error)
        return df, met

    latest_live: dict[str, Any] | None = None
    if config.get("live_mode"):
        live_ticks = config.get("live_frame", pd.DataFrame())
        if len(live_ticks) < window:
            live_info = config.get("live_info", {})
            st.info(
                f"SSI {live_info.get('status', 'CONNECTING')}. "
                f"Waiting for ticks: {len(live_ticks)}/{window}."
            )
            return
        try:
            df, latest_live = _build_live_volatility_frame(
                live_ticks,
                window,
                include_garch=show_garch,
            )
        except (RuntimeError, ValueError) as error:
            st.warning(f"Live volatility forecast is warming up: {error}")
            return
        metrics = {}
    else:
        df, metrics = _load_vol_data(config["date"], window)

    if df is None or df.empty:
        synthetic_data_banner(
            "python scripts/track_a/volatility_demo.py --date 2026-06-12 --window 15"
        )

        # Show demo with synthetic data
        from src.models.qrw_core import QuantumRandomWalk
        from src.models.volatility_forecaster import QRWVolatilityForecaster

        rng = np.random.default_rng(42)
        t = np.arange(200)
        demo_qrw = 0.15 + 0.05 * np.sin(t / 20) + rng.normal(0, 0.01, 200)
        demo_real = 0.14 + 0.04 * np.sin(t / 18 + 0.5) + rng.normal(0, 0.01, 200)
        demo_garch = 0.13 + 0.035 * np.sin(t / 22 + 1.0) + rng.normal(0, 0.01, 200)
        df = pd.DataFrame(
            {"vol_qrw": demo_qrw, "vol_garch": demo_garch, "vol_realized": demo_real}
        )
        metrics = {}

    # Row 1: 4 KPI Cards
    from src.models.volatility_forecaster import QRWVolatilityForecaster

    forecaster = QRWVolatilityForecaster(qrw_model=None)
    current_qrw = float(
        latest_live["vol_qrw"] if latest_live is not None else df["vol_qrw"].iloc[-1]
    )
    current_real = float(
        latest_live["vol_realized"]
        if latest_live is not None
        else df["vol_realized"].iloc[-1]
    )
    current_garch = (
        float(df["vol_garch"].dropna().iloc[-1])
        if "vol_garch" in df.columns and not df["vol_garch"].dropna().empty
        else None
    )
    regime = forecaster.vol_regime(current_qrw)
    regime_color = {"LOW": "#00E676", "MID": "#FFB300", "HIGH": "#FF4444"}[regime]

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "QRW Vol Forecast", format_percent(current_qrw), current_qrw - current_real, "#00D4FF", "vs realized")
    kpi_card(
        c2,
        "GARCH Forecast",
        format_percent(current_garch) if current_garch is not None else "N/A",
        current_garch - current_real if current_garch is not None else None,
        "#B388FF",
        "vs realized",
    )
    kpi_card(c3, "Realized Vol", format_percent(current_real), color="#E8F0FA")
    kpi_card(c4, "Vol Regime", regime, color=regime_color)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Main Vol Chart
    fig = go.Figure()
    x_axis = (
        df["timestamp"]
        if "timestamp" in df.columns
        else list(range(len(df))) if not hasattr(df.index, "strftime") else df.index
    )

    fig.add_trace(go.Scatter(
        x=x_axis, y=df["vol_qrw"] * 100,
        name="QRW (ours)", mode="lines",
        line=dict(color="#00D4FF", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
    ))
    if show_garch and "vol_garch" in df.columns and df["vol_garch"].notna().any():
        fig.add_trace(go.Scatter(
            x=x_axis, y=df["vol_garch"] * 100,
            name="GARCH", mode="lines",
            line=dict(color="#7B68EE", width=1.5, dash="dash"),
        ))
    if show_real:
        fig.add_trace(go.Scatter(
            x=x_axis, y=df["vol_realized"] * 100,
            name="Realized", mode="lines",
            line=dict(color="#E8F0FA", width=1),
        ))

    # Regime threshold lines
    fig.add_hline(y=10, line=dict(color="#4A6080", width=1, dash="dot"),
                  annotation_text="LOW/MID (10%)", annotation_font_color="#4A6080")
    fig.add_hline(y=25, line=dict(color="#4A6080", width=1, dash="dot"),
                  annotation_text="MID/HIGH (25%)", annotation_font_color="#4A6080")

    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title="Volatility Forecast — QRW vs GARCH vs Realized",
        height=380,
        yaxis_title="Annualized Vol (%)",
        xaxis_title="Time",
        legend=dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, width="stretch")

    # Row 3: Metrics table + Distribution
    col_left, col_right = st.columns([6, 4])

    with col_left:
        section_header("Rolling Window Comparison")
        metric_rows = []
        for w in [5, 15, 30]:
            row = metrics.get(f"window_{w}", {})
            if row:
                winner = "QRW" if row.get("qrw_wins_mae") else "GARCH"
                metric_rows.append({
                    "Window": f"{w} ticks",
                    "QRW MAE": f"{row.get('mae_qrw', 0):.5f}",
                    "GARCH MAE": f"{row.get('mae_garch', 0):.5f}",
                    "QRW RMSE": f"{row.get('rmse_qrw', 0):.5f}",
                    "GARCH RMSE": f"{row.get('rmse_garch', 0):.5f}",
                    "Winner": winner,
                })
        if metric_rows:
            st.dataframe(
                pd.DataFrame(metric_rows),
                hide_index=True,
                width="stretch",
            )
        else:
            if config.get("live_mode"):
                st.caption(
                    f"Live SSI buffer: {len(config.get('live_frame', []))} ticks. "
                    "Benchmark metrics remain out-of-sample artifacts."
                )
            else:
                st.caption("Run demo script to populate metrics.")

    with col_right:
        section_header("Vol Distribution")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=df["vol_qrw"] * 100, name="QRW",
            marker_color="#00D4FF", opacity=0.55, nbinsx=25,
        ))
        if "vol_garch" in df.columns and df["vol_garch"].notna().any():
            fig2.add_trace(go.Histogram(
                x=df["vol_garch"] * 100, name="GARCH",
                marker_color="#7B68EE", opacity=0.45, nbinsx=25,
            ))
        fig2.update_layout(
            **PLOTLY_TEMPLATE,
            barmode="overlay",
            height=250,
            showlegend=True,
            legend=dict(orientation="h", y=-0.2),
            xaxis_title="Annualized Vol (%)",
        )
        st.plotly_chart(fig2, width="stretch")

    if metric_rows:
        qrw_wins = sum(1 for row in metric_rows if row["Winner"] == "QRW")
        total_windows = len(metric_rows)
        gap_pct = abs(current_qrw - current_real) * 100
        regime_note = {
            "LOW": "biến động thấp, điều kiện thị trường tương đối ổn định",
            "MID": "biến động trung bình, nên theo dõi sát các mốc chuyển regime",
            "HIGH": "biến động cao — kích thước vị thế/đòn bẩy nên được giảm tương ứng",
        }[regime]
        observations = [
            f"QRW thắng GARCH về sai số dự báo (MAE) ở {qrw_wins}/{total_windows} khung thời gian được so sánh.",
            f"Regime hiện tại: <b>{regime}</b> — {regime_note}.",
            f"Chênh lệch dự báo QRW so với realized: {gap_pct:.2f} điểm phần trăm.",
        ]
        if qrw_wins > total_windows / 2:
            recommendation = (
                "QRW đang dự báo sát thực tế hơn GARCH ở đa số khung thời gian — "
                "có thể ưu tiên tín hiệu QRW cho khung ngắn, vẫn nên đối chiếu GARCH làm tham chiếu."
            )
            tone = "good"
        elif qrw_wins == total_windows / 2:
            recommendation = (
                "QRW và GARCH đang ngang ngửa nhau — chưa đủ cơ sở để ưu tiên một mô hình, "
                "nên dùng cả hai làm tín hiệu chéo kiểm tra lẫn nhau."
            )
            tone = "caution"
        else:
            recommendation = (
                "GARCH vẫn chính xác hơn QRW ở đa số khung thời gian trên dữ liệu này — "
                "dùng QRW như tín hiệu bổ sung, chưa nên thay thế GARCH làm dự báo chính."
            )
            tone = "caution"
        insight_panel(observations, recommendation, tone=tone)

    # Row 4: Methodology
    with st.expander("Mathematical Methodology"):
        st.latex(
            r"\sigma^2_{QRW}(t) = \sum_x x^2 \cdot P(x,t)"
            r" - \left[\sum_x x \cdot P(x,t)\right]^2"
        )
        st.latex(r"\text{Vol}_{QRW}(t) = \sqrt{\frac{\sigma^2_{QRW}(t)}{t}} \times \sqrt{252 \times 6.5 \times 3600}")
        st.markdown("""
        **Intuition:**
        - QRW tạo ra probability distribution trên position space sau mỗi bước.
        - Variance của distribution này tăng theo **t²** (quadratic), nhanh hơn CRW (linear **t**).
        - Điều này phù hợp với **fat-tail behavior** của thị trường tài chính.
        - Không cần additional model — QRW variance là proxy tự nhiên cho volatility.
        """)


# ---------------------------------------------------------------------------
# Tab A2 — Risk Simulator
# ---------------------------------------------------------------------------

def tab_risk(config: dict) -> None:
    import streamlit as st
    from scipy.stats import norm

    from src.models.risk_simulator import QRWRiskSimulator, SCENARIOS

    section_header("RISK SIMULATOR — MODULE A2")
    scenario = st.radio(
        "Scenario",
        list(SCENARIOS),
        format_func=lambda key: SCENARIOS[key]["label"],
        horizontal=True,
        key="risk_scenario",
    )
    control_1, control_2 = st.columns([4, 1])
    with control_1:
        severity = st.slider(
            "Additional severity multiplier", 0.5, 2.0, 1.0, 0.1,
            key="risk_severity",
        )
    with control_2:
        if st.button("Run New Simulation", type="primary", key="run_risk"):
            st.session_state["risk_seed"] = st.session_state.get("risk_seed", 2026) + 1

    simulator = QRWRiskSimulator(
        n_paths=1000,
        seed=int(st.session_state.get("risk_seed", 2026)),
    )

    @st.cache_data(ttl=60)
    def _load_risk_paths(scenario_key: str) -> pd.DataFrame | None:
        artifact = ROOT / "results" / "track_a" / "risk_paths.parquet"
        if not _artifacts_ready(artifact):
            return None
        stored = pd.read_parquet(artifact)
        return stored[stored["scenario"] == scenario_key]

    @st.cache_data(ttl=60)
    def _load_risk_backtest() -> pd.DataFrame | None:
        backtest_artifact = ROOT / "results" / "track_a" / "risk_backtest.parquet"
        if not _artifacts_ready(backtest_artifact):
            return None
        return pd.read_parquet(backtest_artifact)

    selected = _load_risk_paths(scenario)
    if selected is not None:
        paths = selected.pivot(
            index="path_id", columns="step", values="cumulative_return"
        ).to_numpy()
    else:
        synthetic_data_banner("python scripts/track_a/build_demo.py --date 2026-06-12")
        lattice = np.linspace(-1.0, 1.0, 101)
        probability = np.exp(-0.5 * (lattice / 0.18) ** 2)
        probability /= probability.sum()
        paths = simulator.scenario_var(probability, scenario)["paths"]
    paths = paths * severity

    var_95 = simulator.compute_var(paths, 0.95)
    var_99 = simulator.compute_var(paths, 0.99)
    es_95 = simulator.compute_es(paths, 0.95)
    final = paths[:, -1]
    gaussian_95 = float(final.mean() + norm.ppf(0.05) * final.std(ddof=1))
    gaussian_99 = float(final.mean() + norm.ppf(0.01) * final.std(ddof=1))

    backtest = _load_risk_backtest()
    if backtest is None:
        synthetic_data_banner("python scripts/track_a/build_demo.py --date 2026-06-12")
        rng = np.random.default_rng(2026)
        synthetic_returns = rng.normal(0.0, 0.001, 600)
        market = pd.DataFrame(
            {"price": 100 * np.exp(np.cumsum(synthetic_returns))}
        )
        backtest = simulator.backtest_var(market, window=100)["details"]
    violation_rate = float(backtest["violation"].mean())

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "QRW VaR 95%", f"{var_95*100:.2f}%", var_95 - gaussian_95, COLORS["accent_red"], "vs Gaussian")
    kpi_card(c2, "QRW VaR 99%", f"{var_99*100:.2f}%", var_99 - gaussian_99, COLORS["accent_red"], "vs Gaussian")
    kpi_card(c3, "Expected Shortfall", f"{es_95*100:.2f}%", color=COLORS["accent_yellow"])
    kpi_card(c4, "Violation Rate", f"{violation_rate*100:.2f}%", violation_rate - 0.05, COLORS["accent_green"], "vs 5% target")

    path_col, distribution_col = st.columns([55, 45])
    with path_col:
        fan = go.Figure()
        for path in paths[:200]:
            fan.add_trace(go.Scatter(
                y=path * 100, mode="lines",
                line=dict(color="rgba(0,212,255,0.04)", width=0.5),
                showlegend=False, hoverinfo="skip",
            ))
        fan.add_trace(go.Scatter(
            y=np.median(paths, axis=0) * 100, name="Median",
            line=dict(color=COLORS["accent_cyan"], width=2),
        ))
        fan.add_trace(go.Scatter(
            y=np.percentile(paths, 5, axis=0) * 100, name="VaR 5th",
            line=dict(color=COLORS["accent_red"], width=1.5, dash="dash"),
        ))
        fan.update_layout(
            **PLOTLY_TEMPLATE, height=340,
            title=f"1,000 QRW paths — {SCENARIOS[scenario]['label']}",
            yaxis_title="Cumulative return (%)",
        )
        st.plotly_chart(fan, width="stretch")

    with distribution_col:
        distribution = go.Figure()
        distribution.add_trace(go.Histogram(
            x=final * 100, name="QRW",
            marker_color=COLORS["accent_cyan"], opacity=0.55, nbinsx=50,
        ))
        sigma = max(float(final.std(ddof=1)), 1e-12)
        x_values = np.linspace(float(final.min()), float(final.max()), 250)
        bin_width = max(float(final.max() - final.min()) / 50, 1e-12)
        gaussian_density = norm.pdf(
            x_values, float(final.mean()), sigma
        ) * len(final) * bin_width
        distribution.add_trace(go.Scatter(
            x=x_values * 100, y=gaussian_density, name="Gaussian",
            line=dict(color=COLORS["accent_purple"], dash="dash"),
        ))
        distribution.add_annotation(
            x=var_99 * 100, y=max(gaussian_density) * 0.35,
            text="QRW fat tail", showarrow=True,
            arrowcolor=COLORS["accent_red"],
            font=dict(color=COLORS["accent_red"]),
        )
        distribution.update_layout(
            **PLOTLY_TEMPLATE, height=340,
            title="Terminal return distribution", barmode="overlay",
            xaxis_title="Return (%)",
        )
        st.plotly_chart(distribution, width="stretch")

    backtest_figure = go.Figure()
    backtest_figure.add_hrect(
        y0=4.0, y1=6.0, fillcolor="rgba(0,230,118,0.10)", line_width=0,
        annotation_text="4–6% calibration band",
    )
    backtest_figure.add_trace(go.Scatter(
        x=backtest["row"], y=backtest["rolling_violation_rate"] * 100,
        name="Rolling violations",
        line=dict(color=COLORS["accent_yellow"], width=2),
    ))
    backtest_figure.update_layout(
        **PLOTLY_TEMPLATE, height=240,
        title="VaR backtest — rolling violation rate",
        xaxis_title="Observation", yaxis_title="Violation rate (%)",
    )
    st.plotly_chart(backtest_figure, width="stretch")

    # In basis points, not percentage points: VaR gaps here are routinely
    # under 0.01pp, which would otherwise silently round to "0.00".
    tail_gap_bps = (gaussian_95 - var_95) * 10_000
    violation_gap = (violation_rate - 0.05) * 100
    observations = [
        (
            f"QRW VaR 95% ({var_95*100:.2f}%) {'dày hơn' if var_95 < gaussian_95 else 'mỏng hơn'} "
            f"Gaussian ({gaussian_95*100:.2f}%) {abs(tail_gap_bps):.1f} bps."
        ),
        f"Tỷ lệ vi phạm VaR thực tế: {violation_rate*100:.2f}% (mục tiêu 5%).",
    ]
    if violation_gap > 1.0:
        recommendation = (
            "Tỷ lệ vi phạm cao hơn mục tiêu 5% — mô hình đang ĐÁNH GIÁ THẤP rủi ro thực tế. "
            "Cần tăng biên an toàn (nới VaR) hoặc kiểm tra lại calibration trước khi dùng để quản trị rủi ro thật."
        )
        tone = "bad"
    elif violation_gap < -2.0:
        recommendation = (
            "Tỷ lệ vi phạm thấp hơn nhiều so với mục tiêu 5% — mô hình đang thận trọng hơn mức cần thiết, "
            "có thể đang khoá vốn dự phòng nhiều hơn cần. Có thể nới nhẹ ngưỡng VaR để dùng vốn hiệu quả hơn."
        )
        tone = "caution"
    else:
        recommendation = (
            "Tỷ lệ vi phạm gần với mục tiêu 5% — mô hình VaR đang hiệu chỉnh tốt trên dữ liệu này. "
            "Tiếp tục backtest định kỳ để phát hiện sớm nếu calibration trôi theo thời gian."
        )
        tone = "good"
    insight_panel(observations, recommendation, tone=tone)


# ---------------------------------------------------------------------------
# Tab A3 — Signal Engine
# ---------------------------------------------------------------------------

def tab_signal(config: dict) -> None:
    import streamlit as st

    from src.strategy.signal_engine import QRWSignalEngine

    section_header("SIGNAL ENGINE — MODULE A3")
    theta_buy = float(config["theta_buy"])
    theta_sell = float(config["theta_sell"])
    engine = QRWSignalEngine(theta_buy, theta_sell)
    metrics_frame: pd.DataFrame | None = None

    @st.cache_data(ttl=60)
    def _load_signal_artifact() -> pd.DataFrame | None:
        artifact = ROOT / "results" / "track_a" / "signal_log.parquet"
        if _artifacts_ready(artifact):
            return pd.read_parquet(artifact)
        return None

    @st.cache_data(ttl=60)
    def _load_signal_market(date: str) -> pd.DataFrame | None:
        candidates = sorted(
            asset_data_dir("BTCUSDT", "processed").glob(f"*{date}*.parquet")
        ) or sorted(asset_data_dir("BTCUSDT", "processed").glob("*.parquet"))
        if not candidates:
            return None
        return pd.read_parquet(candidates[0]).head(500).copy().reset_index(drop=True)
    if config.get("live_mode"):
        market = config.get("live_frame", pd.DataFrame()).copy().reset_index(drop=True)
        if len(market) < 30:
            live_info = config.get("live_info", {})
            st.info(
                f"SSI {live_info.get('status', 'CONNECTING')}. "
                f"Waiting for signal history: {len(market)}/30."
            )
            return
        try:
            latest_prediction = engine.latest_signal(market, min_history=30)
            metrics_frame = (
                engine.backtest_signals(market, warmup=30)
                if len(market) > 31
                else pd.DataFrame()
            )
        except ValueError as error:
            st.warning(f"Live signal engine is warming up: {error}")
            return
        live_row = {
            **latest_prediction,
            "row": len(market) - 1,
            "ret_1step": np.nan,
            "pnl": 0.0,
            "correct": "PENDING",
        }
        signal_frame = pd.concat(
            [metrics_frame, pd.DataFrame([live_row])],
            ignore_index=True,
            sort=False,
        )
    elif DEMO_MODE and (probability_frame := _load_signal_artifact()) is not None:
        signal_frame = engine.backtest_from_probabilities(probability_frame)
    else:
        market = _load_signal_market(config["date"])
        if market is None:
            synthetic_data_banner("python scripts/track_a/build_demo.py --date 2026-06-12")
            rng = np.random.default_rng(2026)
            returns = rng.normal(0.0, 0.0005, 500)
            market = pd.DataFrame({
                "timestamp": pd.date_range("2026-06-12", periods=500, freq="s"),
                "price": 100 * np.exp(np.cumsum(returns)),
            })
        if "timestamp" not in market.columns:
            market["timestamp"] = np.arange(len(market))
        if "obi" not in market.columns:
            signed = np.sign(market["price"].diff().fillna(0.0))
            market["obi"] = signed.rolling(20, min_periods=1).mean()
        signal_frame = engine.backtest_signals(market)

    latest = signal_frame.iloc[-1]
    signal = str(latest["signal"])
    badge_class = {
        "BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"
    }[signal]
    st.markdown(
        f"""
        <div style="text-align:center; padding:1.5rem 0 2rem;">
          <span class="signal-badge {badge_class}" style="font-size:2rem; padding:1rem 2.5rem;">
            {signal}
          </span>
          <div style="margin-top:1rem; color:#8FA3BE; font-family:'Inter';">
            Confidence: {float(latest['confidence'])*100:.1f}% ·
            P(up)={float(latest['p_up']):.3f} · P(down)={float(latest['p_down']):.3f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics_frame is not None and metrics_frame.empty:
        metrics = {
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
        }
    else:
        metrics = engine.compute_signal_metrics(
            metrics_frame if metrics_frame is not None else signal_frame
        )
    probability_col, metric_col = st.columns([45, 55])
    with probability_col:
        probability_figure = go.Figure(go.Bar(
            x=["P(down)", "P(flat)", "P(up)"],
            y=[latest["p_down"] * 100, latest["p_flat"] * 100, latest["p_up"] * 100],
            marker_color=[COLORS["accent_red"], COLORS["text_muted"], COLORS["accent_green"]],
            text=[f"{latest['p_down']*100:.1f}%", f"{latest['p_flat']*100:.1f}%", f"{latest['p_up']*100:.1f}%"],
            textposition="outside",
        ))
        probability_figure.update_layout(
            **PLOTLY_TEMPLATE, height=240, title="Current QRW probability mass",
            yaxis_range=[0, 100], yaxis_title="Probability (%)",
        )
        st.plotly_chart(probability_figure, width="stretch")
    with metric_col:
        c1, c2, c3 = st.columns(3)
        kpi_card(c1, "Hit Rate", f"{metrics['hit_rate']*100:.1f}%", color=COLORS["accent_green"])
        kpi_card(c2, "Profit Factor", f"{metrics['profit_factor']:.2f}", color=COLORS["accent_cyan"])
        kpi_card(c3, "Net P&L", f"{metrics['net_pnl']*100:+.3f}%", color=COLORS["accent_yellow"])
        st.caption(
            # "Interactive" (recomputed as sliders move), not "live" -- avoid
            # implying a real-time feed when DEMO_MODE serves static data.
            f"Interactive threshold preview: θ_buy={theta_buy:.2f}, θ_sell={theta_sell:.2f} · "
            f"{metrics['n_trades']} trades · max drawdown {metrics['max_drawdown']*100:.3f}%"
        )

    history = go.Figure()
    history.add_trace(go.Scatter(
        x=signal_frame["timestamp"], y=signal_frame["price"],
        name="Price", line=dict(color=COLORS["text_primary"], width=1), yaxis="y",
    ))
    history.add_trace(go.Scatter(
        x=signal_frame["timestamp"], y=signal_frame["momentum"],
        name="QRW momentum", fill="tozeroy",
        line=dict(color=COLORS["accent_cyan"], width=1), yaxis="y2",
    ))
    for label, symbol, color in (
        ("BUY", "triangle-up", COLORS["accent_green"]),
        ("SELL", "triangle-down", COLORS["accent_red"]),
    ):
        selected = signal_frame[signal_frame["signal"] == label]
        history.add_trace(go.Scatter(
            x=selected["timestamp"], y=selected["price"], mode="markers",
            marker=dict(symbol=symbol, size=8, color=color), name=label, yaxis="y",
        ))
    history.update_layout(
        **PLOTLY_TEMPLATE, height=310, title="Signal history and price",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Momentum", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(history, width="stretch")

    section_header("Signal Log — latest 20")
    log = signal_frame.tail(20)[
        ["timestamp", "signal", "confidence", "p_up", "p_down", "correct", "pnl"]
    ].copy()
    outcome_colors = {
        "WIN": "color:#00E676",
        "LOSS": "color:#FF4444",
        "HOLD": "color:#8FA3BE",
        "PENDING": "color:#00D4FF",
    }
    styled = log.style.map(lambda value: outcome_colors.get(value, ""), subset=["correct"])
    st.dataframe(styled, hide_index=True, width="stretch")

    n_trades = int(metrics["n_trades"])
    hit_rate = float(metrics["hit_rate"])
    profit_factor = float(metrics["profit_factor"])
    net_pnl = float(metrics["net_pnl"])
    observations = [
        f"Hit rate: {hit_rate*100:.1f}%, Profit factor: {profit_factor:.2f}, trên {n_trades} lệnh (không tính HOLD).",
        f"Net P&L trên toàn bộ giai đoạn: {net_pnl*100:+.3f}%.",
    ]
    if n_trades < 30:
        observations.append(
            f"Chỉ có {n_trades} lệnh — mẫu nhỏ, các tỷ lệ trên có thể không ổn định."
        )
    if n_trades < 30:
        recommendation = (
            "Số lệnh quá ít để kết luận đáng tin cậy — cần chạy trên khoảng thời gian dài hơn "
            "hoặc nhiều tài sản hơn trước khi đánh giá chất lượng tín hiệu."
        )
        tone = "caution"
    elif hit_rate < 0.5 and profit_factor < 1.0:
        recommendation = (
            "Hit rate dưới 50% và profit factor dưới 1 — chiến lược đang thua nhiều hơn thắng trên dữ liệu này. "
            "CHƯA nên triển khai vốn thật; cần cải thiện mô hình hoặc điều chỉnh ngưỡng θ_buy/θ_sell trước."
        )
        tone = "bad"
    elif profit_factor >= 1.5 and hit_rate >= 0.5:
        recommendation = (
            "Cả hit rate và profit factor đều tích cực trên dữ liệu này — có thể tiếp tục kiểm định "
            "trên dữ liệu ngoài mẫu (out-of-sample) trước khi cân nhắc vốn thật."
        )
        tone = "good"
    else:
        recommendation = (
            "Kết quả trái chiều giữa hit rate và profit factor — nên xem xét thêm max drawdown và "
            "kiểm định trên nhiều giai đoạn/regime khác trước khi ra quyết định."
        )
        tone = "caution"
    insight_panel(observations, recommendation, tone=tone)


# ---------------------------------------------------------------------------
# Tab A4 — Optimizer
# ---------------------------------------------------------------------------

def tab_optimizer(config: dict) -> None:
    import streamlit as st

    from src.strategy.optimizer import QRWStrategyOptimizer

    section_header("DÒ TÌM THAM SỐ TỐT NHẤT")
    controls = st.columns([2, 2, 1])
    objective_labels = {
        "t_stat": "Độ chắc chắn lợi nhuận khác 0 (t-statistic)",
        "hit_rate": "Tỷ lệ lệnh thắng",
        "profit_factor": "Tỷ lệ tiền thắng / tiền thua",
    }
    with controls[0]:
        objective = st.selectbox(
            "Tối ưu theo tiêu chí nào?",
            list(objective_labels),
            format_func=lambda key: objective_labels[key],
            key="opt_obj",
            help="Máy sẽ thử mọi tổ hợp tham số và giữ lại tổ hợp tốt nhất theo "
                 "tiêu chí bạn chọn. Lưu ý: thử càng nhiều tổ hợp thì càng dễ "
                 "tìm được thứ 'có vẻ tốt' do ngẫu nhiên.",
        )
    with controls[1]:
        st.toggle(
            "Tách riêng theo trạng thái thị trường",
            True,
            key="regime_aware",
            help="Tìm tham số riêng cho lúc thị trường êm và lúc biến động mạnh.",
        )
    with controls[2]:
        run_optimization = st.button(
            "Chạy dò tìm", type="primary", key="run_opt"
        )

    if run_optimization:
        candidates = sorted(asset_data_dir("BTCUSDT", "processed").glob("*.parquet"))
        if candidates:
            market = pd.read_parquet(candidates[-1]).head(700).copy().reset_index(drop=True)
        else:
            rng = np.random.default_rng(2026)
            phase = np.arange(700)
            returns = 0.0005 * np.sin(phase / 9) + rng.normal(0, 0.0003, 700)
            market = pd.DataFrame({
                "timestamp": pd.date_range("2026-06-12", periods=700, freq="s"),
                "price": 100 * np.exp(np.cumsum(returns)),
                "obi": np.sin(phase / 9),
            })
        if "timestamp" not in market.columns:
            market["timestamp"] = np.arange(len(market))
        if "obi" not in market.columns:
            market["obi"] = np.sign(market["price"].diff().fillna(0.0)).rolling(20, min_periods=1).mean()
        split = int(len(market) * 0.7)
        progress = st.progress(0.0, text="Running 23×23 threshold search…")
        optimizer = QRWStrategyOptimizer()

        def update_progress(completed: int, total: int) -> None:
            progress.progress(completed / total, text=f"Running grid search… {completed}/{total}")

        best = optimizer.grid_search(
            market.iloc[:split].copy(), objective=objective,
            progress_callback=update_progress,
        )
        oos_result = optimizer.evaluate_out_of_sample(
            market.iloc[split:].reset_index(drop=True), best_params=best
        )
        st.session_state["opt_results"] = best
        st.session_state["opt_surface"] = optimizer.search_surface
        st.session_state["opt_oos"] = oos_result["backtest"]
        st.session_state["opt_metrics"] = {
            key: value for key, value in oos_result.items()
            if key not in {"backtest", "equity_curve", "drawdown_curve"}
        }
        progress.empty()

    @st.cache_data(ttl=60)
    def _load_optimizer_artifacts():
        params_path = ROOT / "results" / "track_a" / "optimizer_params.json"
        surface_path = ROOT / "results" / "track_a" / "optimizer_surface.parquet"
        oos_path = ROOT / "results" / "track_a" / "optimizer_oos.parquet"
        metrics_path = ROOT / "results" / "track_a" / "optimizer_metrics.json"
        if not _artifacts_ready(params_path, surface_path, oos_path, metrics_path):
            return None
        return (
            json.loads(params_path.read_text(encoding="utf-8")),
            pd.read_parquet(surface_path),
            pd.read_parquet(oos_path),
            json.loads(metrics_path.read_text(encoding="utf-8")),
        )

    if "opt_results" in st.session_state:
        results = st.session_state["opt_results"]
        surface = st.session_state["opt_surface"]
        oos = st.session_state["opt_oos"]
        oos_metrics = st.session_state["opt_metrics"]
    elif (artifacts := _load_optimizer_artifacts()) is not None:
        results, surface, oos, oos_metrics = artifacts
    else:
        synthetic_data_banner("python scripts/track_a/build_demo.py --date 2026-06-12")
        theta = np.array(QRWStrategyOptimizer.DEFAULT_GRID)
        x_grid, y_grid = np.meshgrid(theta, theta)
        score = 1.1 - 90 * (x_grid - 0.60) ** 2 - 70 * (y_grid - 0.62) ** 2
        surface = pd.DataFrame({
            "regime": "ALL", "theta_buy": x_grid.ravel(),
            "theta_sell": y_grid.ravel(), "score": score.ravel(),
        })
        results = {"ALL": {"theta_buy": 0.60, "theta_sell": 0.62, "score": 1.1, "metrics": {"hit_rate": 0.58}}}
        rng = np.random.default_rng(2026)
        pnl = rng.normal(0.00005, 0.0004, 160)
        oos = pd.DataFrame({"row": np.arange(len(pnl)), "pnl": pnl})
        equity = oos["pnl"].cumsum()
        oos_metrics = {
            "t_stat": float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl))),
            "sharpe_annualised": None,
            "hit_rate": float((pnl > 0).mean()),
            "max_drawdown": float((equity - equity.cummax()).min()),
            "profit_factor": float(pnl[pnl > 0].sum() / -pnl[pnl < 0].sum()),
            "n_trades": int(len(pnl)),
        }

    heat_col, table_col = st.columns([55, 45])
    all_surface = surface[surface["regime"] == "ALL"]
    heatmap_data = all_surface.pivot(
        index="theta_sell", columns="theta_buy", values="score"
    ).sort_index().sort_index(axis=1)
    optimal = results.get("ALL", next(iter(results.values())))
    with heat_col:
        heatmap = go.Figure(go.Heatmap(
            x=heatmap_data.columns, y=heatmap_data.index, z=heatmap_data.to_numpy(),
            colorscale="Blues", colorbar=dict(title=objective),
        ))
        heatmap.add_trace(go.Scatter(
            x=[optimal["theta_buy"]], y=[optimal["theta_sell"]],
            mode="markers", marker=dict(symbol="star", size=16, color=COLORS["accent_yellow"]),
            name="Optimal",
        ))
        heatmap.update_layout(
            **PLOTLY_TEMPLATE, height=340, title="In-sample objective landscape",
            xaxis_title="θ_buy", yaxis_title="θ_sell",
        )
        st.plotly_chart(heatmap, width="stretch")
    with table_col:
        rows = []
        for regime, values in results.items():
            rows.append({
                "Regime": regime,
                "θ_buy": values.get("theta_buy"),
                "θ_sell": values.get("theta_sell"),
                "Score": values.get("score"),
                "Hit Rate": values.get("metrics", {}).get("hit_rate"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        # The single most important number on this tab: whether the winning
        # configuration survives having been picked as the best of many trials.
        deflated = optimal.get("deflated_sharpe") or {}
        probability = deflated.get("deflated_sharpe_ratio")
        if probability is not None and np.isfinite(probability):
            significant = probability > 0.95
            color = COLORS["accent_green"] if significant else COLORS["accent_red"]
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left:3px solid {color};
                     margin-top:0.75rem;">
                    <div class="kpi-label">Kết quả này có thật hay chỉ do may mắn?</div>
                    <div style="color:{color}; font-size:1.35rem; font-weight:600;">
                        {"Có ý nghĩa thống kê" if significant else "KHÔNG có ý nghĩa thống kê"}
                        &nbsp;<span style="font-size:0.95rem; color:#8FA3BE;">
                        (Deflated Sharpe = {probability:.3f})</span>
                    </div>
                    <div style="margin-top:0.5rem; color:#C7D3E0; font-size:0.86rem;
                         line-height:1.55;">
                        Máy đã thử <b>{deflated.get('n_trials', 0)}</b> tổ hợp tham số.
                        Thử càng nhiều thì càng dễ gặp một tổ hợp trông đẹp
                        <b>hoàn toàn do ngẫu nhiên</b> — với số lần thử này, tổ hợp
                        may mắn nhất sẽ đạt Sharpe khoảng
                        <b>{deflated.get('expected_maximum_sharpe', 0):+.3f}</b> dù
                        không hề có kỹ năng gì. Tổ hợp tốt nhất thực tế đạt
                        <b>{deflated.get('sharpe', 0):+.3f}</b>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.download_button(
            "Export optimal params as JSON",
            data=json.dumps(results, indent=2),
            file_name="optimizer_params.json",
            mime="application/json",
            width="stretch",
        )

    equity = oos["pnl"].cumsum()
    drawdown = equity - equity.cummax().clip(lower=0.0)
    curve = go.Figure()
    x_axis = oos["timestamp"] if "timestamp" in oos.columns else oos.get("row", oos.index)
    curve.add_trace(go.Scatter(x=x_axis, y=equity * 100, name="QRW strategy", line=dict(color=COLORS["accent_cyan"], width=2)))
    curve.add_trace(go.Scatter(x=x_axis, y=drawdown * 100, name="Drawdown", fill="tozeroy", line=dict(color=COLORS["accent_red"], width=1), yaxis="y2"))
    curve.update_layout(
        **PLOTLY_TEMPLATE, height=300, title="Out-of-sample equity and drawdown",
        yaxis=dict(title="Cumulative P&L (%)"),
        yaxis2=dict(title="Drawdown (%)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(curve, width="stretch")
    c1, c2, c3, c4, c5 = st.columns(5)
    annualised = oos_metrics.get("sharpe_annualised")
    kpi_card(
        c1,
        "Sharpe (quy đổi năm)",
        "n/a" if annualised is None else f"{float(annualised):.2f}",
        color=COLORS["accent_cyan"],
    )
    kpi_card(c2, "Tỷ lệ lệnh thắng", f"{float(oos_metrics.get('hit_rate', 0))*100:.1f}%", color=COLORS["accent_green"])
    kpi_card(c3, "Sụt giảm sâu nhất", f"{float(oos_metrics.get('max_drawdown', 0))*100:.2f}%", color=COLORS["accent_red"])
    kpi_card(c4, "Profit Factor", f"{float(oos_metrics.get('profit_factor', 0)):.2f}", color=COLORS["accent_yellow"])
    # Profit Factor and Sharpe can look impressive while resting on very few
    # round trips -- show the count alongside them so a viewer can judge
    # reliability, instead of only seeing the headline ratios.
    kpi_card(c5, "Số lệnh (vòng)", f"{int(oos_metrics.get('n_trades', 0))}", color=COLORS["text_primary"])

    t_stat = float(oos_metrics.get("t_stat", 0.0))
    profit_factor = float(oos_metrics.get("profit_factor", 0.0))
    n_trades = int(oos_metrics.get("n_trades", 0))
    n_observations = len(oos)
    sharpe_text = (
        "Sharpe quy đổi năm: chưa tính được (thiếu mốc thời gian)"
        if annualised is None
        else f"Sharpe quy đổi năm {float(annualised):.2f}"
    )
    observations = [
        f"{sharpe_text}; Profit Factor {profit_factor:.2f}; "
        f"trên {n_trades} lệnh trọn vòng / {n_observations} quan sát out-of-sample.",
        f"t-statistic {t_stat:.2f} — đây <b>không</b> phải Sharpe: nó lớn dần theo "
        "cỡ mẫu, nên không so sánh được giữa các backtest dài ngắn khác nhau.",
    ]
    if profit_factor > 10.0:
        observations.append(
            "Profit Factor rất cao thường bị chi phối bởi một vài lệnh lãi lớn hiếm gặp, "
            "không phản ánh độ ổn định trung bình của chiến lược."
        )
        recommendation = (
            "Profit Factor cực đoan như thế này cần được xác nhận trên nhiều giai đoạn/tài sản khác "
            "trước khi tin tưởng tham số θ_buy/θ_sell tối ưu này cho vốn thật."
        )
        tone = "caution"
    elif n_trades < 30:
        recommendation = (
            f"Chỉ {n_trades} lệnh trong kỳ out-of-sample — mẫu nhỏ, cần backtest dài hơn "
            "trước khi tin tưởng các chỉ số này."
        )
        tone = "caution"
    elif t_stat > 0 and profit_factor >= 1.0:
        recommendation = (
            "Các chỉ số out-of-sample đồng thuận tích cực — có thể tiếp tục theo dõi live-paper-trading "
            "trước khi phân bổ vốn thật."
        )
        tone = "good"
    else:
        recommendation = (
            "Chỉ số out-of-sample chưa đủ tích cực — nên thử objective khác (hit_rate/profit_factor) "
            "hoặc mở rộng lưới θ_buy/θ_sell trước khi triển khai."
        )
        tone = "bad"
    insight_panel(observations, recommendation, tone=tone)


# ---------------------------------------------------------------------------
# Tab A5 — Anomaly Detector
# ---------------------------------------------------------------------------

def tab_anomaly(config: dict) -> None:
    import streamlit as st

    from src.models.anomaly_detector import QRWAnomalyDetector

    section_header("ANOMALY DETECTOR — MODULE A5")

    @st.cache_data(ttl=60)
    def _load_anomaly_artifacts():
        log_path = ROOT / "results" / "track_a" / "anomaly_log.parquet"
        distribution_path = ROOT / "results" / "track_a" / "anomaly_distributions.parquet"
        if not _artifacts_ready(log_path, distribution_path):
            return None
        return pd.read_parquet(log_path), pd.read_parquet(distribution_path)

    @st.cache_data(ttl=60)
    def _load_anomaly_market() -> pd.DataFrame | None:
        candidates = sorted(asset_data_dir("BTCUSDT", "processed").glob("*.parquet"))
        if not candidates:
            return None
        return pd.read_parquet(candidates[-1]).head(500).copy().reset_index(drop=True)

    artifacts = _load_anomaly_artifacts()
    if artifacts is not None:
        anomaly_log, distributions = artifacts
        baseline = distributions["baseline"].to_numpy(dtype=float)
        current = distributions["current"].to_numpy(dtype=float)
    else:
        market = _load_anomaly_market()
        if market is None:
            synthetic_data_banner("python scripts/track_a/build_demo.py --date 2026-06-12")
            rng = np.random.default_rng(2026)
            market = pd.DataFrame({
                "timestamp": pd.date_range("2026-06-12", periods=500, freq="s"),
                "price": 100 * np.exp(np.cumsum(rng.normal(0, 0.0004, 500))),
                "obi": np.clip(rng.normal(0, 0.2, 500), -1, 1),
            })
        if "timestamp" not in market.columns:
            market["timestamp"] = np.arange(len(market))
        if "obi" not in market.columns:
            market["obi"] = np.sign(market["price"].diff().fillna(0.0)).rolling(20, min_periods=1).mean()
        detector = QRWAnomalyDetector(window=50).fit_baseline(market.iloc[:250])
        anomaly_log = detector.rolling_anomaly_scan(market, step=5)
        baseline = detector.baseline
        current = detector.current_distribution

    latest = anomaly_log.iloc[-1]
    score = float(latest["sigma_score"])
    anomaly_type = str(latest["anom_type"])
    type_labels = {
        "normal": ("Normal", COLORS["accent_green"]),
        "suspicious": ("Suspicious", COLORS["accent_yellow"]),
        "directional_pressure": ("Directional Pressure", COLORS["accent_red"]),
        "market_indecision": ("Market Indecision", COLORS["accent_yellow"]),
        "quote_stuffing": ("Quote Stuffing", COLORS["accent_purple"]),
    }
    label, type_color = type_labels.get(
        anomaly_type, (anomaly_type, COLORS["text_muted"])
    )

    gauge_col, type_col, alerts_col = st.columns([35, 30, 35])
    with gauge_col:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(score, 5.0),
            number={"suffix": "σ", "font": {"color": type_color}},
            title={"text": "Anomaly Score"},
            gauge={
                "axis": {"range": [0, 5], "tickcolor": COLORS["text_muted"]},
                "bar": {"color": type_color},
                "bgcolor": COLORS["bg_secondary"],
                "steps": [
                    {"range": [0, 1.5], "color": "rgba(0,230,118,0.12)"},
                    {"range": [1.5, 3], "color": "rgba(255,179,0,0.15)"},
                    {"range": [3, 5], "color": "rgba(255,68,68,0.18)"},
                ],
                "threshold": {"line": {"color": COLORS["accent_red"], "width": 3}, "value": 3},
            },
        ))
        gauge.update_layout(
            **{
                **PLOTLY_TEMPLATE,
                "height": 270,
                "margin": dict(l=30, r=30, t=50, b=20),
            }
        )
        st.plotly_chart(gauge, width="stretch")
    with type_col:
        st.markdown(
            f"""
            <div class="kpi-card" style="margin-top:2.5rem; border-left:3px solid {type_color}; text-align:center;">
              <div class="kpi-label">Detected Type</div>
              <div style="font-family:'Inter'; font-size:1.1rem; color:{type_color};">{label}</div>
              <div style="font-size:0.72rem; color:#8FA3BE; margin-top:0.5rem;">KL={float(latest['raw_score']):.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with alerts_col:
        section_header("Latest Alerts")
        alerts = anomaly_log[anomaly_log["alert"]].tail(5)
        if alerts.empty:
            st.success("No >3σ anomalies in the current window")
        else:
            st.dataframe(
                alerts[["timestamp", "sigma_score", "anom_type"]],
                hide_index=True, width="stretch",
            )

    time_series = go.Figure()
    upper = max(5.0, float(anomaly_log["sigma_score"].max()) * 1.05)
    time_series.add_hrect(y0=0, y1=1.5, fillcolor="rgba(0,230,118,0.08)", line_width=0)
    time_series.add_hrect(y0=1.5, y1=3, fillcolor="rgba(255,179,0,0.10)", line_width=0)
    time_series.add_hrect(y0=3, y1=upper, fillcolor="rgba(255,68,68,0.10)", line_width=0)
    time_series.add_trace(go.Scatter(
        x=anomaly_log["timestamp"], y=anomaly_log["sigma_score"],
        name="Anomaly σ", line=dict(color=COLORS["accent_yellow"], width=2),
    ))
    alerts = anomaly_log[anomaly_log["alert"]]
    time_series.add_trace(go.Scatter(
        x=alerts["timestamp"], y=alerts["sigma_score"], mode="markers",
        marker=dict(color=COLORS["accent_red"], size=8, symbol="diamond"),
        name=">3σ alert",
    ))
    time_series.update_layout(
        **PLOTLY_TEMPLATE, height=300, title="Distribution drift over time",
        yaxis_title="Sigma score", yaxis_range=[0, upper],
    )
    st.plotly_chart(time_series, width="stretch")

    drift_col, calendar_col = st.columns(2)
    with drift_col:
        positions = np.arange(len(baseline)) - len(baseline) // 2
        drift = go.Figure()
        drift.add_trace(go.Scatter(x=positions, y=baseline, name="Baseline", line=dict(color=COLORS["text_muted"], dash="dash")))
        drift.add_trace(go.Scatter(x=positions, y=current, name="Current", fill="tonexty", line=dict(color=COLORS["accent_yellow"])))
        drift.update_layout(**PLOTLY_TEMPLATE, height=280, title="Baseline vs current probability", xaxis_title="QRW position", yaxis_title="Probability")
        st.plotly_chart(drift, width="stretch")
    with calendar_col:
        calendar = anomaly_log.copy()
        timestamps = pd.to_datetime(calendar["timestamp"], errors="coerce")
        calendar["day"] = timestamps.dt.strftime("%Y-%m-%d").fillna("Current")
        calendar["hour"] = timestamps.dt.hour.fillna(0).astype(int)
        heat = calendar.pivot_table(index="day", columns="hour", values="sigma_score", aggfunc="mean", fill_value=0)
        calendar_figure = go.Figure(go.Heatmap(
            x=heat.columns, y=heat.index, z=heat.to_numpy(), colorscale="YlOrRd",
        ))
        calendar_figure.update_layout(**PLOTLY_TEMPLATE, height=280, title="Anomaly calendar", xaxis_title="Hour", yaxis_title="Day")
        st.plotly_chart(calendar_figure, width="stretch")

    recent_alert_count = int(len(anomaly_log[anomaly_log["alert"]].tail(50)))
    observations = [
        f"Điểm bất thường hiện tại: {score:.2f}σ, loại: {label}.",
        f"Số lần vượt ngưỡng 3σ trong 50 quan sát gần nhất: {recent_alert_count}.",
    ]
    type_advice = {
        "quote_stuffing": (
            "Nghi ngờ quote stuffing/spoofing — nên tăng giám sát order book và cân nhắc trì hoãn "
            "khớp lệnh cho tới khi mẫu hình ổn định trở lại."
        ),
        "directional_pressure": (
            "Áp lực một chiều mạnh — cân nhắc điều chỉnh vị thế theo hướng đó nhưng thận trọng "
            "với rủi ro đảo chiều đột ngột."
        ),
        "market_indecision": (
            "Thị trường đang giằng co, thiếu hướng rõ ràng — nên giảm kích thước lệnh và mở rộng "
            "biên độ dừng lỗ trong giai đoạn này."
        ),
        "suspicious": (
            "Có dấu hiệu bất thường nhưng chưa rõ loại — theo dõi thêm trước khi hành động."
        ),
    }
    if score >= 3.0:
        recommendation = type_advice.get(
            anomaly_type,
            "Bất thường mạnh (>3σ) — nên tạm dừng giao dịch tự động và kiểm tra thủ công trước khi tiếp tục.",
        )
        tone = "bad"
    elif score >= 1.5:
        recommendation = type_advice.get(
            anomaly_type,
            "Có dấu hiệu bất thường nhẹ — theo dõi sát, chưa cần can thiệp ngay.",
        )
        tone = "caution"
    else:
        recommendation = "Thị trường đang trong trạng thái bình thường — không cần hành động thêm."
        tone = "good"
    insight_panel(observations, recommendation, tone=tone)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _render_dashboard(config: dict) -> None:
    import streamlit as st

    runtime_config = dict(config)
    collector = runtime_config.get("collector")
    if runtime_config.get("live_mode") and collector is not None:
        runtime_config["live_info"] = collector.status_snapshot()
        runtime_config["live_frame"] = collector.snapshot()

    render_header(runtime_config)
    live_info = runtime_config.get("live_info")
    if live_info and live_info["status"] == "MISSING CREDENTIALS":
        st.warning(
            "SSI credentials are not configured. Set SSI_CONSUMER_ID and "
            "SSI_CONSUMER_SECRET or provide the ignored local credential file."
        )
    elif live_info and live_info["status"] == "MARKET CLOSED":
        st.info("Vietnamese market is closed. The live collector will wait safely.")
    elif live_info and live_info.get("last_error"):
        st.caption(f"SSI connection detail: {live_info['last_error']}")

    # Plain-language names first, the technical term in parentheses so a
    # reader who knows the jargon can still navigate.
    tabs = st.tabs([
        "🏁 Bắt đầu ở đây",
        "📈 Mức biến động",
        "🛡️ Rủi ro thua lỗ",
        "🎯 Tín hiệu mua/bán",
        "🎛️ Dò tham số",
        "🚨 Bất thường",
    ])

    with tabs[0]:
        render_start_here()
    with tabs[1]:
        tab_explainer("volatility")
        tab_volatility(runtime_config)
        glossary_expander()
    with tabs[2]:
        tab_explainer("risk")
        tab_risk(runtime_config)
        glossary_expander()
    with tabs[3]:
        tab_explainer("signal")
        tab_signal(runtime_config)
        glossary_expander()
    with tabs[4]:
        tab_explainer("optimizer")
        tab_optimizer(runtime_config)
        glossary_expander()
    with tabs[5]:
        tab_explainer("anomaly")
        tab_anomaly(runtime_config)
        glossary_expander()


def main() -> None:
    _configure_page()
    import streamlit as st

    config = render_sidebar()
    if config.get("live_mode"):
        refresh_seconds = float(config.get("refresh_seconds", 1))

        @st.fragment(run_every=refresh_seconds)
        def live_dashboard() -> None:
            _render_dashboard(config)

        live_dashboard()
    else:
        _render_dashboard(config)


if __name__ == "__main__":
    main()
