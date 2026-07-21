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
        f"⚠ No real artifact found — showing synthetic demo data. Run:\n\n"
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

        date = st.date_input(
            "Analysis Date",
            value=datetime(2026, 6, 12).date(),
            key="sidebar_date",
        )
        asset = st.selectbox(
            "Asset",
            ["VN30F1M", "VNINDEX", "Synthetic"],
            key="sidebar_asset",
        )

        st.markdown("---")
        st.markdown(
            '<div class="kpi-label">Live Data</div>',
            unsafe_allow_html=True,
        )
        requested_live = st.toggle(
            "Live SSI Mode",
            value=False,
            disabled=asset == "Synthetic",
            key="ssi_live_mode",
        )
        live_mode = bool(requested_live and asset != "Synthetic")
        refresh_seconds = st.slider(
            "Refresh interval (seconds)",
            1,
            10,
            1,
            disabled=not live_mode,
            key="ssi_refresh_seconds",
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
            '<div class="kpi-label">Forecast Controls</div>',
            unsafe_allow_html=True,
        )
        window = st.slider(
            "Rolling window (ticks)", 5, 30, 15, 5, key="vol_window"
        )
        show_garch = st.checkbox("Show GARCH", True, key="show_garch")
        show_real = st.checkbox("Show Realized Vol", True, key="show_real")
        theta_buy = st.slider(
            "θ_buy", 0.50, 0.80, 0.60, 0.01, key="theta_buy"
        )
        theta_sell = st.slider(
            "θ_sell", 0.50, 0.80, 0.60, 0.01, key="theta_sell"
        )

        st.markdown("---")
        st.markdown(
            '<div class="kpi-label">Module Status</div>',
            unsafe_allow_html=True,
        )
        results_dir = ROOT / "results" / "track_a"
        modules = {
            "A1 VOL": (results_dir / "vol_metrics.json").exists(),
            "A2 RISK": (results_dir / "risk_paths.parquet").exists(),
            "A3 SIG": (results_dir / "signal_log.parquet").exists(),
            "A4 OPT": (results_dir / "optimizer_params.json").exists(),
            "A5 ANM": (results_dir / "anomaly_log.parquet").exists(),
        }
        status_html = '<div style="display:flex; flex-direction:column; gap:0.4rem; margin-top:0.5rem;">'
        for name, ready in modules.items():
            color = "#00E676" if ready else "#4A6080"
            status = "READY" if ready else "PENDING"
            status_html += (
                f'<div style="font-family:\'JetBrains Mono\'; font-size:0.75rem; color:{color};">'
                f"{name}: {status}"
                f"</div>"
            )
        status_html += "</div>"
        st.markdown(status_html, unsafe_allow_html=True)

        st.markdown("---")
        ready_count = sum(modules.values())
        st.caption(
            "Track A — QRW Breadth Platform  \n"
            f"{ready_count}/5 modules ready · Demo mode {'ON' if DEMO_MODE else 'OFF'}"
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
        status = "READY" if ready else "PENDING"
        pills_html += (
            f'<span style="background:#0D1420; border:1px solid {border_color}; '
            f'color:{text_color}; padding:0.2rem 0.6rem; border-radius:3px; '
            f'font-size:0.7rem; font-family:\'JetBrains Mono\';">'
            f"{name} {status}</span>"
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
    kpi_card(c1, "QRW Vol Forecast", f"{current_qrw*100:.1f}%", current_qrw - current_real, "#00D4FF", "vs realized")
    kpi_card(
        c2,
        "GARCH Forecast",
        f"{current_garch*100:.1f}%" if current_garch is not None else "N/A",
        current_garch - current_real if current_garch is not None else None,
        "#B388FF",
        "vs realized",
    )
    kpi_card(c3, "Realized Vol", f"{current_real*100:.1f}%", color="#E8F0FA")
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


# ---------------------------------------------------------------------------
# Tab A4 — Optimizer
# ---------------------------------------------------------------------------

def tab_optimizer(config: dict) -> None:
    import streamlit as st

    from src.strategy.optimizer import QRWStrategyOptimizer

    section_header("STRATEGY OPTIMIZER — MODULE A4")
    controls = st.columns([2, 2, 1])
    with controls[0]:
        objective = st.selectbox(
            "Objective", ["sharpe", "hit_rate", "profit_factor"], key="opt_obj"
        )
    with controls[1]:
        st.toggle("Regime-aware", True, key="regime_aware")
    with controls[2]:
        run_optimization = st.button(
            "Run Optimization", type="primary", key="run_opt"
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
            "sharpe": float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl))),
            "hit_rate": float((pnl > 0).mean()),
            "max_drawdown": float((equity - equity.cummax()).min()),
            "profit_factor": float(pnl[pnl > 0].sum() / -pnl[pnl < 0].sum()),
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
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "OOS Sharpe", f"{float(oos_metrics.get('sharpe', 0)):.2f}", color=COLORS["accent_cyan"])
    kpi_card(c2, "Hit Rate", f"{float(oos_metrics.get('hit_rate', 0))*100:.1f}%", color=COLORS["accent_green"])
    kpi_card(c3, "Max Drawdown", f"{float(oos_metrics.get('max_drawdown', 0))*100:.2f}%", color=COLORS["accent_red"])
    kpi_card(c4, "Profit Factor", f"{float(oos_metrics.get('profit_factor', 0)):.2f}", color=COLORS["accent_yellow"])


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

    tabs = st.tabs([
        "Volatility",
        "Risk",
        "Signal",
        "Optimizer",
        "Anomaly",
    ])

    with tabs[0]:
        tab_volatility(runtime_config)
    with tabs[1]:
        tab_risk(runtime_config)
    with tabs[2]:
        tab_signal(runtime_config)
    with tabs[3]:
        tab_optimizer(runtime_config)
    with tabs[4]:
        tab_anomaly(runtime_config)


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
