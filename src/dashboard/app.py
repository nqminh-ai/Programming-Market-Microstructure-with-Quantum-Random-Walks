"""Interactive QRW Market Microstructure Dashboard.

Upgraded dashboard with Plotly charts, dark theme, side-by-side comparison,
data uploader, plain-English insights, preset scenarios, and sensitivity
analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.coin_operators import biased_coin, hadamard_coin
from src.models.qrw_core import DensityMatrixQRW

# ---------------------------------------------------------------------------
# Theme & constants
# ---------------------------------------------------------------------------
DARK_BG = "#0e1117"
CARD_BG = "#1a1d23"
ACCENT = "#00d4aa"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#4ecdc4"
TEXT_COLOR = "#e0e0e0"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=DARK_BG,
    plot_bgcolor="#161b22",
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    margin=dict(l=50, r=30, t=45, b=40),
)

PRESETS = {
    "High Volatility Day": {
        "gamma": 0.15, "alpha": 1.6, "coin": "Hadamard", "steps": 80,
        "description": "Ngày thị trường biến động mạnh — decoherence thấp, "
                       "QRW giữ coherence lâu → phân phối rộng (bimodal).",
    },
    "Sideways / Quiet Day": {
        "gamma": 2.5, "alpha": 0.3, "coin": "Hadamard", "steps": 60,
        "description": "Ngày đi ngang, ít biến động — decoherence cao, "
                       "QRW nhanh chóng suy biến thành CRW cổ điển.",
    },
    "Strong Directional Trend": {
        "gamma": 0.8, "alpha": 1.8, "coin": "Biased", "steps": 100,
        "description": "Xu hướng mạnh (pump hoặc dump) — coin lệch, "
                       "phân phối QRW bất đối xứng rõ rệt.",
    },
}

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1a1d23 0%, #21252d 100%);
        border: 1px solid #2d333b;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4aa;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-delta {
        font-size: 0.8rem;
        color: #ff6b6b;
    }

    .insight-box {
        background: linear-gradient(135deg, #1a2332 0%, #1a1d23 100%);
        border-left: 4px solid #00d4aa;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        color: #c9d1d9;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    }

    .stSelectbox > div > div { border-color: #2d333b; }
</style>
"""


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def run_qrw(gamma: float, alpha: float, coin_type: str, n_steps: int):
    """Run a QRW simulation and return history + variances."""
    coin = hadamard_coin() if coin_type == "Hadamard" else biased_coin(
        np.pi / 4.0 * alpha
    )
    walk = DensityMatrixQRW(2 * n_steps + 3, coin=coin)
    history, variances = [], []
    for _ in range(n_steps):
        prob = walk.step_with_decoherence(gamma)
        history.append(prob)
        mean = float(prob @ walk.positions)
        variances.append(float(prob @ (walk.positions - mean) ** 2))
    return walk, history, variances


def run_crw(n_steps: int, n_paths: int = 5000):
    """Run classical random walk simulations for comparison."""
    steps = np.random.choice([-1, 1], size=(n_paths, n_steps))
    positions = np.cumsum(steps, axis=1)
    # Build probability distribution at final step matching QRW lattice range
    final_positions = positions[:, -1]
    pos_range = np.arange(-n_steps, n_steps + 1)
    counts = np.zeros(len(pos_range))
    for i, p in enumerate(pos_range):
        counts[i] = np.sum(final_positions == p)
    prob = counts / n_paths
    variances = [float(np.var(positions[:, :t + 1][:, -1]))
                 for t in range(n_steps)]
    return pos_range, prob, variances


def generate_insight(gamma: float, alpha: float, coin_type: str,
                     qrw_var: float, crw_var: float) -> str:
    """Generate plain-English insight based on current parameters."""
    parts = []

    if gamma < 0.3:
        parts.append(
            "🟢 <b>Decoherence rất thấp</b> — QRW giữ được tính lượng tử "
            "(coherence) mạnh. Phân phối vị trí sẽ mang dấu ấn bimodal đặc "
            "trưng của quantum walk, khác hẳn so với classical."
        )
    elif gamma < 1.5:
        parts.append(
            "🟡 <b>Decoherence trung bình</b> — QRW bắt đầu mất coherence. "
            "Phân phối dần chuyển từ bimodal sang dạng bell-curve, vẫn giữ "
            "được một phần tính chất lượng tử."
        )
    else:
        parts.append(
            "🔴 <b>Decoherence cao</b> — QRW gần như suy biến thành CRW "
            "cổ điển. Đây giống với thị trường hiệu quả (EMH) nơi "
            "thông tin bị nhiễu mạnh."
        )

    ratio = qrw_var / max(crw_var, 1e-12)
    if ratio > 1.5:
        parts.append(
            f"📊 Phương sai QRW gấp <b>{ratio:.1f}×</b> CRW → thị trường "
            "có thể đang trong giai đoạn biến động ballistic (không phải "
            "diffusive)."
        )
    elif ratio > 0.8:
        parts.append(
            f"📊 Phương sai QRW ≈ CRW ({ratio:.2f}×) → hai mô hình cho "
            "kết quả tương tự, decoherence đã triệt tiêu phần lớn hiệu ứng "
            "lượng tử."
        )
    else:
        parts.append(
            f"📊 Phương sai QRW thấp hơn CRW ({ratio:.2f}×) → interference "
            "destructive đang giữ QRW 'gọn' hơn classical walk."
        )

    return "<br>".join(parts)


def sensitivity_heatmap(alpha_val: float, coin_type: str, n_steps: int):
    """Compute variance ratio heatmap over gamma × alpha grid."""
    gammas = np.linspace(0.05, 3.0, 12)
    alphas = np.linspace(0.2, 2.0, 10)
    z = np.zeros((len(gammas), len(alphas)))

    # Baseline CRW variance at n_steps
    crw_var = float(n_steps)  # CRW Var(X_t) = t

    for i, g in enumerate(gammas):
        for j, a in enumerate(alphas):
            _, _, var_list = run_qrw(g, a, coin_type, n_steps)
            z[i, j] = var_list[-1] / crw_var if crw_var > 0 else 0.0

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{a:.1f}" for a in alphas],
        y=[f"{g:.2f}" for g in gammas],
        colorscale="Viridis",
        colorbar=dict(title="Var(QRW)/Var(CRW)"),
        hovertemplate="α=%{x}<br>γ=%{y}<br>Ratio=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Sensitivity: Variance Ratio (QRW/CRW)",
        xaxis_title="Alpha (coin angle multiplier)",
        yaxis_title="Gamma (decoherence)",
        **PLOTLY_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Streamlit dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title="QRW Market Microstructure",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("## ⚛️ QRW Controls")
        st.caption(
            "Exploratory interface — current results do not establish "
            "QRW superiority over classical baselines."
        )

        st.markdown("---")
        st.markdown("### 📋 Preset Scenarios")
        preset_choice = st.selectbox(
            "Load a preset",
            ["Custom"] + list(PRESETS.keys()),
            index=0,
        )
        if preset_choice != "Custom":
            p = PRESETS[preset_choice]
            gamma = st.slider("Decoherence γ", 0.0, 5.0, p["gamma"], 0.05)
            alpha = st.slider("Coin angle multiplier α", 0.1, 2.0,
                              p["alpha"], 0.05)
            coin_type = p["coin"]
            st.selectbox("Coin", ("Hadamard", "Biased"),
                         index=0 if p["coin"] == "Hadamard" else 1,
                         disabled=True, key="coin_preset")
            n_steps = st.slider("Steps", 10, 150, p["steps"], 5)
            st.info(f"💡 {p['description']}")
        else:
            gamma = st.slider("Decoherence γ", 0.0, 5.0, 0.5, 0.05)
            alpha = st.slider("Coin angle multiplier α", 0.1, 2.0, 1.0, 0.05)
            coin_type = st.selectbox("Coin", ("Hadamard", "Biased"))
            n_steps = st.slider("Steps", 10, 150, 60, 5)

        st.markdown("---")
        st.markdown("### 📂 Upload Data")
        uploaded = st.file_uploader(
            "Drop a raw tick CSV here",
            type=["csv", "csv.gz"],
            help="Upload a raw tick file to process through the causal "
                 "pipeline. Must contain: trade_id, price, quantity, "
                 "timestamp, is_buyer_maker columns.",
        )
        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded)
                from src.data.tick_processor import TickProcessor
                processor = TickProcessor()
                processed, report = processor.process(raw_df)
                st.success(
                    f"✅ Processed {report['input_records']:,} → "
                    f"{report['output_records']:,} ticks "
                    f"({report['removed_fraction']:.2%} removed)"
                )
            except Exception as e:
                st.error(f"❌ Processing error: {e}")

    # ---- Title ----
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>"
        "⚛️ Quantum Random Walk — Market Microstructure</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#8b949e; margin-top:0;'>"
        "Interactive explorer for QRW vs Classical Random Walk behavior"
        "</p>",
        unsafe_allow_html=True,
    )

    # ---- Run simulations ----
    walk, history, qrw_variances = run_qrw(gamma, alpha, coin_type, n_steps)
    crw_positions, crw_prob, crw_variances = run_crw(n_steps)

    qrw_final_var = qrw_variances[-1] if qrw_variances else 0.0
    crw_final_var = crw_variances[-1] if crw_variances else 0.0

    # ---- Metric Cards ----
    scorecard_path = ROOT / "results" / "scorecard.csv"
    brier_score = "N/A"
    qrw_rank = "N/A"
    top_model = "N/A"
    if scorecard_path.exists():
        sc = pd.read_csv(scorecard_path)
        qrw_row = sc[sc["model"].str.contains("QRW", case=False)]
        top_row = sc[sc["overall_rank"] == 1.0]
        if not qrw_row.empty:
            qrw_rank = f"#{int(qrw_row.iloc[0]['overall_rank'])}"
            brier_score = f"{qrw_row.iloc[0]['mean_rank']:.2f}"
        if not top_row.empty:
            top_model = top_row.iloc[0]["model"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">QRW Overall Rank</div>'
            f'<div class="metric-value">{qrw_rank}</div>'
            f'<div class="metric-delta">of 6 models</div>'
            f'</div>', unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Mean Rank Score</div>'
            f'<div class="metric-value">{brier_score}</div>'
            f'<div class="metric-delta">lower is better</div>'
            f'</div>', unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Top Model</div>'
            f'<div class="metric-value" style="font-size:1.4rem">'
            f'{top_model}</div>'
            f'<div class="metric-delta">Phase 5 scorecard</div>'
            f'</div>', unsafe_allow_html=True,
        )
    with col4:
        ratio = qrw_final_var / max(crw_final_var, 1e-12)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Var(QRW) / Var(CRW)</div>'
            f'<div class="metric-value">{ratio:.2f}×</div>'
            f'<div class="metric-delta">at t={n_steps}</div>'
            f'</div>', unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Plain English Insight ----
    insight = generate_insight(
        gamma, alpha, coin_type, qrw_final_var, crw_final_var
    )
    st.markdown(
        f'<div class="insight-box">{insight}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Charts Row 1: Distribution + Variance ----
    chart1, chart2 = st.columns(2)

    with chart1:
        fig = go.Figure()
        # QRW distribution
        fig.add_trace(go.Scatter(
            x=walk.positions, y=history[-1],
            mode="lines", name="QRW",
            line=dict(color=ACCENT, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.15)",
        ))
        # CRW distribution
        fig.add_trace(go.Scatter(
            x=crw_positions, y=crw_prob,
            mode="lines", name="CRW",
            line=dict(color=ACCENT2, width=2, dash="dash"),
        ))
        fig.update_layout(
            title=f"Position Probability at t={n_steps}",
            xaxis_title="Position",
            yaxis_title="Probability",
            legend=dict(x=0.02, y=0.98),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        steps_arr = np.arange(1, n_steps + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps_arr, y=qrw_variances,
            mode="lines", name="QRW",
            line=dict(color=ACCENT, width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=steps_arr, y=crw_variances,
            mode="lines", name="CRW (simulated)",
            line=dict(color=ACCENT2, width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=steps_arr, y=steps_arr.astype(float),
            mode="lines", name="CRW (theoretical: t)",
            line=dict(color="#666", width=1, dash="dot"),
        ))
        fig.update_layout(
            title="Variance Growth: QRW vs CRW",
            xaxis_title="Step",
            yaxis_title="Variance",
            legend=dict(x=0.02, y=0.98),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tabs: Scorecard / Sensitivity / Raw Data ----
    tab1, tab2, tab3 = st.tabs([
        "📊 Phase 5 Scorecard", "🔥 Sensitivity Analysis", "📋 Raw Data",
    ])

    with tab1:
        if scorecard_path.exists():
            sc = pd.read_csv(scorecard_path)
            # Horizontal bar chart of overall ranks
            sc_sorted = sc.sort_values("mean_rank")
            colors = [
                ACCENT if "QRW" in m else "#4a5568" for m in sc_sorted["model"]
            ]
            fig = go.Figure(go.Bar(
                y=sc_sorted["model"],
                x=sc_sorted["mean_rank"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.2f}" for v in sc_sorted["mean_rank"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Model Rankings (lower = better)",
                xaxis_title="Mean Rank",
                yaxis=dict(autorange="reversed"),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                sc.style.highlight_min(subset=["mean_rank"], color="#1a3a2a"),
                use_container_width=True,
            )
        else:
            st.warning("Run Phase 5 pipeline to populate the scorecard.")

    with tab2:
        st.markdown(
            "Heatmap cho thấy tỉ lệ **Var(QRW) / Var(CRW)** thay đổi "
            "theo γ (decoherence) và α (coin sensitivity). "
            "Vùng sáng = QRW variance vượt trội (ballistic), "
            "vùng tối = QRW bị suppress (interference destructive)."
        )
        with st.spinner("Computing sensitivity grid..."):
            fig = sensitivity_heatmap(alpha, coin_type, min(n_steps, 40))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        benchmark_path = ROOT / "results" / "benchmark_results.csv"
        if benchmark_path.exists():
            bench = pd.read_csv(benchmark_path)
            st.dataframe(bench, use_container_width=True)
        else:
            st.warning("Run Phase 4/5 to populate benchmark results.")

        comparison_path = ROOT / "results" / "final_comparison_table.csv"
        if comparison_path.exists():
            comp = pd.read_csv(comparison_path)
            st.subheader("Final Comparison Table")
            st.dataframe(comp, use_container_width=True)


if __name__ == "__main__":
    main()
