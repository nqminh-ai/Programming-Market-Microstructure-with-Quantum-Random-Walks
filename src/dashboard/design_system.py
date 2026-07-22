"""Design System — QRW Financial Platform.

Color tokens, CSS, and Plotly template shared across all dashboard tabs.
"""

# Color Tokens
COLORS = {
    "bg_primary":    "#080C12",
    "bg_secondary":  "#0D1420",
    "bg_tertiary":   "#121A28",
    "bg_hover":      "#1A2438",
    "border_subtle": "#1C2A3D",
    "border_active": "#2A3F5C",
    "text_primary":  "#E8F0FA",
    "text_secondary": "#8FA3BE",
    "text_muted":    "#4A6080",
    "text_mono":     "#7EC8C8",
    "accent_cyan":   "#00D4FF",
    "accent_green":  "#00E676",
    "accent_red":    "#FF4444",
    "accent_yellow": "#FFB300",
    "accent_purple": "#B388FF",
    "chart_qrw":     "#00D4FF",
    "chart_crw":     "#4A6080",
    "chart_garch":   "#7B68EE",
    "chart_gbm":     "#3A5068",
    "chart_actual":  "#E8F0FA",
}

PLOTLY_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,20,32,1)",
    font=dict(
        family="JetBrains Mono, Inter, monospace",
        color="#E8F0FA",
    ),
    margin=dict(l=50, r=30, t=45, b=40),
)

# Global CSS
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Inter:wght@300;400;500;600&display=swap');

.stApp { background-color: #080C12; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

[data-testid="stSidebar"] {
    background-color: #0D1420;
    border-right: 1px solid #1C2A3D;
}

.kpi-card {
    background: #0D1420;
    border: 1px solid #1C2A3D;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    height: 100%;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #2A3F5C; }
.kpi-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4A6080;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #00D4FF;
    line-height: 1;
}
.kpi-delta { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; margin-top: 0.3rem; }
.kpi-delta.pos { color: #00E676; }
.kpi-delta.neg { color: #FF4444; }
.kpi-delta.neu { color: #8FA3BE; }

.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4A6080;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1C2A3D;
    margin-bottom: 1rem;
}

.signal-badge { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; padding: 0.5rem 1.2rem; border-radius: 3px; letter-spacing: 0.08em; }
.signal-buy  { background: rgba(0,230,118,0.12); color: #00E676; border: 1px solid rgba(0,230,118,0.3); }
.signal-sell { background: rgba(255,68,68,0.12);  color: #FF4444; border: 1px solid rgba(255,68,68,0.3); }
.signal-hold { background: rgba(143,163,190,0.1); color: #8FA3BE; border: 1px solid rgba(143,163,190,0.2); }

.alert-normal   { border-left: 3px solid #00E676; background: rgba(0,230,118,0.05); padding: 0.8rem 1rem; border-radius: 0 3px 3px 0; }
.alert-warning  { border-left: 3px solid #FFB300; background: rgba(255,179,0,0.05);  padding: 0.8rem 1rem; border-radius: 0 3px 3px 0; }
.alert-critical { border-left: 3px solid #FF4444; background: rgba(255,68,68,0.08);  padding: 0.8rem 1rem; border-radius: 0 3px 3px 0; }

.dataframe { font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important; }
.dataframe th { background: #121A28 !important; color: #4A6080 !important; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.65rem !important; }

#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080C12; }
::-webkit-scrollbar-thumb { background: #1C2A3D; border-radius: 3px; }
</style>
"""
