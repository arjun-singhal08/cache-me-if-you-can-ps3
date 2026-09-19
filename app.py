from __future__ import annotations
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import frontend_utils as utils

# -----------------------------------------------------------------------------
# 1. VISUAL AESTHETIC & THEME: Dark-Mode Industrial Slate
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Senlytics | Fleet Condition Intelligence",
    layout="wide",
    page_icon="🚆",
    initial_sidebar_state="expanded"
)

# Custom Industrial Mission Control CSS
CUSTOM_CSS = """
<style>
/* Root CSS Variables */
:root {
    --bg-dark: #0A0F1D;
    --bg-card: rgba(30, 41, 59, 0.72);
    --border-card: #334155;
    --cyan: #38BDF8;
    --emerald: #10B981;
    --amber: #F59E0B;
    --crimson: #EF4444;
    --text-primary: #F8FAFC;
    --text-muted: #94A3B8;
}

/* Background grid & body */
.stApp {
    background-color: var(--bg-dark);
    background-image: 
        radial-gradient(circle at 12% 8%, rgba(56, 189, 248, 0.09), transparent 35%),
        radial-gradient(circle at 88% 92%, rgba(16, 185, 129, 0.07), transparent 35%),
        linear-gradient(rgba(51, 65, 85, 0.16) 1px, transparent 1px),
        linear-gradient(90deg, rgba(51, 65, 85, 0.16) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px;
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.block-container {
    max-width: 1650px;
    padding-top: 1rem;
    padding-bottom: 4rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* Glassmorphic Industrial Surfaces */
.industrial-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-card);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.45);
    transition: transform 0.18s ease, border-color 0.18s ease;
}

.industrial-card:hover {
    border-color: rgba(56, 189, 248, 0.55);
}

/* Command Header Hero */
.command-hero {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.7) 100%);
    backdrop-filter: blur(14px);
    border: 1px solid #334155;
    border-left: 4px solid var(--cyan);
    border-radius: 16px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 14px 38px rgba(0, 0, 0, 0.4);
}

.command-title {
    font-size: 2.1rem;
    font-weight: 850;
    letter-spacing: -0.035em;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    line-height: 1.1;
}

.command-title span.accent {
    color: var(--cyan);
}

.command-subtitle {
    color: var(--text-muted);
    font-size: 0.88rem;
    font-weight: 500;
    margin-top: 0.35rem;
    letter-spacing: 0.01em;
}

/* Radar & Pulsing Badges */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.78rem;
    font-weight: 750;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.status-online {
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(16, 185, 129, 0.45);
    color: #34D399;
}

.status-alert {
    background: rgba(239, 68, 68, 0.16);
    border: 1px solid rgba(239, 68, 68, 0.5);
    color: #F87171;
}

.status-monitor {
    background: rgba(245, 158, 11, 0.16);
    border: 1px solid rgba(245, 158, 11, 0.5);
    color: #FBBF24;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--emerald);
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: radar-pulse 1.8s infinite cubic-bezier(0.66, 0, 0, 1);
}

@keyframes radar-pulse {
    to {
        box-shadow: 0 0 0 9px rgba(16, 185, 129, 0);
    }
}

/* Metric Pills */
.metric-pill-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.85rem 1.05rem;
    margin-bottom: 0.5rem;
}

.metric-pill-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted);
}

.metric-pill-value {
    font-size: 1.55rem;
    font-weight: 850;
    color: var(--text-primary);
    margin-top: 0.15rem;
    line-height: 1.2;
}

.metric-pill-sub {
    font-size: 0.75rem;
    color: var(--cyan);
    margin-top: 0.2rem;
}

/* Train Consist Schematic */
.consist-container {
    display: flex;
    gap: 0.5rem;
    overflow-x: auto;
    padding: 0.75rem 0.25rem 1.25rem;
}

.consist-car {
    flex: 1;
    min-width: 120px;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.85rem 0.65rem;
    text-align: center;
    position: relative;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.consist-car:hover {
    transform: translateY(-2px);
    border-color: var(--cyan);
}

.consist-car.critical {
    border: 2px solid var(--crimson);
    background: rgba(239, 68, 68, 0.12);
}

.consist-car.monitor {
    border: 1.5px solid var(--amber);
    background: rgba(245, 158, 11, 0.1);
}

.consist-car.nominal {
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.car-badge {
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    display: inline-block;
    margin-top: 0.35rem;
}

/* Automated Work Order Box */
.work-order-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
    border: 1px solid #475569;
    border-left: 5px solid var(--amber);
    border-radius: 14px;
    padding: 1.25rem 1.45rem;
    margin-top: 1.4rem;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5);
}

.work-order-card.p1 {
    border-left-color: var(--crimson);
}

.wo-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(71, 85, 105, 0.6);
    padding-bottom: 0.65rem;
    margin-bottom: 0.85rem;
}

.wo-id {
    font-family: monospace;
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--cyan);
}

.wo-grid {
    display: grid;
    grid-template-columns: 1.2fr 1fr 1fr;
    gap: 1rem;
    margin-bottom: 0.85rem;
}

.wo-field-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    color: var(--text-muted);
    font-weight: 750;
    letter-spacing: 0.06em;
}

.wo-field-val {
    font-size: 0.88rem;
    color: var(--text-primary);
    font-weight: 600;
    margin-top: 0.2rem;
}

.wo-action-box {
    background: rgba(10, 15, 29, 0.7);
    border: 1px dashed #475569;
    border-radius: 8px;
    padding: 0.75rem 0.95rem;
    font-size: 0.86rem;
    line-height: 1.45;
    color: #E2E8F0;
    margin-bottom: 0.85rem;
}

/* Tabs Styling */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid #334155;
    padding: 0.4rem;
    border-radius: 14px;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 0.65rem 1.15rem;
    font-weight: 750;
    color: var(--text-muted);
    transition: all 0.15s ease;
}

[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(56, 189, 248, 0.15) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(56, 189, 248, 0.5) !important;
}

/* Global button styling */
.stButton>button, .stDownloadButton>button {
    border-radius: 10px;
    border: 1px solid #475569;
    font-weight: 750;
    transition: all 0.15s ease;
}

.stButton>button:hover, .stDownloadButton>button:hover {
    border-color: var(--cyan);
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
}

/* File uploader styling */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 23, 42, 0.55);
    border: 1px dashed rgba(56, 189, 248, 0.4);
    border-radius: 14px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Plotly Dark Theme Helper
# -----------------------------------------------------------------------------
def format_fig(fig: go.Figure, height: int = 380, title: Optional[str] = None) -> go.Figure:
    """Applies consistent dark-slate industrial mission control styling to Plotly charts."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 15, 29, 0.45)",
        font=dict(color="#F1F5F9", family="-apple-system, Segoe UI, Roboto, sans-serif"),
        title=dict(text=title, font=dict(size=14, color="#E2E8F0", weight=700)) if title else None,
        margin=dict(l=24, r=24, t=48 if title else 20, b=24),
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="#334155",
            borderwidth=1
        ),
        xaxis=dict(gridcolor="rgba(51, 65, 85, 0.35)", zerolinecolor="rgba(51, 65, 85, 0.5)"),
        yaxis=dict(gridcolor="rgba(51, 65, 85, 0.35)", zerolinecolor="rgba(51, 65, 85, 0.5)")
    )
    return fig

# -----------------------------------------------------------------------------
# Session State Initialization & Benchmark Telemetry Loader
# -----------------------------------------------------------------------------
if "telemetry_loaded" not in st.session_state:
    st.session_state.telemetry_loaded = False
if "benchmark_data" not in st.session_state:
    st.session_state.benchmark_data = None
if "history" not in st.session_state:
    st.session_state.history = []

def trigger_benchmark_load():
    with st.spinner("Hydrating high-fidelity benchmark consist telemetry across all 4 subsystems..."):
        st.session_state.benchmark_data = utils.load_benchmark_data()
        st.session_state.telemetry_loaded = True
        st.session_state.history.insert(0, {
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Action": "⚡ Benchmark Consist Telemetry Ingested",
            "Scope": "4 Subsystems (SHM, ACV, Rail, Door)",
            "Status": "Verified Nominal"
        })
        st.toast("✅ Benchmark Consist Telemetry successfully loaded!", icon="⚡")

# Auto-hydrate benchmark on initial launch for instantaneous showcase experience
if not st.session_state.telemetry_loaded:
    st.session_state.benchmark_data = utils.load_benchmark_data()
    st.session_state.telemetry_loaded = True

bm = st.session_state.benchmark_data

# -----------------------------------------------------------------------------
# SIDEBAR: Consist Health & Fast Control Center
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.75rem'>
            <span style='font-size:1.8rem'>🚆</span>
            <div>
                <div style='font-weight:900;font-size:1.25rem;letter-spacing:-0.03em;color:#FFF'>SENLYTICS</div>
                <div style='font-size:0.72rem;color:#38BDF8;font-weight:700;letter-spacing:0.05em'>FLEET INTELLIGENCE</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='metric-pill-card'>
            <div class='status-pill status-online'><span class='pulse-dot'></span> TELEMETRY SYNCHRONIZED</div>
            <div style='margin-top:0.6rem;font-size:0.8rem;color:#94A3B8'>
                Consist ID: <b style='color:#F8FAFC'>CR400-ALPHA-08</b><br>
                Active Subsystems: <b style='color:#10B981'>4 / 4 ONLINE</b><br>
                Sample Frequency: <b style='color:#F8FAFC'>10,000 Hz / Real-Time</b>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚡ Zero-Friction Demo")
    if st.button("⚡ Load Benchmark Consist Telemetry", type="primary", use_container_width=True, help="Immediately hydrate full pre-packaged telemetry without manual file uploads"):
        trigger_benchmark_load()
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 🎯 Telemetry Mode")
    mode = st.radio("Display Feed", ["Active Telemetry Consist", "Raw Model Diagnostic View"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 🛠 Fleet Dispatch Status")
    st.markdown("""
        <div style='font-size:0.82rem;line-height:1.6;color:#CBD5E1'>
            • <b>Bogie Fatigue</b>: <span style='color:#10B981'>Optimal (D=0.038)</span><br>
            • <b>ACV Refrigerant</b>: <span style='color:#EF4444'>Car 01 Deficit</span><br>
            • <b>Rail Corrugation</b>: <span style='color:#EF4444'>Side I λ=42mm</span><br>
            • <b>Door Obstruction</b>: <span style='color:#F59E0B'>Drag Score 0.84</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Senlytics Autonomous Rail Fleet Condition Intelligence Platform · NebulaX 2026")

# -----------------------------------------------------------------------------
# TOP COMMAND HERO HEADER
# -----------------------------------------------------------------------------
st.markdown("""
    <div class='command-hero'>
        <div>
            <div class='command-title'>
                Senlytics<span class='accent'>.</span> Rail Fleet Condition Intelligence
            </div>
            <div class='command-subtitle'>
                Industrial Mission Control Telemetry · Autonomous Structural, Pneumatic, Track, & Kinematic Diagnostics
            </div>
        </div>
        <div style='text-align:right'>
            <div class='status-pill status-online'><span class='pulse-dot'></span> 4 SUBSYSTEMS MONITORED</div>
            <div style='font-size:0.75rem;color:#94A3B8;margin-top:0.4rem;font-family:monospace'>
                TELEMETRY TIMESTAMP: 2026-09-20 03:15:00 UTC
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SUBSYSTEM TABS
# -----------------------------------------------------------------------------
tab_shm, tab_acv, tab_rail, tab_door, tab_dispatch, tab_history = st.tabs([
    "🏗️ Structural Health (SHM)",
    "❄️ ACV Refrigerant Localization",
    "〰️ Rail Corrugation (Axle-Box)",
    "🚪 Passenger Door Kinematics",
    "📋 Automated Maintenance Work-Orders",
    "↺ Ingestion Log"
])

# =============================================================================
# 1. STRUCTURAL HEALTH MONITORING (SHM) TAB
# =============================================================================
with tab_shm:
    st.markdown("""
        <div class='industrial-card'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                    <h3 style='margin:0;font-size:1.35rem;font-weight:800;color:#F8FAFC'>Bogie Frame Structural Fatigue & Rainflow Monitoring</h3>
                    <div style='color:#94A3B8;font-size:0.85rem;margin-top:0.2rem'>
                        ASTM E1049-85 4-point rainflow cycle counting, Goodman mean-stress boundaries, and multi-exponent Palmgren-Miner cumulative fatigue damage.
                    </div>
                </div>
                <div class='status-pill status-online'>SENSOR HEALTHY · CH-01 STRESS</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Custom File Uploader for SHM
    shm_upload = st.file_uploader("Upload custom dynamic-stress CSV files (.csv)", type=["csv"], accept_multiple_files=True, key="shm_uploader")
    if shm_upload:
        if st.button("🚀 Run Live SHM Inference on Uploads", type="primary", key="shm_run_btn"):
            with st.spinner("Extracting physics features & executing hybrid fatigue ensemble regressor..."):
                t0 = time.perf_counter()
                shm_pred_custom = utils.run_shm(shm_upload)
                # Compute rainflow from first uploaded file
                first_bytes = shm_upload[0].getvalue()
                stress_arr = utils.stress_from_bytes(first_bytes)
                custom_damage = float(shm_pred_custom.iloc[0]["prediction"]) if not shm_pred_custom.empty else 0.05
                bundle_custom = utils.compute_shm_physics_bundle(stress_arr, damage=custom_damage)
                st.session_state.shm_custom = {
                    "pred_df": shm_pred_custom,
                    "bundle": bundle_custom,
                    "selected": shm_upload[0].name
                }
                st.toast(f"SHM analysis completed in {time.perf_counter()-t0:.2f}s", icon="✅")
                
    # Active data source: custom or benchmark
    if "shm_custom" in st.session_state:
        shm_data = st.session_state.shm_custom["bundle"]
        shm_table = st.session_state.shm_custom["pred_df"]
        active_file = st.session_state.shm_custom["selected"]
    else:
        shm_data = bm["shm"]["bundle"]
        shm_table = bm["shm"]["predictions"]
        active_file = bm["shm"]["selected_file"]

    # Top Row: 4 Metric Pills
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
            <div class='metric-pill-card'>
                <div class='metric-pill-label'>Bogie Health Index</div>
                <div class='metric-pill-value' style='color:#10B981'>{shm_data['health_index']:.1f}%</div>
                <div class='metric-pill-sub'>Structural Integrity Level</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
            <div class='metric-pill-card'>
                <div class='metric-pill-label'>Peak Stress</div>
                <div class='metric-pill-value' style='color:#38BDF8'>{shm_data['peak_stress']:.1f} <span style='font-size:0.9rem'>MPa</span></div>
                <div class='metric-pill-sub'>Max Dynamic Tension</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
            <div class='metric-pill-card'>
                <div class='metric-pill-label'>RMS Stress</div>
                <div class='metric-pill-value' style='color:#F59E0B'>{shm_data['rms_stress']:.1f} <span style='font-size:0.9rem'>MPa</span></div>
                <div class='metric-pill-sub'>Root Mean Square Power</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
            <div class='metric-pill-card'>
                <div class='metric-pill-label'>Estimated RUL</div>
                <div class='metric-pill-value' style='color:#38BDF8'>{shm_data['rul_km']:,} <span style='font-size:0.9rem'>km</span></div>
                <div class='metric-pill-sub'>Palmgren-Miner Remaining Life</div>
            </div>
        """, unsafe_allow_html=True)

    # Visualization Row 1: Health Index Gauge & Rainflow Histogram
    col_g1, col_g2 = st.columns([1.1, 1.9])
    
    with col_g1:
        # Gauge Meter for Bogie Frame Fatigue Health Index
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=shm_data["health_index"],
            delta={"reference": 100.0, "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}, "suffix": "%"},
            number={"suffix": "%", "font": {"size": 36, "color": "#F1F5F9"}},
            title={"text": "Bogie Frame Fatigue Health Index", "font": {"size": 14, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": "#38BDF8", "thickness": 0.28},
                "bgcolor": "rgba(30, 41, 59, 0.5)",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.4)"},
                    {"range": [40, 75], "color": "rgba(245, 158, 11, 0.35)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.35)"}
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 3},
                    "thickness": 0.8,
                    "value": 40.0
                }
            }
        ))
        st.plotly_chart(format_fig(gauge_fig, height=340), use_container_width=True)

    with col_g2:
        # Interactive Rainflow Cycle Distribution Bar Chart
        rf_df = shm_data["rainflow_df"]
        rf_fig = px.histogram(
            rf_df,
            x="Stress amplitude",
            y="Cycle count",
            nbins=36,
            histfunc="sum",
            title=f"ASTM E1049-85 Rainflow Cycle Stress-Range Distribution · {active_file}",
            color_discrete_sequence=["#38BDF8"]
        )
        rf_fig.update_layout(
            xaxis_title="Stress Amplitude Δσ/2 (MPa)",
            yaxis_title="Accumulated Rainflow Cycles",
            bargap=0.1
        )
        st.plotly_chart(format_fig(rf_fig, height=340), use_container_width=True)

    # Visualization Row 2: S-N Curve & Goodman Diagram
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        # Goodman Diagram with Boundaries
        g_data = shm_data["goodman"]
        goodman_fig = go.Figure()
        # Scatter of actual cycles
        goodman_fig.add_trace(go.Scatter(
            x=g_data["cycle_means"][:400],
            y=g_data["cycle_amps"][:400],
            mode="markers",
            name="Operational Cycles",
            marker=dict(size=4, color="#38BDF8", opacity=0.65)
        ))
        # Goodman boundary
        goodman_fig.add_trace(go.Scatter(
            x=g_data["mean_axis"],
            y=g_data["goodman_boundary"],
            mode="lines",
            name="Goodman Limit (Linear)",
            line=dict(color="#EF4444", width=2.5, dash="solid")
        ))
        # Gerber boundary
        goodman_fig.add_trace(go.Scatter(
            x=g_data["mean_axis"],
            y=g_data["gerber_boundary"],
            mode="lines",
            name="Gerber Parabola",
            line=dict(color="#F59E0B", width=2, dash="dash")
        ))
        goodman_fig.update_layout(
            title="Goodman Mean-Stress Endurance Boundaries",
            xaxis_title="Mean Stress σ_m (MPa)",
            yaxis_title="Stress Amplitude σ_a (MPa)"
        )
        st.plotly_chart(format_fig(goodman_fig, height=360), use_container_width=True)

    with col_c2:
        # S-N Wohler Fatigue Life Consumption Timeline
        sn_data = shm_data["sn_curve"]
        sn_fig = go.Figure()
        sn_fig.add_trace(go.Scatter(
            x=sn_data["fatigue_life_cycles"],
            y=sn_data["stress_ranges"],
            mode="lines",
            name="S355 Weld S-N Curve (m=3.5)",
            line=dict(color="#10B981", width=2.5)
        ))
        # Operating stress range marker
        sn_fig.add_trace(go.Scatter(
            x=[sn_data["fatigue_life_cycles"][45]],
            y=[sn_data["operating_delta_sigma"]],
            mode="markers+text",
            name="Active Bogie Duty Cycle",
            text=["Current Operational Point"],
            textposition="top right",
            marker=dict(size=12, color="#EF4444", symbol="diamond")
        ))
        sn_fig.update_xaxes(type="log", title="Permissible Cycles to Failure N (log-scale)")
        sn_fig.update_yaxes(title="Stress Range Δσ (MPa)")
        sn_fig.update_layout(title="S-N Curve Fatigue Life Consumption")
        st.plotly_chart(format_fig(sn_fig, height=360), use_container_width=True)

    # File Summary Table & Work Order
    with st.expander("📊 Subsystem File-Level Predictions Table", expanded=False):
        st.dataframe(shm_table, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download Competition Submission (shm_predictions.csv)",
            utils.csv_bytes(shm_table),
            "shm_predictions.csv",
            "text/csv",
            use_container_width=True
        )

    # Automated Industrial Maintenance Work-Order Card
    wo_shm = bm["shm"]["work_order"]
    st.markdown(f"""
        <div class='work-order-card { "p1" if shm_data["health_index"] < 75 else "" }'>
            <div class='wo-header'>
                <div class='wo-id'>🛠️ AUTOMATED WORK-ORDER: {wo_shm['work_order_id']}</div>
                <div class='status-pill { "status-alert" if shm_data["health_index"] < 75 else "status-online" }'>
                    {wo_shm['priority']}
                </div>
            </div>
            <div class='wo-grid'>
                <div>
                    <div class='wo-field-label'>Equipment & Target Node</div>
                    <div class='wo-field-val'>{wo_shm['equipment_id']}</div>
                </div>
                <div>
                    <div class='wo-field-label'>Assigned Depot & Bay</div>
                    <div class='wo-field-val'>{wo_shm['assigned_depot']}</div>
                </div>
                <div>
                    <div class='wo-field-label'>Specialist Crew Dispatched</div>
                    <div class='wo-field-val'>{wo_shm['specialist_team']}</div>
                </div>
            </div>
            <div class='wo-action-box'>
                <b>Standard Operating Procedure (SOP) Action:</b><br>
                {wo_shm['recommended_action']}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "📥 Export Work-Order Telemetry (JSON)",
            utils.work_order_to_json(wo_shm),
            f"{wo_shm['work_order_id']}.json",
            "application/json",
            use_container_width=True
        )
    with col_d2:
        st.download_button(
            "📥 Export Work-Order Summary (CSV)",
            utils.work_order_to_csv(wo_shm),
            f"{wo_shm['work_order_id']}.csv",
            "text/csv",
            use_container_width=True
        )

# =============================================================================
# 2. ACV REFRIGERANT LEAK LOCALIZATION TAB
# =============================================================================
with tab_acv:
    st.markdown("""
        <div class='industrial-card'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                    <h3 style='margin:0;font-size:1.35rem;font-weight:800;color:#F8FAFC'>ACV Refrigerant Leak Localization & Thermal Deficit Analysis</h3>
                    <div style='color:#94A3B8;font-size:0.85rem;margin-top:0.2rem'>
                        8-Car consist thermal schematic, cross-car supply vs return differential (ΔT = T_sup - T_ret), and 5-model calibrated ensemble ranking.
                    </div>
                </div>
                <div class='status-pill status-alert'>PRIMARY LEAK SUSPECT: CAR 01</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Custom File Uploader for ACV
    acv_upload = st.file_uploader("Upload custom Consist telemetry workbook (.xlsx)", type=["xlsx"], key="acv_uploader")
    if acv_upload:
        if st.button("🚀 Run Live ACV Ensemble Inference", type="primary", key="acv_run_btn"):
            with st.spinner("Extracting cross-car thermodynamic differentials & executing 5-model rank ensemble..."):
                t0 = time.perf_counter()
                pred_acv, raw_acv, diag_acv = utils.run_acv(acv_upload)
                bundle_custom = utils.compute_acv_analytics(pred_acv.iloc[0]["ranked_cars"], raw_acv, diag_acv)
                st.session_state.acv_custom = {
                    "pred_df": pred_acv,
                    "bundle": bundle_custom,
                    "selected": acv_upload.name
                }
                st.toast(f"ACV analysis completed in {time.perf_counter()-t0:.2f}s", icon="✅")
                
    if "acv_custom" in st.session_state:
        acv_data = st.session_state.acv_custom["bundle"]
        acv_table = st.session_state.acv_custom["pred_df"]
    else:
        acv_data = bm["acv"]["bundle"]
        acv_table = bm["acv"]["predictions"]

    # Interactive 8-Car Consist Heatmap: Visual Train Schematic
    st.markdown("#### 🚆 Consist Trainset Schematic · Thermal & Refrigerant Leak Risk Status")
    consist_df = acv_data["consist_df"]
    
    cars_html = ["<div class='consist-container'>"]
    for _, r in consist_df.iterrows():
        c_status = r["Status"].lower().replace(" ", "_")
        cls = "critical" if "critical" in c_status else ("monitor" if "monitor" in c_status else "nominal")
        badge_bg = "#EF4444" if cls == "critical" else ("#F59E0B" if cls == "monitor" else "#10B981")
        cars_html.append(f"""
            <div class='consist-car {cls}'>
                <div style='font-size:0.75rem;color:#94A3B8;font-weight:700'>POSITION #{r['Rank']}</div>
                <div style='font-size:1.35rem;font-weight:900;color:#FFF;margin:0.2rem 0'>{r['Car']}</div>
                <div style='font-size:0.8rem;color:#E2E8F0'>ΔT: <b>{r['Delta_T']:.1f}°C</b></div>
                <div style='font-size:0.72rem;color:#94A3B8'>Load: {r['Compressor_Duty']:.0f}%</div>
                <span class='car-badge' style='background:{badge_bg};color:#FFF'>{r['Status']}</span>
            </div>
        """)
    cars_html.append("</div>")
    st.markdown("".join(cars_html), unsafe_allow_html=True)

    # Plotly Line Chart: Cross-Car Supply vs Return Differential (Delta-T)
    dt_df = acv_data["delta_t_series"]
    dt_fig = go.Figure()
    
    for c in consist_df["Car_ID"]:
        width = 3.5 if c == acv_data["primary_suspect"] else 1.2
        color = "#EF4444" if c == acv_data["primary_suspect"] else None
        dt_fig.add_trace(go.Scatter(
            x=dt_df["Time"],
            y=dt_df[f"Car {c}"],
            mode="lines",
            name=f"Car {c}" + (" [Primary Leak]" if c == acv_data["primary_suspect"] else ""),
            line=dict(width=width, color=color)
        ))
        
    # Fleet Mean & Dynamic Thresholds
    dt_fig.add_trace(go.Scatter(
        x=dt_df["Time"],
        y=dt_df["Fleet Mean"],
        mode="lines",
        name="Fleet Mean Baseline",
        line=dict(color="#38BDF8", width=2, dash="dash")
    ))
    dt_fig.add_trace(go.Scatter(
        x=dt_df["Time"],
        y=dt_df["Upper 2-Sigma Threshold"],
        mode="lines",
        name="Upper Control Limit (+2σ)",
        line=dict(color="rgba(245, 158, 11, 0.6)", width=1.5, dash="dot")
    ))
    dt_fig.update_layout(
        title="Cross-Car Cooling Differential Telemetry (ΔT = T_sup - T_ret) with Dynamic Fleet Envelope",
        xaxis_title="Telemetry Timestamp",
        yaxis_title="Temperature Differential ΔT (°C)"
    )
    st.plotly_chart(format_fig(dt_fig, height=390), use_container_width=True)

    # Ranked Car Failure Probability Table with Confidence Progress Bars
    col_t1, col_t2 = st.columns([1.3, 1.0])
    
    with col_t1:
        st.markdown("#### 🎯 Ranked Car Failure Probability")
        rank_display = consist_df.copy()
        rank_display["Leak Probability"] = rank_display["Probability"].map(lambda p: f"{p*100:.1f}%")
        st.dataframe(
            rank_display[["Rank", "Car", "Status", "Delta_T", "Indoor_Temp", "Leak Probability"]],
            use_container_width=True,
            hide_index=True
        )

    with col_t2:
        st.markdown("#### 🤖 5-Model Ensemble Agreement Matrix")
        st.dataframe(acv_data["model_agreement"], use_container_width=True, hide_index=True)

    # Automated Work Order Card for ACV
    wo_acv = bm["acv"]["work_order"]
    st.markdown(f"""
        <div class='work-order-card p1'>
            <div class='wo-header'>
                <div class='wo-id'>🛠️ AUTOMATED WORK-ORDER: {wo_acv['work_order_id']}</div>
                <div class='status-pill status-alert'>{wo_acv['priority']}</div>
            </div>
            <div class='wo-grid'>
                <div>
                    <div class='wo-field-label'>Equipment & Target Node</div>
                    <div class='wo-field-val'>{wo_acv['equipment_id']}</div>
                </div>
                <div>
                    <div class='wo-field-label'>Assigned Depot & Bay</div>
                    <div class='wo-field-val'>{wo_acv['assigned_depot']}</div>
                </div>
                <div>
                    <div class='wo-field-label'>Specialist Crew Dispatched</div>
                    <div class='wo-field-val'>{wo_acv['specialist_team']}</div>
                </div>
            </div>
            <div class='wo-action-box'>
                <b>Standard Operating Procedure (SOP) Action:</b><br>
                {wo_acv['recommended_action']}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col_ad1, col_ad2 = st.columns(2)
    with col_ad1:
        st.download_button(
            "📥 Export ACV Work-Order (JSON)",
            utils.work_order_to_json(wo_acv),
            f"{wo_acv['work_order_id']}.json",
            "application/json",
            use_container_width=True
        )
    with col_ad2:
        st.download_button(
            "📥 Export Competition Submission (acv_predictions.csv)",
            utils.csv_bytes(acv_table),
            "acv_predictions.csv",
            "text/csv",
            use_container_width=True
        )

# =============================================================================
# 3. RAIL CORRUGATION (AXLE-BOX VIBRATION) TAB
# =============================================================================
with tab_rail:
    st.markdown("""
        <div class='industrial-card'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                    <h3 style='margin:0;font-size:1.35rem;font-weight:800;color:#F8FAFC'>Bilateral Axle-Box Rail Corrugation Detection & Order Tracking</h3>
                    <div style='color:#94A3B8;font-size:0.85rem;margin-top:0.2rem'>
                        Speed-dependent spatial order tracking (λ = v/f), bilateral Left-Right axle energy asymmetry, and high-frequency TKEO shock metrics.
                    </div>
                </div>
                <div class='status-pill status-alert'>CRITICAL DEFECT: SIDE I CORRUGATION</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Custom File Uploader for Rail Corrugation
    rail_upload = st.file_uploader("Upload custom axle-box vibration CSV files (.csv)", type=["csv"], accept_multiple_files=True, key="rail_uploader")
    if rail_upload:
        if st.button("🚀 Run Live Rail Corrugation Inference", type="primary", key="rail_run_btn"):
            with st.spinner("Extracting bilateral spatial features & classifying with optimal voting ensemble..."):
                t0 = time.perf_counter()
                pred_rail, diag_rail = utils.run_rail(rail_upload)
                first_pred = str(pred_rail.iloc[0]["prediction"]) if not pred_rail.empty else "Side I"
                first_diag = diag_rail.get(rail_upload[0].name, {})
                probas = first_diag.get("_probas", None)
                bundle_custom = utils.compute_rail_physics_bundle(first_diag, prediction=first_pred, probas=probas)
                st.session_state.rail_custom = {
                    "pred_df": pred_rail,
                    "bundle": bundle_custom,
                    "selected": rail_upload[0].name
                }
                st.toast(f"Rail analysis completed in {time.perf_counter()-t0:.2f}s", icon="✅")

    if "rail_custom" in st.session_state:
        rail_data = st.session_state.rail_custom["bundle"]
        rail_table = st.session_state.rail_custom["pred_df"]
    else:
        rail_data = bm["rail"]["bundle"]
        rail_table = bm["rail"]["predictions"]

    # 3-Class Classification Banner & Sensor Confidence Gauge
    banner_bg = "rgba(239, 68, 68, 0.2)" if "ANOMALY" in rail_data["banner_label"] else "rgba(16, 185, 129, 0.2)"
    banner_border = rail_data["banner_color"]
    st.markdown(f"""
        <div style='background:{banner_bg};border:2px solid {banner_border};border-radius:12px;padding:1rem 1.4rem;display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem'>
            <div>
                <div style='font-size:0.75rem;font-weight:800;letter-spacing:0.08em;color:#94A3B8;text-transform:uppercase'>DIAGNOSTIC CLASSIFICATION RESULT</div>
                <div style='font-size:1.85rem;font-weight:900;color:{banner_border};margin-top:0.2rem'>
                    ● {rail_data['banner_label']}
                </div>
            </div>
            <div style='text-align:right'>
                <div style='font-size:0.75rem;font-weight:800;color:#94A3B8;text-transform:uppercase'>CLASSIFIER CONFIDENCE</div>
                <div style='font-size:1.85rem;font-weight:900;color:#FFF;margin-top:0.2rem'>{rail_data['confidence']:.1f}%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 4 Key Signal Metrics
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.markdown(f"""
            <div class='metric-pill-card'>
                <div class='metric-pill-label'>Kurtosis (Impulsiveness)</div>
                <div class='metric-pill-value' style='color:#F59E0B'>{rail_data['kurtosis']:.2f}</div>
                <div class='metric-pill-sub'>Non-Gaussian Impact Ratio</div>
            </div>
        """, unsafe_allow_html=True)
    with col_r2:
        st.markdown(f"""
            <div class='metric-pill-card'>
                <div class='metric-pill-label'>Crest Factor</div>
                <div class='metric-pill-value' style='color:#38BDF8'>{rail_data['crest_factor']:.2f}</div>
                <div class='metric-pill-sub'>Peak-to-RMS Dynamic Margin</div>
            </div>
        """, unsafe_allow_html=True)
    with col_r3:
        st.markdown(f"""
            <div class='metric-pill-card'>
                <div class='metric-pill-label'>TKEO Energy Power</div>
                <div class='metric-pill-value' style='color:#EF4444'>{rail_data['tkeo_power']:.0f}</div>
                <div class='metric-pill-sub'>Teager-Kaiser Operator Power</div>
            </div>
        """, unsafe_allow_html=True)
    with col_r4:
        st.markdown(f"""
            <div class='metric-pill-card'>
                <div class='metric-pill-label'>Vibration RMS</div>
                <div class='metric-pill-value' style='color:#10B981'>{rail_data['rms_vibration']:.2f} <span style='font-size:0.9rem'>g</span></div>
                <div class='metric-pill-sub'>Axle-Box Acceleration Power</div>
            </div>
        """, unsafe_allow_html=True)

    # Visualizations: Spatial Order Tracking Plot & Bilateral Energy Asymmetry
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        # Spatial Order Tracking Plot (lambda = v / f)
        sp_df = rail_data["spatial_df"]
        sp_fig = go.Figure()
        sp_fig.add_trace(go.Scatter(
            x=sp_df["Wavelength_mm"],
            y=sp_df["PSD_Energy"],
            mode="lines",
            name="Spatial Power Spectral Density",
            line=dict(color="#38BDF8", width=2.5)
        ))
        # Add shaded zones for short, medium, long pitch corrugation
        sp_fig.add_vrect(x0=20, x1=40, fillcolor="rgba(16, 185, 129, 0.12)", line_width=0, annotation_text="Short-Pitch (20-40mm)", annotation_position="top left")
        sp_fig.add_vrect(x0=40, x1=80, fillcolor="rgba(239, 68, 68, 0.16)", line_width=0, annotation_text="Medium-Pitch (40-80mm)", annotation_position="top left")
        sp_fig.add_vrect(x0=80, x1=150, fillcolor="rgba(245, 158, 11, 0.12)", line_width=0, annotation_text="Long-Pitch (80-150mm)", annotation_position="top left")
        
        if rail_data["dominant_wavelength"] > 0:
            sp_fig.add_annotation(
                x=rail_data["dominant_wavelength"],
                y=sp_df["PSD_Energy"].max(),
                text=f"Dominant Corrugation Peak (λ={rail_data['dominant_wavelength']}mm)",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#EF4444",
                arrowsize=1.2,
                font=dict(color="#EF4444", size=12, weight=750)
            )
            
        sp_fig.update_layout(
            title="Speed-Dependent Spatial Order Tracking (λ = v / f)",
            xaxis_title="Defect Wavelength λ (mm)",
            yaxis_title="Spatial Energy Density (PSD)"
        )
        st.plotly_chart(format_fig(sp_fig, height=360), use_container_width=True)

    with col_v2:
        # Bilateral Axle-box Energy Asymmetry Balance Chart
        bi_df = rail_data["bilateral_df"]
        bi_fig = go.Figure()
        bi_fig.add_trace(go.Bar(
            name="Side I (Left Wheelsets)",
            x=bi_df["Axle_Location"],
            y=bi_df["Left_SideI_Energy"],
            marker_color="#EF4444"
        ))
        bi_fig.add_trace(go.Bar(
            name="Side II (Right Wheelsets)",
            x=bi_df["Axle_Location"],
            y=bi_df["Right_SideII_Energy"],
            marker_color="#38BDF8"
        ))
        bi_fig.update_layout(
            title="Bilateral Axle-Box Energy Balance (Side I vs Side II)",
            barmode="group",
            xaxis_title="Bogie Axle Location",
            yaxis_title="Vibration Energy Magnitude"
        )
        st.plotly_chart(format_fig(bi_fig, height=360), use_container_width=True)

    # Automated Work Order Card for Rail Corrugation
    wo_rail = bm["rail"]["work_order"]
    st.markdown(f"""
        <div class='work-order-card { "p1" if "ANOMALY" in rail_data["banner_label"] else "" }'>
            <div class='wo-header'>
                <div class='wo-id'>🛠️ AUTOMATED WORK-ORDER: {wo_rail['work_order_id']}</div>
                <div class='status-pill { "status-alert" if "ANOMALY" in rail_data["banner_label"] else "status-online" }'>
                    {wo_rail['priority']}
                </div>
            </div>
            <div class='wo-grid'>
                <div>
                    <div class='wo-field-label'>Equipment & Target Node</div>
                    <div class='wo-field-val'>{wo_rail['equipment_id']}</div>
                </div>
                <div>
                    <div class='wo-field-label'>Assigned Depot & Bay</div>
                    <div class='wo-field-val'>{wo_rail['assigned_depot']}</div>
                </div>
                <div>
                    <div class='wo-field-label'>Specialist Crew Dispatched</div>
                    <div class='wo-field-val'>{wo_rail['specialist_team']}</div>
                </div>
            </div>
            <div class='wo-action-box'>
                <b>Standard Operating Procedure (SOP) Action:</b><br>
                {wo_rail['recommended_action']}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col_rd1, col_rd2 = st.columns(2)
    with col_rd1:
        st.download_button(
            "📥 Export Rail Work-Order (JSON)",
            utils.work_order_to_json(wo_rail),
            f"{wo_rail['work_order_id']}.json",
            "application/json",
            use_container_width=True
        )
    with col_rd2:
        st.download_button(
            "📥 Export Competition Submission (rail_predictions.csv)",
            utils.csv_bytes(rail_table),
            "rail_predictions.csv",
            "text/csv",
            use_container_width=True
        )

# =============================================================================
# 4. PASSENGER DOOR DIAGNOSTICS TAB
# =============================================================================
with tab_door:
    st.markdown("""
        <div class='industrial-card'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                    <h3 style='margin:0;font-size:1.35rem;font-weight:800;color:#F8FAFC'>Passenger Door Operating Mechanism Diagnostics & Segmentation</h3>
                    <div style='color:#94A3B8;font-size:0.85rem;margin-top:0.2rem'>
                        PELT change-point segmentation (Opening, Dwell, Closing phases), dynamic mechanical drag anomaly scoring, and obstacle resistance profiling.
                    </div>
                </div>
                <div class='status-pill status-monitor'>ABNORMAL MECHANICAL DRAG DETECTED</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Custom File Uploader for Door
    door_upload = st.file_uploader("Upload custom continuous Door telemetry CSV (.csv)", type=["csv"], key="door_uploader")
    if door_upload:
        if st.button("🚀 Run Live Door Cycle Segmentation & Inference", type="primary", key="door_run_btn"):
            with st.spinner("Segmenting cycles with PELT & classifying mechanical friction resistance..."):
                t0 = time.perf_counter()
                pred_door, raw_door = utils.run_door(door_upload)
                bundle_custom = utils.compute_door_physics_bundle(raw_door, pred_door)
                st.session_state.door_custom = {
                    "pred_df": pred_door,
                    "bundle": bundle_custom,
                    "selected": door_upload.name
                }
                st.toast(f"Door analysis completed in {time.perf_counter()-t0:.2f}s", icon="✅")

    if "door_custom" in st.session_state:
        door_data = st.session_state.door_custom["bundle"]
        door_table = st.session_state.door_custom["pred_df"]
    else:
        door_data = bm["door"]["bundle"]
        door_table = bm["door"]["predictions"]

    # 4 Key Door Metrics
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.markdown(f"""
            <div class='metric-pill-card'>
                <div class='metric-pill-label'>Mechanical Drag Score</div>
                <div class='metric-pill-value' style='color:{door_data['drag_color']}'>{door_data['drag_anomaly_score']:.2f}</div>
                <div class='metric-pill-sub'>{door_data['drag_status']}</div>
            </div>
        """, unsafe_allow_html=True)
    with col_d2:
        st.markdown(f"""
            <div class='metric-pill-card'>
                <div class='metric-pill-label'>Peak Motor Current</div>
                <div class='metric-pill-value' style='color:#38BDF8'>{door_data['peak_current_mA']:.0f} <span style='font-size:0.9rem'>mA</span></div>
                <div class='metric-pill-sub'>Inrush & Obstacle Spike</div>
            </div>
        """, unsafe_allow_html=True)
    with col_d3:
        st.markdown(f"""
            <div class='metric-pill-card'>
                <div class='metric-pill-label'>Opening Stroke Duration</div>
                <div class='metric-pill-value' style='color:#10B981'>{door_data['opening_duration_s']:.2f} <span style='font-size:0.9rem'>s</span></div>
                <div class='metric-pill-sub'>Nominal Range: 2.8 - 3.4s</div>
            </div>
        """, unsafe_allow_html=True)
    with col_d4:
        st.markdown(f"""
            <div class='metric-pill-card'>
                <div class='metric-pill-label'>Dwell Platform Stability</div>
                <div class='metric-pill-value' style='color:#38BDF8'>{door_data['dwell_stability_s']:.1f} <span style='font-size:0.9rem'>s</span></div>
                <div class='metric-pill-sub'>Hold Current Ripple &lt; 50mA</div>
            </div>
        """, unsafe_allow_html=True)

    # Time-Series Motor Current & Position Waveform with Highlighted PELT Segmentation
    w_df = door_data["waveform_df"]
    door_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Motor Current Trace
    door_fig.add_trace(
        go.Scatter(
            x=w_df["Time_s"],
            y=w_df["Motor_Current_mA"],
            name="Motor Current (mA)",
            line=dict(color="#38BDF8", width=2.5)
        ),
        secondary_y=False
    )
    # Dynamic Threshold
    door_fig.add_trace(
        go.Scatter(
            x=w_df["Time_s"],
            y=w_df["Dynamic_Threshold_mA"],
            name="Nominal Resistance Boundary",
            line=dict(color="rgba(245, 158, 11, 0.7)", width=1.8, dash="dash")
        ),
        secondary_y=False
    )
    # Door Leaf Position Trace
    door_fig.add_trace(
        go.Scatter(
            x=w_df["Time_s"],
            y=w_df["Door_Position_mm"],
            name="Door Leaf Position (mm)",
            line=dict(color="#10B981", width=2, dash="dot")
        ),
        secondary_y=True
    )
    
    # Shaded regions for PELT segments (Opening, Dwell, Closing)
    door_fig.add_vrect(x0=0, x1=3.2, fillcolor="rgba(56, 189, 248, 0.12)", line_width=0, annotation_text="PHASE 1: OPENING", annotation_position="top left")
    door_fig.add_vrect(x0=3.2, x1=14.5, fillcolor="rgba(71, 85, 105, 0.12)", line_width=0, annotation_text="PHASE 2: DWELL", annotation_position="top left")
    door_fig.add_vrect(x0=14.5, x1=18.0, fillcolor="rgba(239, 68, 68, 0.14)", line_width=0, annotation_text="PHASE 3: CLOSING (DRAG ZONE)", annotation_position="top left")
    
    # Drag annotation
    door_fig.add_annotation(
        x=16.3,
        y=w_df["Motor_Current_mA"].max(),
        text="Abnormal Friction Drag Zone",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#EF4444",
        font=dict(color="#EF4444", size=11, weight=750)
    )
    
    door_fig.update_xaxes(title_text="Cycle Relative Time (seconds)")
    door_fig.update_yaxes(title_text="Motor Current (mA)", secondary_y=False)
    door_fig.update_yaxes(title_text="Door Position (mm)", secondary_y=True)
    door_fig.update_layout(title="Continuous Motor Current & Position Waveform with PELT Change-Point Cycle Segmentation")
    st.plotly_chart(format_fig(door_fig, height=410), use_container_width=True)

    # Door Cycle Segmentation Timeline
    if not door_table.empty and "start_time" in door_table.columns:
        p = door_table.copy()
        p["start"] = utils.parse_door_times(p["start_time"])
        p["end"] = utils.parse_door_times(p["end_time"])
        p["Cycle"] = [f"Cycle {i+1}" for i in range(len(p))]
        colors = {"Normal": "#10B981", "Abnormal resistance": "#EF4444"}
        tl_fig = px.timeline(
            p,
            x_start="start",
            x_end="end",
            y="Cycle",
            color="prediction",
            color_discrete_map=colors,
            title="Continuous Controller Door Cycle Segmentation Timeline"
        )
        tl_fig.update_yaxes(visible=False)
        st.plotly_chart(format_fig(tl_fig, height=280), use_container_width=True)

    # Automated Work Order Card for Door
    wo_door = bm["door"]["work_order"]
    st.markdown(f"""
        <div class='work-order-card { "p1" if door_data["drag_anomaly_score"] > 0.5 else "" }'>
            <div class='wo-header'>
                <div class='wo-id'>🛠️ AUTOMATED WORK-ORDER: {wo_door['work_order_id']}</div>
                <div class='status-pill status-monitor'>{wo_door['priority']}</div>
            </div>
            <div class='wo-grid'>
                <div>
                    <div class='wo-field-label'>Equipment & Target Node</div>
                    <div class='wo-field-val'>{wo_door['equipment_id']}</div>
                </div>
                <div>
                    <div class='wo-field-label'>Assigned Depot & Bay</div>
                    <div class='wo-field-val'>{wo_door['assigned_depot']}</div>
                </div>
                <div>
                    <div class='wo-field-label'>Specialist Crew Dispatched</div>
                    <div class='wo-field-val'>{wo_door['specialist_team']}</div>
                </div>
            </div>
            <div class='wo-action-box'>
                <b>Standard Operating Procedure (SOP) Action:</b><br>
                {wo_door['recommended_action']}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col_dd1, col_dd2 = st.columns(2)
    with col_dd1:
        st.download_button(
            "📥 Export Door Work-Order (JSON)",
            utils.work_order_to_json(wo_door),
            f"{wo_door['work_order_id']}.json",
            "application/json",
            use_container_width=True
        )
    with col_dd2:
        st.download_button(
            "📥 Export Competition Submission (door_predictions.csv)",
            utils.csv_bytes(door_table),
            "door_predictions.csv",
            "text/csv",
            use_container_width=True
        )

# =============================================================================
# 5. AUTOMATED MAINTENANCE WORK-ORDERS & FLEET DISPATCH TAB
# =============================================================================
with tab_dispatch:
    st.markdown("""
        <div class='industrial-card'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                    <h3 style='margin:0;font-size:1.35rem;font-weight:800;color:#F8FAFC'>Consist-Wide Autonomous Maintenance Work-Order Dispatch</h3>
                    <div style='color:#94A3B8;font-size:0.85rem;margin-top:0.2rem'>
                        Unified telemetry synthesis across all subsystems. Direct export to computerized maintenance management systems (CMMS).
                    </div>
                </div>
                <div class='status-pill status-online'>DISPATCH READY · 4 WORK ORDERS ACTIVE</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    all_wo = bm["all_work_orders"]
    wo_df = pd.DataFrame([
        {
            "Work Order ID": w["work_order_id"],
            "Subsystem": w["subsystem"],
            "Priority": w["priority"],
            "Equipment Node": w["equipment_id"],
            "Assigned Depot": w["assigned_depot"],
            "Diagnosis": w["diagnosis"]
        }
        for w in all_wo
    ])
    st.dataframe(wo_df, use_container_width=True, hide_index=True)
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button(
            "📥 Export Complete Consist Work-Orders (JSON)",
            json.dumps(all_wo, indent=2, default=str),
            "consist_maintenance_work_orders.json",
            "application/json",
            use_container_width=True
        )
    with col_e2:
        st.download_button(
            "📥 Export Complete Consist Work-Orders (CSV)",
            wo_df.to_csv(index=False),
            "consist_maintenance_work_orders.csv",
            "text/csv",
            use_container_width=True
        )

# =============================================================================
# 6. INGESTION & AUDIT LOG TAB
# =============================================================================
with tab_history:
    st.markdown("### ↺ Ingestion & Operational Audit Log")
    if st.session_state.history:
        h_df = pd.DataFrame(st.session_state.history)
        st.dataframe(h_df, use_container_width=True, hide_index=True)
        if st.button("Clear Session Log"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No session operations logged yet.")

# Footer
st.markdown("""
    <div style='margin-top:3.5rem;padding-top:1rem;border-top:1px solid #334155;text-align:center;color:#64748B;font-size:0.82rem'>
        <b>Senlytics: Autonomous Rail Fleet Condition Intelligence Platform</b> · Industrial Mission Control<br>
        Cache Me If You Can · LTA NebulaX 2026 Hackathon
    </div>
""", unsafe_allow_html=True)
