import os
import io
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.shm.service import SHMService

st.set_page_config(
    page_title="SHM Engineer Assistant — NebulaX 2026",
    page_icon="🚆",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .report-box {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 18px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚆 NebulaX 2026 — SHM Engineer Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Train Condition Monitoring (PS3) | Team: Cache Me If You Can</div>', unsafe_allow_html=True)

# Subsystem Selector
subsystem = st.sidebar.selectbox(
    "Choose Subsystem",
    [
        "Subsystem 1: Structural Health Monitoring (SHM)",
        "Subsystem 2: ACV (Air Conditioning & Vent)",
        "Subsystem 3: Rail Corrugation",
        "Subsystem 4: Door Anomaly Detection"
    ]
)

@st.cache_resource
def get_shm_service():
    return SHMService()

service = get_shm_service()

if subsystem.startswith("Subsystem 1:"):
    st.markdown("""
    Welcome to the **SHM Engineer Assistant**. This dashboard performs automated **signal profiling, ASTM Rainflow cycle counting, cumulative fatigue damage regression**, and generates an **engineering diagnostic report** for train bogies and structural carbodies.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Validation Accuracy Score", f"{service.cv_score*100:.2f}%")
    col2.metric("Cross-Validation MAPE", f"{(1-service.cv_score)*100:.2f}%")
    col3.metric("Standard Used", "ASTM E1049-85 Rainflow")
    
    st.write("---")
    
    if not service.is_model_loaded():
        st.warning("⚠️ Model weights not found on disk.")
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training ExtraTrees model..."):
                from src.shm.train import train_shm
                train_shm()
                st.success("Model trained! Reloading...")
                st.rerun()
        st.stop()
        
    input_tab1, input_tab2 = st.tabs(["📂 Upload Stress Signal CSV", "🧪 Held-Out Test Set Quick Run"])
    
    selected_input = None
    input_filename = "uploaded_stress_data.csv"
    
    with input_tab1:
        uploaded_file = st.file_uploader("Upload 1D Dynamic Stress CSV file", type=["csv"])
        if uploaded_file is not None:
            selected_input = uploaded_file
            input_filename = uploaded_file.name
            
    with input_tab2:
        test_dir = r"C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\SHM\Test"
        if os.path.exists(test_dir):
            test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.csv')])
            chosen_file = st.selectbox("Select test signal:", test_files)
            if st.button("Run Analysis on Selected Test File"):
                selected_input = os.path.join(test_dir, chosen_file)
                input_filename = chosen_file
        else:
            st.info("Organiser test folder not found locally. Please use the upload tab above.")

    if selected_input is not None:
        with st.spinner("Analyzing signal dynamics & running Rainflow cycle decomposition..."):
            res = service.analyze_signal(selected_input, filename=input_filename)
            
        if not res['success']:
            st.error(f"❌ Validation Error: {res['error']}")
        else:
            prof = res['profile']
            feats = res['features']
            damage = res['predicted_damage']
            signal = res['signal']
            
            st.success(f"✅ Analysis Complete for `{input_filename}` ({prof['n_samples']:,} data points)")
            
            # 1. Prediction & Key Metrics
            st.subheader("🎯 1. Fatigue Damage Estimate & Key Indicators")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Damage ($D$)", f"{damage:.4f}")
            m2.metric("Total Stress Cycles", f"{int(feats['rf_total_cycles']):,}")
            m3.metric("Peak Stress Range", f"{prof['peak_to_peak']:.2f} MPa")
            m4.metric("RMS Stress", f"{prof['rms_stress']:.2f} MPa")
            
            # 2. Engineering Explanation Report
            st.subheader("📋 2. Engineer Diagnostic Report")
            st.markdown(f"<div class='report-box'>{res['explanation']}</div>", unsafe_allow_html=True)
            
            # 3. Interactive Signal Plots
            st.subheader("📈 3. Signal Waveform & Cycle Distribution")
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                st.markdown("**Dynamic Stress Time Series (Sampled Waveform)**")
                step = max(1, len(signal) // 2000)
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.plot(np.arange(0, len(signal), step), signal[::step], color='#1E40AF', lw=0.8)
                ax.set_ylabel("Stress (MPa)")
                ax.set_xlabel("Time Step")
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
                plt.close(fig)
                
            with col_p2:
                st.markdown("**Stress Amplitude Histogram (Rainflow Bins)**")
                fig, ax = plt.subplots(figsize=(7, 3.5))
                bin_cols = [f'rf_bin_{i}' for i in range(10)]
                bin_vals = [feats[b] for b in bin_cols]
                ax.bar(range(10), bin_vals, color='#D97706', edgecolor='black', alpha=0.85)
                ax.set_xlabel("Stress Range Bin (Low -> High Amplitude)")
                ax.set_ylabel("Cycle Count")
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
                plt.close(fig)
                
            # 4. Feature Summary Table
            with st.expander("🔍 View All 53 Extracted Engineering Features"):
                df_disp = pd.DataFrame([feats]).T.reset_index()
                df_disp.columns = ["Feature Name", "Value"]
                st.dataframe(df_disp, use_container_width=True)

    st.write("---")
    st.subheader("📥 4. Submission File Download")
    pred_path = "predictions/shm_predictions.csv"
    if os.path.exists(pred_path):
        df_sub = pd.read_csv(pred_path)
        st.dataframe(df_sub, use_container_width=True)
        csv_bytes = df_sub.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Official shm_predictions.csv",
            data=csv_bytes,
            file_name="shm_predictions.csv",
            mime="text/csv"
        )
else:
    st.header(f"{subsystem}")
    st.info("Subsystem ready for implementation according to the PS3 Development Plan.")
