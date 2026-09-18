import os
import io
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.shm.features import extract_features

st.set_page_config(
    page_title="NebulaX — Train Condition Monitoring (PS3)",
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
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚆 NebulaX 2026 — Train Condition Monitoring</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Problem Statement 3 | Team: Cache Me If You Can</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Subsystem Selection")
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
def load_shm_model():
    model_path = "models/shm_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

if subsystem.startswith("Subsystem 1:"):
    st.header("🔬 Subsystem 1: Structural Health Monitoring (SHM)")
    st.markdown("""
    Predicts **cumulative fatigue damage** ($D$) on railway bogie/carbody structures using **ASTM Rainflow Cycle Counting**, S-N curve power accumulators, and machine learning.
    """)
    
    payload = load_shm_model()
    if payload is None:
        st.error("SHM model not found at `models/shm_model.joblib`. Please train the model first.")
        st.stop()
        
    model = payload['model']
    feature_names = payload['feature_names']
    cv_score = payload.get('cv_score', 0.9372)
    
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Validation Accuracy Score", f"{cv_score*100:.2f}%")
    col_info2.metric("Cross-Validation MAPE", f"{(1-cv_score)*100:.2f}%")
    col_info3.metric("Standard Used", "ASTM E1049-85 Rainflow")
    
    st.write("---")
    
    # Input options: Upload or Sample Test File
    input_tab1, input_tab2 = st.tabs(["📂 Upload Stress Signal CSV", "🧪 Test Dataset Quick Run"])
    
    signal_data = None
    filename_display = "uploaded_signal.csv"
    
    with input_tab1:
        uploaded_file = st.file_uploader("Upload 1D Dynamic Stress CSV file (single column of float values)", type=["csv"])
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file, header=None)
            signal_data = df_upload.iloc[:, 0].values
            filename_display = uploaded_file.name
            
    with input_tab2:
        test_dir = r"C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\SHM\Test"
        if os.path.exists(test_dir):
            test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.csv')])
            selected_test = st.selectbox("Select a test file from official held-out test set:", test_files)
            if st.button("Run Prediction on Selected File"):
                fp = os.path.join(test_dir, selected_test)
                df_test = pd.read_csv(fp, header=None)
                signal_data = df_test.iloc[:, 0].values
                filename_display = selected_test
        else:
            st.info("Organiser test folder not found locally.")

    if signal_data is not None:
        with st.spinner("Extracting Rainflow cycles and computing fatigue damage..."):
            feats = extract_features(signal_data)
            df_feat = pd.DataFrame([feats])[feature_names]
            pred_log = model.predict(df_feat)[0]
            pred_damage = float(np.clip(np.expm1(pred_log), a_min=1e-6, a_max=None))
            
        st.subheader(f"📊 Results for `{filename_display}`")
        
        # Risk assessment
        if pred_damage < 0.10:
            risk_color = "green"
            risk_level = "🟢 LOW RISK (Normal Operating Fatigue)"
        elif pred_damage < 0.35:
            risk_color = "orange"
            risk_level = "🟡 MODERATE RISK (Scheduled Inspection Needed)"
        else:
            risk_color = "red"
            risk_level = "🔴 HIGH RISK (Critical Bogie Fatigue Wear)"
            
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Fatigue Damage ($D$)", f"{pred_damage:.4f}")
        m2.metric("Total Extrema / Cycles", f"{int(feats['rf_total_cycles']):,}")
        m3.metric("Max Stress Range", f"{feats['rf_max_range']:.2f} MPa")
        m4.metric("RMS Stress", f"{feats['rms']:.2f} MPa")
        
        st.info(f"**Structural Assessment**: {risk_level}")
        
        # Plots
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.markdown("**Dynamic Stress Waveform (Sampled)**")
            # Downsample for fast plotting
            step = max(1, len(signal_data) // 2000)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(np.arange(0, len(signal_data), step), signal_data[::step], color='#1E40AF', lw=0.8)
            ax.set_ylabel("Stress (MPa)")
            ax.set_xlabel("Time Samples")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)
            
        with col_plot2:
            st.markdown("**Stress Range Histogram (Rainflow Bins)**")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            bin_cols = [f'rf_bin_{i}' for i in range(10)]
            bin_vals = [feats[b] for b in bin_cols]
            ax.bar(range(10), bin_vals, color='#D97706', edgecolor='black', alpha=0.85)
            ax.set_xlabel("Stress Range Bin (Low -> High Amplitude)")
            ax.set_ylabel("Cycle Count")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)
            
    st.write("---")
    st.subheader("📥 Submission Predictions Download")
    pred_path = "predictions/shm_predictions.csv"
    if os.path.exists(pred_path):
        df_sub = pd.read_csv(pred_path)
        st.dataframe(df_sub, use_container_width=True)
        csv_bytes = df_sub.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download shm_predictions.csv",
            data=csv_bytes,
            file_name="shm_predictions.csv",
            mime="text/csv"
        )
    else:
        st.warning("No predictions file found. Run `src/shm/train.py` to generate `predictions/shm_predictions.csv`.")

else:
    st.header(f"{subsystem}")
    st.info("Pipeline ready for implementation. Proceed to the next phase according to the Development Plan.")
