import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import ACVDataLoader
from features import build_feature_matrix, extract_car_features
from models import (
    LightGBMRanker, XGBoostRanker, ClassicalEnsemble, 
    UnsupervisedAnomalyDetector, SiameseRanker
)


DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@st.cache_resource
def load_models(model_dir: str = None):
    """Load all trained models and metadata."""
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    metadata = joblib.load(os.path.join(model_dir, 'ensemble_metadata.pkl'))
    weights = metadata['weights']
    feature_names = metadata['feature_names']
    model_names = metadata['model_names']
    
    models = {}
    for name in model_names:
        model_path = os.path.join(model_dir, f'{name}_model.pkl')
        if name == 'lgbm':
            models[name] = LightGBMRanker.load(model_path)
        elif name == 'xgb':
            models[name] = XGBoostRanker.load(model_path)
        elif name == 'classical':
            models[name] = ClassicalEnsemble.load(model_path)
        elif name == 'unsup':
            models[name] = UnsupervisedAnomalyDetector.load(model_path)
        elif name == 'siamese':
            models[name] = SiameseRanker.load(model_path)
    
    return models, weights, feature_names, metadata


def parse_car_dataframes(df: pd.DataFrame):
    """Parse a dataframe into per-car dataframes."""
    car_params = {}
    for col in df.columns:
        if col.startswith('Car ') and ' - ' in col:
            cid, param = col.split(' - ', 1)
            cid = cid.replace('Car ', '')
            if cid not in car_params:
                car_params[cid] = []
            car_params[cid].append(param)
    
    id_cols = ['Car model', 'Train number', 'Time']
    car_dfs = {}
    for car_id, params in car_params.items():
        car_cols = id_cols + [f'Car {car_id} - {p}' for p in params]
        car_df = df[car_cols].copy()
        car_df.columns = id_cols + params
        car_dfs[car_id] = car_df
    
    return car_dfs


def build_test_features(test_df: pd.DataFrame, feature_names: list):
    """Build feature matrix for test data."""
    car_dfs = parse_car_dataframes(test_df)
    
    test_features = []
    test_car_ids = []
    
    for car_id in sorted(car_dfs.keys()):
        car_df = car_dfs[car_id]
        feats = extract_car_features(car_df, car_id, car_dfs)
        test_features.append(feats)
        test_car_ids.append(car_id)
    
    test_feat_df = pd.DataFrame(test_features)
    test_feat_df = test_feat_df.reindex(columns=feature_names, fill_value=0.0)
    test_feat_df = test_feat_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    return test_feat_df.values, test_car_ids, car_dfs


def predict_case(models: dict, weights: np.ndarray, feature_names: list, 
                 test_df: pd.DataFrame):
    """Predict ranked cars for a single test case."""
    X_test, test_car_ids, car_dfs = build_test_features(test_df, feature_names)
    
    test_scores_dict = {}
    for name, model in models.items():
        test_scores_dict[name] = model.predict(X_test)
    
    score_matrix = np.column_stack([test_scores_dict[name] for name in models.keys()])
    ensemble_scores = score_matrix @ weights
    
    ranked_indices = np.argsort(-ensemble_scores)
    ranked_cars = [test_car_ids[i] for i in ranked_indices]
    ranked_scores = [ensemble_scores[i] for i in ranked_indices]
    
    return {
        'ranked_cars': ranked_cars,
        'ranked_scores': ranked_scores,
        'per_model_scores': test_scores_dict,
        'test_car_ids': test_car_ids,
        'car_dfs': car_dfs
    }


# Page config
st.set_page_config(
    page_title="ACV Refrigerant Leak Localisation",
    page_icon="🚂",
    layout="wide"
)

# Title
st.title("🚂 ACV Refrigerant Leak Localisation")
st.markdown("""
**Train Air Conditioning & Ventilation System - Fault Diagnosis**
Upload ACV telemetry data to identify which car has a refrigerant leak.
""")

# Load models
with st.spinner("Loading models..."):
    models, weights, feature_names, metadata = load_models()

st.sidebar.success(f"✅ Models loaded: {len(models)} models")
st.sidebar.markdown("### Model Weights")
for name, w in zip(metadata['model_names'], weights):
    st.sidebar.write(f"- {name}: {w:.2f}")

# File upload
st.header("📁 Upload Data File")
uploaded_file = st.file_uploader(
    "Choose an ACV telemetry .xlsx file", 
    type=['xlsx'],
    help="Upload a file with ACV sensor data from all 8 cars"
)

if uploaded_file is not None:
    with st.spinner("Processing file..."):
        loader = ACVDataLoader()
        # Save uploaded file temporarily
        temp_path = "temp_upload.xlsx"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = loader.load_case(temp_path)
        os.remove(temp_path)
    
    st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Show file info
    with st.expander("📊 File Overview"):
        st.write(f"**Rows:** {len(df)}")
        st.write(f"**Time range:** {df['Time'].min()} to {df['Time'].max()}")
        st.write(f"**Sampling interval:** ~30 seconds")
        
        # Show car columns
        cars = set()
        for c in df.columns:
            if c.startswith('Car ') and ' - ' in c:
                cid = c.split(' - ')[0].replace('Car ', '')
                cars.add(cid)
        st.write(f"**Cars detected:** {', '.join(sorted(cars))}")
    
    # Run prediction
    with st.spinner("Running leak localisation..."):
        result = predict_case(models, weights, feature_names, df)
    
    # Display results
    st.header("🎯 Prediction Results")
    
    # Main ranking
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ranked Cars (Most → Least Likely Faulty)")
        
        # Create ranking dataframe
        rank_df = pd.DataFrame({
            'Rank': range(1, 9),
            'Car ID': result['ranked_cars'],
            'Ensemble Score': [f"{s:.4f}" for s in result['ranked_scores']]
        })
        
        # Highlight top prediction
        def highlight_top(row):
            if row['Rank'] == 1:
                return ['background-color: #ffeb3b; font-weight: bold'] * len(row)
            elif row['Rank'] <= 3:
                return ['background-color: #fff3e0'] * len(row)
            return [''] * len(row)
        
        st.dataframe(rank_df.style.apply(highlight_top, axis=1), use_container_width=True)
        
        # Download button
        pred_csv = pd.DataFrame({
            'file_id': [uploaded_file.name],
            'ranked_cars': [' | '.join(result['ranked_cars'])]
        }).to_csv(index=False)
        st.download_button(
            "📥 Download Predictions (CSV)",
            pred_csv,
            file_name="acv_predictions.csv",
            mime="text/csv"
        )
    
    with col2:
        st.subheader("📈 Score Visualization")
        
        # Bar chart of ensemble scores
        fig = px.bar(
            x=result['ranked_cars'],
            y=result['ranked_scores'],
            labels={'x': 'Car ID', 'y': 'Ensemble Score'},
            title="Leak Likelihood Score by Car",
            color=result['ranked_scores'],
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Per-model comparison
    st.header("🔍 Per-Model Analysis")
    
    model_names = list(models.keys())
    model_scores_df = pd.DataFrame(result['per_model_scores'], index=result['test_car_ids'])
    model_scores_df = model_scores_df[model_names]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Scores Heatmap")
        fig = px.imshow(
            model_scores_df.T,
            labels=dict(x="Car", y="Model", color="Score"),
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Anomaly Scores by Model and Car"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Ranking Agreement")
        
        # Show each model's ranking
        for name in model_names:
            scores = result['per_model_scores'][name]
            ranked = np.argsort(-scores)
            ranked_cars_model = [result['test_car_ids'][i] for i in ranked]
            st.write(f"**{name.upper()}**: {' → '.join(ranked_cars_model)}")
    
    # Sensor data visualization
    st.header("📊 Sensor Data Exploration")
    
    car_dfs = result['car_dfs']
    selected_car = st.selectbox("Select Car to Inspect", sorted(car_dfs.keys()))
    
    if selected_car:
        car_df = car_dfs[selected_car]
        
        # Get numeric columns (excluding IDs)
        numeric_cols = car_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['Car model', 'Train number']]
        
        if numeric_cols:
            selected_params = st.multiselect(
                "Select Parameters to Plot",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if selected_params:
                fig = go.Figure()
                for param in selected_params:
                    fig.add_trace(go.Scatter(
                        x=car_df['Time'],
                        y=car_df[param],
                        mode='lines',
                        name=param,
                        line=dict(width=1)
                    ))
                
                fig.update_layout(
                    title=f"Car {selected_car} - Sensor Time Series",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Cross-car comparison for key parameters
    st.header("🔄 Cross-Car Comparison")
    
    # Find common parameters across all cars
    all_params = set()
    for car_df in car_dfs.values():
        numeric = car_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if c not in ['Car model', 'Train number']]
        all_params.update(numeric)
    
    # Find params present in all cars
    common_params = []
    for param in all_params:
        if all(param in car_dfs[c].select_dtypes(include=[np.number]).columns 
               for c in car_dfs.keys()):
            common_params.append(param)
    
    if common_params:
        compare_param = st.selectbox(
            "Select Parameter for Cross-Car Comparison",
            common_params
        )
        
        if compare_param:
            fig = go.Figure()
            for car_id in sorted(car_dfs.keys()):
                car_df = car_dfs[car_id]
                fig.add_trace(go.Scatter(
                    x=car_df['Time'],
                    y=car_df[compare_param],
                    mode='lines',
                    name=f"Car {car_id}",
                    line=dict(width=1)
                ))
            
            fig.update_layout(
                title=f"Cross-Car Comparison: {compare_param}",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics table
            stats_data = []
            for car_id in sorted(car_dfs.keys()):
                vals = pd.to_numeric(car_dfs[car_id][compare_param], errors='coerce').dropna()
                if len(vals) > 0:
                    stats_data.append({
                        'Car': car_id,
                        'Mean': f"{vals.mean():.2f}",
                        'Std': f"{vals.std():.2f}",
                        'Min': f"{vals.min():.2f}",
                        'Max': f"{vals.max():.2f}",
                        'Median': f"{vals.median():.2f}"
                    })
            
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)


if __name__ == "__main__":
    # This is needed for Streamlit to run properly
    pass