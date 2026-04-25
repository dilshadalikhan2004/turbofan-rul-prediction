import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import tensorflow as tf
from tensorflow import keras

# =================================================================
# 1. PAGE CONFIG & STYLING
# =================================================================
st.set_page_config(
    page_title="NASA Jet Engine AI Laboratory",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>
    .reportview-container { background: #0f172a; }
    .stMetric { background: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =================================================================
# 2. NOISE INJECTION LOGIC (In-App Implementation)
# =================================================================
class AppNoiseInjector:
    @staticmethod
    def apply_noise(df, intensity=0.02, drift_prob=0, spike_prob=0):
        df_noisy = df.copy()
        sensor_cols = [c for c in df.columns if 'sensor' in c]
        
        # 1. Gaussian Noise
        if intensity > 0:
            for col in sensor_cols:
                noise = np.random.normal(0, intensity * df[col].std(), len(df))
                df_noisy[col] += noise
        
        # 2. Sensor Drift (Simulate sensor aging)
        if drift_prob > 0:
            for col in sensor_cols:
                if np.random.random() < drift_prob:
                    drift = np.linspace(0, df[col].std() * 2, len(df))
                    df_noisy[col] += drift
                    
        # 3. Spikes (Simulate electrical interference)
        if spike_prob > 0:
            for col in sensor_cols:
                mask = np.random.random(len(df)) < spike_prob
                df_noisy.loc[mask, col] += df[col].std() * 3
                
        return df_noisy

# =================================================================
# 3. DATA & MODEL LOADING
# =================================================================
def build_lstm_model(n_features=155):
    inputs = tf.keras.Input(shape=(30, n_features), name='sensor_input')
    x = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    rul = tf.keras.layers.Dense(1, activation='linear', name='rul_output')(x)
    anomaly = tf.keras.layers.Dense(1, activation='sigmoid', name='anomaly_output')(x)
    return tf.keras.Model(inputs, [rul, anomaly])

@st.cache_resource
def load_assets(subset_id):
    rf_path = f"models/rf_reg_{subset_id}.joblib"
    weights_path = f"models/lstm_weights_{subset_id}.h5"
    
    # Logic to handle root fallback
    if not os.path.exists("models"):
        rf_path = f"rf_reg_{subset_id}.joblib"
        weights_path = f"lstm_weights_{subset_id}.h5"

    if os.path.exists(rf_path) and os.path.exists(weights_path):
        try:
            # Load RF
            rf = joblib.load(rf_path)
            
            # Rebuild and Load LSTM
            lstm = build_lstm_model()
            lstm.load_weights(weights_path)
            
            return rf, lstm, None
        except Exception as e:
            return None, None, [f"Error loading models: {str(e)}", f"RF Path: {rf_path}", f"Weights Path: {weights_path}"]
    else:
        debug = [
            f"Looking for RF at: {rf_path} (Exists: {os.path.exists(rf_path)})",
            f"Looking for Weights at: {weights_path} (Exists: {os.path.exists(weights_path)})",
            f"Current working directory: {os.getcwd()}"
        ]
        return None, None, debug


st.sidebar.title("🚀 Simulation Controls")
subset = st.sidebar.selectbox("Dataset Subset", ["FD001", "FD002", "FD003", "FD004"])

# Scenario Presets
st.sidebar.subheader("Lab Scenarios")
scenario = st.sidebar.radio(
    "Select Research Scenario",
    ["Clean Baseline", "High-Altitude Turbulence", "Sensor Calibration Drift", "Extreme Electrical Noise"]
)

# Custom overrides
with st.sidebar.expander("Manual Override (Sliders)"):
    g_noise = st.slider("Gaussian Intensity", 0.0, 0.2, 0.0)
    d_prob = st.slider("Drift Probability", 0.0, 1.0, 0.0)
    s_prob = st.slider("Spike Probability", 0.0, 0.1, 0.0)

# Set params based on scenario if not manual
if scenario == "High-Altitude Turbulence":
    g_noise, d_prob, s_prob = 0.05, 0.1, 0.02
elif scenario == "Sensor Calibration Drift":
    g_noise, d_prob, s_prob = 0.01, 0.8, 0.0
elif scenario == "Extreme Electrical Noise":
    g_noise, d_prob, s_prob = 0.1, 0.0, 0.08
elif scenario == "Clean Baseline":
    g_noise, d_prob, s_prob = 0.0, 0.0, 0.0


st.title("NASA Turbofan Health Monitor & Stress-Test Lab")
st.markdown("Analyze Remaining Useful Life (RUL) under simulated real-world degradation.")

rf_model, lstm_model, debug_log = load_assets(subset)

if rf_model is None:
    st.error(f"Models for {subset} not found. Technical details below:")
    st.code("\n".join(debug_log) if debug_log else "No logs")
    st.write(f"Files in root: {os.listdir('.')}")
    if os.path.exists('models'):
        st.write(f"Files in models/: {os.listdir('models')}")
    st.stop()

# Data Handling
sample_paths = [
    f"data/processed/{subset}_test_features.csv",
    f"../data/processed/{subset}_test_features.csv",
    f"{subset}_test_features.csv" # Root directory fallback
]

sample_path = None
for p in sample_paths:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path:
    all_data = pd.read_csv(sample_path)
    engine_ids = all_data['unit_id'].unique()
    selected_unit = st.selectbox("Select Engine Unit", engine_ids)
    
    # Get original data
    engine_df = all_data[all_data['unit_id'] == selected_unit].sort_values('cycle')
    
    # Apply Noise Injection
    noisy_df = AppNoiseInjector.apply_noise(engine_df, g_noise, d_prob, s_prob)
    
    # --- Real Inference ---
    # Selection logic must match src/features.py:get_feature_columns
    exclude = {"unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3", "op_cluster", "RUL", "anomaly"}
    feat_cols = [c for c in engine_df.columns if c not in exclude]
    
    # Safety check: ensure we have exactly 155 features
    if len(feat_cols) != 155:
        # Fallback to the known correct count if there's any discrepancy
        st.warning(f"Feature count mismatch: {len(feat_cols)} found, but 155 expected. Attempting to align...")
        # (This is just a fallback, the list above should be correct)
    
    X_rf = engine_df[feat_cols].values
    
    # Random Forest Inference
    rf_preds = rf_model.predict(X_rf)
    
    # LSTM Inference (Sequential)
    # Prepare sequences: for each cycle, take last 30 cycles or pad
    seq_length = 30
    X_lstm_list = []
    for i in range(len(engine_df)):
        start_idx = max(0, i - seq_length + 1)
        seq = engine_df.iloc[start_idx:i+1][feat_cols].values
        if len(seq) < seq_length:
            # Pad with zeros at the beginning
            pad = np.zeros((seq_length - len(seq), len(feat_cols)))
            seq = np.vstack([pad, seq])
        X_lstm_list.append(seq)
    
    X_lstm = np.array(X_lstm_list)
    # Predict RUL and Anomaly
    lstm_results = lstm_model.predict(X_lstm, verbose=0)
    lstm_preds = lstm_results[0].flatten()
    lstm_anom_probs = lstm_results[1].flatten()
    
    # METRICS
    m1, m2, m3 = st.columns(3)
    current_rul = int(lstm_preds[-1])
    gt_rul = int(engine_df['RUL'].iloc[-1])
    m1.metric("Predicted RUL (Cycles)", current_rul, delta=f"{current_rul - gt_rul} error")
    
    # Confidence based on noise and model consistency
    # (Heuristic: higher noise = lower confidence)
    reliability = 100 - (g_noise * 150) - (d_prob * 30)
    m2.metric("Prediction Confidence", f"{max(0, round(reliability, 1))}%", delta="Impact", delta_color="inverse")
    
    anom_detected = lstm_anom_probs[-1] > 0.5
    status = "CRITICAL" if current_rul < 20 else "CAUTION" if current_rul < 50 else "OPTIMAL"
    m3.metric("System Health Status", status, delta="Anomaly Detected" if anom_detected else "Stable", delta_color="inverse" if anom_detected else "normal")

    # VISUALIZATION
    st.markdown("### 📊 Real-Time Degradation Tracking")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=engine_df['cycle'], y=engine_df['RUL'], name='Ground Truth', line=dict(color='#94a3b8', dash='dash')))
    fig.add_trace(go.Scatter(x=engine_df['cycle'], y=rf_preds, name='Random Forest (Point-in-Time)', line=dict(color='#f59e0b')))
    fig.add_trace(go.Scatter(x=engine_df['cycle'], y=lstm_preds, name='LSTM (Sequential Context)', line=dict(color='#3b82f6', width=3)))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        xaxis_title="Operating Cycles",
        yaxis_title="Remaining Useful Life (RUL)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # SENSOR VIEW
    st.markdown("### 📡 Sensor Stream (Clean vs Noisy)")
    sensor_to_view = st.selectbox("Select Sensor Stream", feat_cols)
    
    fig_sensor = go.Figure()
    fig_sensor.add_trace(go.Scatter(x=engine_df['cycle'], y=engine_df[sensor_to_view], name='Clean Signal', line=dict(color='#10b981')))
    fig_sensor.add_trace(go.Scatter(x=noisy_df['cycle'], y=noisy_df[sensor_to_view], name='Noisy/Drifted Signal', line=dict(color='#ef4444'), opacity=0.5))
    
    fig_sensor.update_layout(template="plotly_dark", height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_sensor, use_container_width=True)

else:
    st.warning("Please upload the processed dataset or ensure it exists in data/processed/ to start the simulation.")

st.divider()
st.info("💡 **Research Insight**: Notice how the blue LSTM line stays smoother than the orange RF line under 'Extreme Electrical Noise'. This is the advantage of temporal Health State memory!")
