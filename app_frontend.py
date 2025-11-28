import streamlit as st
import numpy as np
import pandas as pd
import time
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier # For dummy model creation

# --- CONFIGURATION CONSTANTS (Matching Backend) ---
SAMPLE_RATE_HZ = 1000
TEST_DURATION_SECONDS = 5
TOTAL_SAMPLES = SAMPLE_RATE_HZ * TEST_DURATION_SECONDS
MODEL_PATH = 'inspection_model.joblib'
MAX_FORCE_UP_DOWN = 4.0 # N
MAX_FORCE_SLIDING = 6.0 # N

# --- DUMMY MODEL FUNCTIONS (Needed to ensure the model exists) ---
@st.cache_resource # Cache model loading for efficiency
def load_or_train_model(path=MODEL_PATH):
    """Loads the model or trains a dummy one if it doesn't exist."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        # Simplified features: [peak_fz, peak_fx_or_fy, smoothness_score]
        X = np.array([[3.5, 5.0, 0.05], [8.5, 4.0, 0.15], [3.1, 5.8, 0.02], [5.0, 5.0, 0.05]])
        y = np.array([0, 1, 0, 1])
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        joblib.dump(model, path)
        st.info("NOTE: Dummy ML Model trained and saved. Use real data for production.")
        return model

# --- SIMULATED DATA ACQUISITION ---
def read_sensor_data(sample_index, simulated_force_max):
    """
    Simulates reading synchronized data from all sensors.
    simulated_force_max controls the peak vertical force (Fz).
    """
    t = sample_index / SAMPLE_RATE_HZ
    
    # Simulate Fz (vertical force) - Peak will be approx simulated_force_max * 0.5
    Fz = simulated_force_max * (1 - np.cos(2 * np.pi * t / TEST_DURATION_SECONDS)) * 0.5 + np.random.normal(0, 0.1)
    
    # Simulate lateral force (Fx/Fy) - Fx peaks around 3.5N
    Fx = 7.0 * np.sin(2 * np.pi * t / TEST_DURATION_SECONDS) * 0.5 + np.random.normal(0, 0.2) 
    Fy = np.random.normal(0, 0.2)
    
    # Simulate torques and position
    Mx = 0.5 * np.sin(2 * np.pi * t / TEST_DURATION_SECONDS) + np.random.normal(0, 0.05)
    Pz = 100 * (1 - np.cos(2 * np.pi * t / TEST_DURATION_SECONDS)) * 0.5

    return {
        'Timestamp_ms': int(t * 1000),
        'Fx_N': Fx, 'Fy_N': Fy, 'Fz_N': Fz,
        'Mx_Nm': Mx, 'My_Nm': 0.0, 'Mz_Nm': 0.0,
        'Pz_mm': Pz
    }

def run_inspection_cycle(simulated_force_max, progress_bar):
    """Executes a single louver inspection cycle and collects data."""
    data_records = []
    
    for i in range(TOTAL_SAMPLES):
        record = read_sensor_data(i, simulated_force_max)
        data_records.append(record)
        
        # Update progress bar
        progress_bar.progress((i + 1) / TOTAL_SAMPLES)
        
        # In a real system, you'd wait here for 1/SAMPLE_RATE_HZ
        # time.sleep(1/SAMPLE_RATE_HZ) 
        
    df = pd.DataFrame(data_records)
    return df

# --- FEATURE ENGINEERING ---
def feature_engineering(df: pd.DataFrame) -> np.ndarray:
    """Transforms raw time-series data into the features the AI model expects."""
    
    peak_fz = df['Fz_N'].abs().max()
    peak_fx_or_fy = np.maximum(df['Fx_N'].abs().max(), df['Fy_N'].abs().max())

    df['Fz_diff'] = df['Fz_N'].diff().fillna(0)
    smoothness_score = df['Fz_diff'].std()
    
    # Feature for ML model: [peak_fz, peak_fx_or_fy, smoothness_score]
    features = np.array([[peak_fz, peak_fx_or_fy, smoothness_score]])
    
    return features, peak_fz, peak_fx_or_fy, smoothness_score

# --- PREDICTION LOGIC ---
def predict_quality(df: pd.DataFrame, model) -> dict:
    """Runs the full inspection: Hard Limit Check + ML Classification."""
    
    features, peak_fz, peak_fx_or_fy, smoothness_score = feature_engineering(df)
    
    result = {'Fz_Actual': peak_fz, 'Fxy_Actual': peak_fx_or_fy, 'Smoothness_STD': smoothness_score}
    
    # 1. HARD LIMIT CHECK
    if peak_fz > MAX_FORCE_UP_DOWN:
        result['Quality_Result'] = "FAIL - Fz HARD LIMIT BREACH"
        result['Failure_Cause'] = f'Vertical force ({peak_fz:.2f} N) > {MAX_FORCE_UP_DOWN:.1f} N'
        return result
    
    if peak_fx_or_fy > MAX_FORCE_SLIDING:
        result['Quality_Result'] = "FAIL - Fx/Fy HARD LIMIT BREACH"
        result['Failure_Cause'] = f'Sideways force ({peak_fx_or_fy:.2f} N) > {MAX_FORCE_SLIDING:.1f} N'
        return result

    # 2. ML CLASSIFICATION (Soft Limits / Smoothness)
    try:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
    except Exception:
        result['Quality_Result'] = "ERROR - ML PREDICTION FAILED"
        result['Failure_Cause'] = "Model input error."
        return result
    
    if prediction != 0: # Assuming 0 is the "Good" class
         result['Quality_Result'] = "FAIL - ML: Stiffness/Binding"
         result['Failure_Cause'] = 'Excessive friction/torque detected by ML model'
    else:
        result['Quality_Result'] = "PASS - Meets all mechanical specs"
        result['Failure_Cause'] = 'N/A'
        result['Confidence'] = np.max(probabilities)

    return result

# --- STREAMLIT FRONTEND LAYOUT ---

st.set_page_config(layout="wide")

st.title(" AC Vent Inspection Console")
st.caption("Actuator Force/Motion Analysis (No Vision) | Limits: Fz < 4.0N, Fxy < 6.0N")

st.markdown("---")

# Load the model once
quality_model = load_or_train_model()

# --- INPUTS AND CONTROLS ---
col1, col2 = st.columns(2)

with col1:
    test_mode = st.radio("Select Test Mode:", 
                         ('PASS (Simulated 1.0N Max)', 
                          'FAIL - Fz Breach (Simulated 5.0N Max)',
                          'FAIL - Fxy Breach (Simulated 7.0N Max)'), 
                         index=0)
    
    if test_mode == 'PASS (Simulated 1.0N Max)':
        sim_max_force = 2.0 # Peak Fz approx 1.0N
    elif test_mode == 'FAIL - Fz Breach (Simulated 5.0N Max)':
        sim_max_force = 10.0 # Peak Fz approx 5.0N (> 4.0N limit)
    elif test_mode == 'FAIL - Fxy Breach (Simulated 7.0N Max)':
        sim_max_force = 1.0 # Fz will pass, but Fx/Fy is hardcoded to peak > 6N
        
    start_test = st.button("▶️ Start Inspection Cycle", type="primary")

st.markdown("---")

# --- RESULTS AREA ---
if start_test:
    st.session_state['status'] = 'running'
    
    # Placeholder for the progress bar
    progress_container = st.empty()
    progress_bar = progress_container.progress(0, text="Initializing actuator and sensors...")
    
    # 1. Run Data Acquisition
    with st.spinner("Executing Actuator Motion and Collecting Force Data..."):
        time.sleep(1) # Simulate hardware initialization delay
        df_raw = run_inspection_cycle(sim_max_force, progress_bar)
        
    progress_bar.empty()
    progress_container.empty()
    
    # 2. Run AI Inference
    st.subheader("📊 Inspection Results")
    
    final_result = predict_quality(df_raw, quality_model)
    
    # 3. Display Final Decision
    if "PASS" in final_result['Quality_Result']:
        st.success(f"## ✅ {final_result['Quality_Result']}")
    elif "FAIL" in final_result['Quality_Result']:
        st.error(f"## ❌ {final_result['Quality_Result']}")
    else:
        st.warning(f"## ⚠️ {final_result['Quality_Result']}")

    st.markdown(f"**Reason for Decision:** `{final_result['Failure_Cause']}`")
    st.markdown("---")

    # 4. Display Metrics and Charts
    
    metrics_cols = st.columns(4)
    
    # Display Key Metrics
    metrics_cols[0].metric("Peak Fz (Vertical)", f"{final_result['Fz_Actual']:.2f} N", 
                           delta_color="off" if final_result['Fz_Actual'] <= MAX_FORCE_UP_DOWN else "inverse")
    metrics_cols[1].metric("Peak Fx/Fy (Sliding)", f"{final_result['Fxy_Actual']:.2f} N", 
                           delta_color="off" if final_result['Fxy_Actual'] <= MAX_FORCE_SLIDING else "inverse")
    metrics_cols[2].metric("Smoothness (Fz STD)", f"{final_result['Smoothness_STD']:.4f}")
    if 'Confidence' in final_result:
        metrics_cols[3].metric("ML Confidence", f"{final_result['Confidence']:.2f}")

    # Display the Force-Position Curve 
    st.subheader("Force-Position Curve Analysis (Fz vs. Pz)")
    
    # Create the Force vs. Position DataFrame for plotting
    plot_df = df_raw[['Pz_mm', 'Fz_N', 'Mx_Nm']].copy()
    
    st.line_chart(plot_df, x='Pz_mm', y=['Fz_N', 'Mx_Nm'])
    st.caption("Fz (Vertical Force) should be smooth, and Mx (Binding Torque) should remain low.")

    # Optional: Display Raw Data (for debugging)
    with st.expander("View Raw Collected Data"):
        st.dataframe(df_raw.head(100))

# --- Initial State ---
else:
    st.info("Press 'Start Inspection Cycle' to begin the test and view results.")