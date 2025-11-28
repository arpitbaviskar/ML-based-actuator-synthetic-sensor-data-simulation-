import numpy as np
import time
import pandas as pd
import joblib # Needed for saving/loading ML models
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier # Used for dummy model

# --- CONFIGURATION CONSTANTS ---
SAMPLE_RATE_HZ = 1000  # 1000 Hz or 1kHz
TEST_DURATION_SECONDS = 5
TOTAL_SAMPLES = SAMPLE_RATE_HZ * TEST_DURATION_SECONDS

# --- FORCE LIMITS ---
MAX_FORCE_UP_DOWN = 4.0 # N
MAX_FORCE_SLIDING = 6.0 # N

# --- DUMMY MODEL FUNCTIONS ---
# This is a placeholder for your actual trained AI model
def train_dummy_model(path='dummy_model.joblib'):
    """Creates and saves a dummy model for demonstration."""
    print("INFO: Training dummy classifier...")
    # Simplified features: [peak_fz, peak_fx_or_fy, smoothness_score]
    X = np.array([[3.5, 5.0, 0.05], [8.5, 4.0, 0.15], [3.1, 5.8, 0.02]])
    # 0 = Good, 1 = Stiff/Binding, 2 = Loose (We only use 0 and 1 here)
    y = np.array([0, 1, 0])
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    joblib.dump(model, path)
    return model

# --- FEATURE ENGINEERING ---
def feature_engineering(df: pd.DataFrame) -> np.ndarray:
    """
    REQUIRED: Transforms raw time-series data into the features the AI model expects.
    This is the intellectual core of the analysis.
    """
    
    # 1. Peak Fz (Up/Down force) - Primary Hard Limit Check
    peak_fz = df['Fz_N'].abs().max()

    # 2. Peak Fx/Fy (Sliding force) - Primary Hard Limit Check
    # We take the max of the absolute values of the lateral forces.
    peak_fx_or_fy = np.maximum(df['Fx_N'].abs().max(), df['Fy_N'].abs().max())

    # 3. Smoothness Score (Standard Deviation of Force Derivative) - Soft Limit Check
    # High standard deviation of force changes indicates "grittiness" or binding spikes.
    df['Fz_diff'] = df['Fz_N'].diff().fillna(0)
    smoothness_score = df['Fz_diff'].std()
    
    # The model expects an array of features: [peak_fz, peak_fx_or_fy, smoothness_score]
    # NOTE: The order here MUST match the order used during model training.
    features = np.array([[peak_fz, peak_fx_or_fy, smoothness_score]])
    
    return features


def predict_quality(df: pd.DataFrame, model) -> dict:
    """
    Runs the full inspection: Hard Limit Check + ML Classification.
    """
    # 1. Feature Engineering
    features = feature_engineering(df)
    
    # Example Features mapping to the array indices
    peak_fz = features[0, 0] 
    peak_fx_or_fy = features[0, 1] 
    
    # --- HARD LIMIT CHECK ---
    if peak_fz > MAX_FORCE_UP_DOWN:
        return {'Quality_Result': "FAIL - Fz HARD LIMIT BREACH", 
                'Failure_Cause': f'Vertical force ({peak_fz:.2f} N) exceeds {MAX_FORCE_UP_DOWN:.1f} N', 
                'Fz_Actual': peak_fz, 'Fxy_Actual': peak_fx_or_fy}
    
    if peak_fx_or_fy > MAX_FORCE_SLIDING:
        return {'Quality_Result': "FAIL - Fx/Fy HARD LIMIT BREACH", 
                'Failure_Cause': f'Sideways force ({peak_fx_or_fy:.2f} N) exceeds {MAX_FORCE_SLIDING:.1f} N',
                'Fz_Actual': peak_fz, 'Fxy_Actual': peak_fx_or_fy}

    # --- ML CLASSIFICATION (Soft Limits / Smoothness) ---
    try:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
    except Exception as e:
        return {'Quality_Result': "ERROR - ML PREDICTION FAILED", 
                'Failure_Cause': f'Model error: {e}'}

    
    if prediction != 0: # Assuming 0 is the "Good" class
         return {'Quality_Result': "FAIL - ML: Stiffness/Binding", 
                 'Failure_Cause': f'Excessive friction/torque detected (Smoothness STD: {features[0, 2]:.4f})'}

    return {'Quality_Result': "PASS - Meets all mechanical specs", 
            'Failure_Cause': 'N/A', 
            'Fz_Actual': peak_fz, 'Fxy_Actual': peak_fx_or_fy,
            'Confidence': np.max(probabilities)}

# --- SIMULATED HARDWARE LIBRARIES (unchanged) ---
def initialize_hardware():
    """Initializes communication with the F/T sensor and motion controller."""
    print("STATUS: Initializing 2-Axis F/T Sensor...")
    time.sleep(0.5)

def read_sensor_data(sample_index, max_force=50, max_torque=2.0):
    """
    Simulates reading synchronized data from all sensors.
    We manipulate the max_force to intentionally fail the hard limit test
    when the simulation is run below.
    """
    t = sample_index / SAMPLE_RATE_HZ
    
    # Simulate a force curve (Fz changes with Pz)
    Fz = max_force * (1 - np.cos(2 * np.pi * t / TEST_DURATION_SECONDS)) * 0.5 + np.random.normal(0, 0.1)
    
    # Simulate a small binding torque (Mx, My) and sideways friction (Fx, Fy)
    Mx = max_torque * np.sin(2 * np.pi * t / TEST_DURATION_SECONDS) * 0.1 + np.random.normal(0, 0.05)
    My = max_torque * np.cos(2 * np.pi * t / TEST_DURATION_SECONDS) * 0.1 + np.random.normal(0, 0.05)
    # Fx and Fy are lateral forces - we'll make them large enough to test the 6N limit occasionally
    Fx = 7.0 * np.sin(2 * np.pi * t / TEST_DURATION_SECONDS) * 0.5 + np.random.normal(0, 0.2) 
    Fy = np.random.normal(0, 0.2)
    
    # Simulate robot/actuator position (Pz drives the test)
    Pz = 100 * (1 - np.cos(2 * np.pi * t / TEST_DURATION_SECONDS)) * 0.5
    Px = 500
    Py = 750

    return {
        'Timestamp_ms': int(t * 1000),
        'Fx_N': Fx, 'Fy_N': Fy, 'Fz_N': Fz,
        'Mx_Nm': Mx, 'My_Nm': My, 'Mz_Nm': 0.0,
        'Px_mm': Px, 'Py_mm': Py, 'Pz_mm': Pz
    }

def run_inspection_cycle(cycle_id, simulated_force_max):
    """Executes a single louver inspection cycle and collects data."""
    print(f"\n--- Starting Inspection Cycle {cycle_id} ---")
    
    data_records = []
    start_time = time.perf_counter()
    
    for i in range(TOTAL_SAMPLES):
        record = read_sensor_data(i, simulated_force_max) # Pass the force max here
        data_records.append(record)
        time.sleep(1/SAMPLE_RATE_HZ) 
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"INFO: Collected {len(data_records)} samples in {duration:.3f} seconds.")
    
    df = pd.DataFrame(data_records)
    
    file_path = f'force_data_{cycle_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(file_path, index=False)
    print(f"SUCCESS: Data saved to {file_path}")
    
    return df

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # --- Setup and Initialization ---
    initialize_hardware()
    
    # Load or train the dummy model
    # We train the model here for the first run, but in production, you only load it.
    MODEL_PATH = 'inspection_model.joblib'
    try:
        quality_model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        quality_model = train_dummy_model(MODEL_PATH)
        
    # --- Test 1: Intentional Hard Limit Failure (Fz) ---
    # REDUCED SIMULATED FORCE: Setting simulated_force_max to 10.0.
    # This results in a peak Fz of approximately 10.0 * 0.5 = 5.0 N (which is > 4.0 N limit).
    test_data_fail_fz = run_inspection_cycle(cycle_id="FAIL_FZ", simulated_force_max=3.0) 
    result_fail_fz = predict_quality(test_data_fail_fz, quality_model)
    print("\n\n=============== TEST 1 RESULT ===============")
    print(f"**DECISION: {result_fail_fz['Quality_Result']}**")
    print(f"REASON: {result_fail_fz['Failure_Cause']}")
    
    # --- Test 2: Intentional ML Classification Failure (Simulate high smoothness STD/Grittiness) ---
    # Max Fz simulation will be 2N (below 4N limit). Fx/Fy will also be below 6N.
    test_data_fail_ml = run_inspection_cycle(cycle_id="FAIL_ML", simulated_force_max=4.5) 
    result_fail_ml = predict_quality(test_data_fail_ml, quality_model)
    print("\n\n=============== TEST 2 RESULT ===============")
    print(f"**DECISION: {result_fail_ml['Quality_Result']}**")
    print(f"REASON: {result_fail_ml['Failure_Cause']}")
    
    # --- Test 3: Successful Pass (Requires realistic, low force data) ---
    # Simulating very low force (0.5N max) to ensure a PASS.
    test_data_pass = run_inspection_cycle(cycle_id="PASS_ALL", simulated_force_max=1.0) 
    result_pass = predict_quality(test_data_pass, quality_model)
    print("\n\n=============== TEST 3 RESULT ===============")
    print(f"**DECISION: {result_pass['Quality_Result']}**")
    print(f"REASON: {result_pass['Failure_Cause']}")