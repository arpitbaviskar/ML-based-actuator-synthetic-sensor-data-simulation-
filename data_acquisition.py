import numpy as np
import time
import pandas as pd
from datetime import datetime

# --- CONFIGURATION CONSTANTS ---
SAMPLE_RATE_HZ = 1000  # 1000 Hz or 1kHz
TEST_DURATION_SECONDS = 5
TOTAL_SAMPLES = SAMPLE_RATE_HZ * TEST_DURATION_SECONDS

MAX_FORCE_UP_DOWN = 4.0 # N
MAX_FORCE_SLIDING = 6.0 # N

def predict_quality(df: pd.DataFrame, model) -> dict:
    # ... (Feature Engineering runs here) ...
    
    # Example Features (simplified for illustration)
    peak_fz = features[0, 0] # Peak Fz is the first feature
    peak_fx_or_fy = features[0, 1] # Peak Fx or Fy is the second feature

    # --- HARD LIMIT CHECK ---
    if peak_fz > MAX_FORCE_UP_DOWN:
        return {'Quality_Result': "FAIL - Fz HARD LIMIT BREACH", 'Failure_Cause': 'Vertical force too high'}
    
    if peak_fx_or_fy > MAX_FORCE_SLIDING:
        return {'Quality_Result': "FAIL - Fx/Fy HARD LIMIT BREACH", 'Failure_Cause': 'Sideways force too high'}

    # --- ML CLASSIFICATION (Soft Limits / Smoothness) ---
    prediction = model.predict(features)[0]
    
    if prediction != 0: # Assuming 0 is the "Good" class
         return {'Quality_Result': "FAIL - ML: Stiffness/Binding", 'Failure_Cause': 'Excessive friction/torque detected'}

    return {'Quality_Result': "PASS - Meets all mechanical specs", 'Failure_Cause': 'N/A'}

# --- SIMULATED HARDWARE LIBRARIES ---
# In a real system, these would be API calls to your sensor/robot controller
def initialize_hardware():
    """Initializes communication with the F/T sensor and motion controller."""
    print("STATUS: Initializing 6-Axis F/T Sensor...")
    time.sleep(0.5)
    print("STATUS: Initializing Gantry & Z-Actuator Motion Controller...")
    time.sleep(0.5)
    print("STATUS: Hardware Ready.")
    # In reality, this would return device handlers

def read_sensor_data(sample_index, max_force=50, max_torque=2.0):
    """
    Simulates reading synchronized data from all sensors.
    In a real system, this is the core loop reading the DAQ system.
    """
    t = sample_index / SAMPLE_RATE_HZ
    
    # Simulate a force curve (e.g., a simple stiffness test with some noise)
    # Fz (vertical force) changes with Pz (actuator position)
    Fz = max_force * (1 - np.cos(2 * np.pi * t / TEST_DURATION_SECONDS)) * 0.5 + np.random.normal(0, 0.1)
    
    # Simulate a small binding torque (Mx, My) and sideways friction (Fx, Fy)
    Mx = max_torque * np.sin(2 * np.pi * t / TEST_DURATION_SECONDS) * 0.1 + np.random.normal(0, 0.05)
    My = max_torque * np.cos(2 * np.pi * t / TEST_DURATION_SECONDS) * 0.1 + np.random.normal(0, 0.05)
    Fx = np.random.normal(0, 0.2)
    Fy = np.random.normal(0, 0.2)
    
    # Simulate robot/actuator position (Pz drives the test)
    Pz = 100 * (1 - np.cos(2 * np.pi * t / TEST_DURATION_SECONDS)) * 0.5  # Actuator moves 0 to 100 mm
    Px = 500  # Gantry X position (fixed during the test)
    Py = 750  # Gantry Y position (fixed during the test)

    return {
        'Timestamp_ms': int(t * 1000),
        'Fx_N': Fx, 'Fy_N': Fy, 'Fz_N': Fz,
        'Mx_Nm': Mx, 'My_Nm': My, 'Mz_Nm': 0.0, # Assuming Mz is zero for this motion
        'Px_mm': Px, 'Py_mm': Py, 'Pz_mm': Pz
    }

def run_inspection_cycle(cycle_id):
    """Executes a single louver inspection cycle and collects data."""
    print(f"\n--- Starting Inspection Cycle {cycle_id} ---")
    
    # In a real system, trigger the gantry/actuator motion command here.
    
    data_records = []
    start_time = time.perf_counter()
    
    for i in range(TOTAL_SAMPLES):
        # 1. Read Data
        record = read_sensor_data(i)
        
        # 2. Add Metadata
        record['Cycle_ID'] = cycle_id
        record['Time_ISO'] = datetime.now().isoformat()
        
        # 3. Store
        data_records.append(record)
        
        # 4. Maintain Sample Rate (Crude simulation, real system uses interrupts/DAQ clock)
        time.sleep(1/SAMPLE_RATE_HZ) 
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"INFO: Collected {len(data_records)} samples in {duration:.3f} seconds.")
    
    # Convert to DataFrame for analysis and storage
    df = pd.DataFrame(data_records)
    
    # 5. Store Data
    file_path = f'force_data_{cycle_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(file_path, index=False)
    print(f"SUCCESS: Data saved to {file_path}")
    
    return df

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    initialize_hardware()
    
    # Run the test and get the data for immediate AI processing
    test_data_df = run_inspection_cycle(cycle_id="VENT_TEST_001")
    
    # Display the first few rows
    print("\nCollected Data Head:")
    print(test_data_df[['Timestamp_ms', 'Fz_N', 'Mx_Nm', 'Pz_mm']].head())
    
    # In a production environment, this DataFrame would immediately be passed
    # to the AI inference module (model_inference.py).