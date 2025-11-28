import numpy as np
import pandas as pd
import joblib # Common library for saving/loading ML models

# --- CONFIGURATION ---
MODEL_PATH = 'trained_force_classifier.joblib'

# --- SIMULATED ML MODEL ---
# In a real system, this model would be trained on thousands of samples.
def train_dummy_model():
    """Creates and saves a dummy model for demonstration."""
    from sklearn.ensemble import RandomForestClassifier
    print("INFO: Training dummy classifier...")
    
    # Create dummy data (simplified features for a classifier)
    # 0 = Good, 1 = Stiff, 2 = Loose
    X = np.array([[5.2, 0.1, 0.05], [8.5, 0.4, 0.15], [3.1, 0.08, 0.02], [10.0, 0.2, 0.1]]) 
    y = np.array([0, 1, 2, 1])
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"INFO: Dummy model saved to {MODEL_PATH}")

# --- AI INFERENCE FUNCTIONS ---

def load_ai_model(path):
    """Loads the trained ML model from disk."""
    try:
        model = joblib.load(path)
        print("STATUS: AI Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {path}. Training dummy model.")
        train_dummy_model()
        return joblib.load(path)


def feature_engineering(df: pd.DataFrame) -> np.ndarray:
    """
    Transforms raw time-series data into the features the AI model expects.
    This is the crucial step of turning data into intelligence.
    """
    print("STATUS: Starting Feature Engineering...")

    # Feature 1: Peak Force (Fz_N)
    peak_force = df['Fz_N'].max()

    # Feature 2: Peak Torque (Mx_Nm) - indicates max binding
    peak_torque = df['Mx_Nm'].abs().max()

    # Feature 3: Smoothness Score (Standard Deviation of Force Derivative)
    # Low standard deviation of force changes = smooth motion
    df['Fz_diff'] = df['Fz_N'].diff().fillna(0)
    smoothness_score = df['Fz_diff'].std()
    
    # The model expects an array of features: [peak_force, peak_torque, smoothness_score]
    features = np.array([[peak_force, peak_torque, smoothness_score]])
    
    print(f"INFO: Features calculated: {features}")
    return features


def predict_quality(df: pd.DataFrame, model) -> dict:
    """
    Runs inference on the collected data to predict quality.
    """
    
    # 1. Feature Extraction
    features = feature_engineering(df)
    
    # 2. Prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Map numerical prediction to a human-readable label
    labels = {0: "PASS - Good Performance", 1: "FAIL - Excessive Stiffness/Torque", 2: "FAIL - Too Loose/Low Force"}
    
    # 3. Decision Output
    result = {
        'Prediction_ID': int(prediction),
        'Quality_Result': labels.get(prediction, "UNKNOWN"),
        'Confidence_Score': float(np.max(probabilities)),
        'Peak_Fz_N': float(features[0, 0]),
        'Peak_Mx_Nm': float(features[0, 1]),
        'Smoothness_STD': float(features[0, 2])
    }
    
    return result

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Load the data collected from data_acquisition.py (we'll just call the function)
    # *In a real system, you would load the CSV file saved earlier.*
    from data_acquisition import run_inspection_cycle
    raw_data_df = run_inspection_cycle(cycle_id="VENT_TEST_INFERENCE_002")

    # 1. Load the AI Model
    quality_model = load_ai_model(MODEL_PATH)
    
    # 2. Run Inference
    final_result = predict_quality(raw_data_df, quality_model)
    
    # 3. Display Final Decision (Pass/Fail)
    print("\n--- AI INSPECTION RESULT ---")
    print(f"Decision: **{final_result['Quality_Result']}**")
    print(f"Confidence: {final_result['Confidence_Score']:.2f}")
    print("--- Detailed Metrics ---")
    print(f"Max Vertical Force (Fz): {final_result['Peak_Fz_N']:.2f} N")
    print(f"Max Binding Torque (Mx): {final_result['Peak_Mx_Nm']:.2f} Nm")
    print(f"Smoothness (Fz STD): {final_result['Smoothness_STD']:.3f}")
    
    # In a production environment, this result is sent to the PLC/Controller
    # to trigger sorting (Pass/Fail bin).