AI-Powered AC Vent Inspection Tool

This project showcases an Automated Quality Control (AQC) system that evaluates the mechanical “feel” and functional response of automotive AC vent louvers. The inspection relies entirely on load cell force data and actuator position measurements, replacing subjective manual tactile evaluations with objective, repeatable force-based metrics.

📌 Key Features
1. Objective Quality Control

Replaces subjective human assessment with quantifiable and repeatable metrics captured from the load cell sensor.

2. Actuator-Driven Testing

A two-actuator motion system performs precise, programmable interactions with the vent louver to measure mechanical resistance.

3. Hard Limit Enforcement

Strict engineering force limits are applied:

𝐹
𝑧
≤
4.0
 N
F
z
	​

≤4.0 N (vertical / up-down)

𝐹
𝑥
𝑦
≤
6.0
 N
F
xy
	​

≤6.0 N (side-to-side)

If exceeded, the sample fails immediately.

4. ML-Based Soft Limit Analysis

A lightweight Random-Forest classifier (simulated in this version) identifies subtle anomalies such as:

grittiness

binding

stick-slip motion

excessive friction

The classifier uses the force-displacement curve to detect patterns beyond human perception.

5. Streamlit Frontend

A clean interface for:

real-time visualization

interactive test configuration

pass/fail result display

force-position curve analysis

⚙️ System Architecture (Simulated Environment)

The system emulates a real production line workflow:

1. Data Acquisition

Simulated high-frequency force readings from a load cell sensor and actuator position sensors
(
𝑃
𝑥
,
𝑃
𝑦
,
𝑃
𝑧
P
x
	​

,P
y
	​

,P
z
	​

) at 1000 Hz.

2. Actuator Control

A 2-actuator mechanism performs controlled motion profiles to interact with the AC vent.

3. Feature Engineering

Raw time-series data is transformed into features such as:

peak forces

force variability

smoothness score

gradient changes

4. Inference Pipeline

Data checked against hard mechanical limits.

If safe, passed to the ML classifier for PASS/FAIL prediction.

5. Presentation

The Streamlit web app instantly displays:

test verdict

reason for PASS/FAIL

force vs. displacement graphs

🚀 Getting Started
Prerequisites

Python 3.8+

Git installed

1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/AI-Actuator-Inspection-Tool.git
cd AI-Actuator-Inspection-Tool

2. Create a Virtual Environment

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate


Windows

python -m venv .venv
.\.venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application

This also auto-creates a dummy ML model (inspection_model.joblib) on first run.

streamlit run app_frontend.py


Open your browser at:

http://localhost:8501

🧪 Usage & Testing

Choose scenarios from the sidebar:

Test Mode	Description	Expected Outcome
PASS (Simulated 1.0 N)	Smooth, low-force louver	PASS
FAIL – Fz Breach (5 N)	Excess vertical force (>4 N)	FAIL – Hard Limit
FAIL – Fxy Breach (7 N)	Excess side force (>6 N)	FAIL – Hard Limit
FAIL – Binding	Rough or sticky movement detected by ML	FAIL – ML

The result window displays:

PASS/FAIL decision

cause of failure

force vs. displacement curve

data summary

📝 File Structure
File	Description
app_frontend.py	Main Streamlit app handling simulation, feature extraction, plots, and ML inference.
requirements.txt	Python dependencies.
inspection_model.joblib	Dummy ML model (auto-generated on first run).
.gitignore	Excludes cache and temporary files.
🤝 Contributing

Contributions are welcome! Potential improvements:

Integrate with real actuator and load-cell hardware APIs

Replace dummy model with production-grade ML

Add predictive maintenance analytics for actuator health

📄 License

This project is licensed under the MIT License.
