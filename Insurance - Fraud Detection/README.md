# Insurance Fraud Detection System

Insurance fraud detection application built with Streamlit. Upload and analyze historical claims data to evaluate risk factors and classify potential fraud, featuring a dark-mode neumorphic interface for professional decision support.

## Tools Used

- **Streamlit** - Interactive web app framework
- **Scikit-learn** - Machine learning model inference
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Pickle** - Model serialization

## Features

- **Financial Analysis** - Monitors Reimbursement amounts (Inpatient/Outpatient) and Deductible amounts.
- **Patient Demographics** - Profiles Attending Physician, Gender, Race, and Renal Disease indicators.
- **Chronic Conditions** - Integrated tracking for 11 chronic conditions including Alzheimer's, Heart Failure, Kidney Disease, Cancer, Diabetes, and more.
- **Privacy First** - Local-only processing; data is not stored after prediction.
- **Interactive UI** - Dark-mode neumorphic design for professional data visualization.
- **Workflow-Driven** - Implements anomaly detection, ensemble modeling, threshold tuning, and audit pipelines.

## Steps to Run
note: model.pkl and preprocessing.pkl are too big and stored at : https://huggingface.co/spaces/hemanth003/InsuranceFraudDetection/tree/main

1. **Clone** the project
   ```bash 
   git clone https://github.com/hemanthreddyyanamala/Innomatics_Research_Labs.git
   cd Innomatics_Research_Labs/'Insurance - Fraud Detection'/
2. **Create** virtual environment
   ```bash
   python3 -m venv .venv
3. **Activate** .venv
   ```bash
   .venv\Scripts\activate # windows
   source .venv/bin/activate # linux/mac
4. **Install** dependencies/requirements:
   ```bash
   pip install -r requirements.txt
5. **Run** the app:
   ```bash
   streamlit run app.py
6. Open http://localhost:8501 in your browser <br>
Live at [https://huggingface.co/spaces/hemanth003/InsuranceFraudDetection]
