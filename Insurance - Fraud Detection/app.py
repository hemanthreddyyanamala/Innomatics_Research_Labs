import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Mode Neumorphic Design
st.markdown("""
    <style>
    /* Base dark neumorphic background */
    .main {
        background: linear-gradient(145deg, #1a1d29, #1e2130);
    }
    
    /* Neumorphic containers */
    .stApp {
        background: linear-gradient(145deg, #1a1d29, #1e2130);
    }
    
    /* Sidebar dark neumorphic styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(145deg, #161920, #1a1d29);
        box-shadow: 
            inset 5px 5px 10px #0f1117,
            inset -5px -5px 10px #252834;
    }
    
    /* Main header with dark neumorphic effect */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e0e6ed;
        text-align: center;
        margin-bottom: 2rem;
        padding: 30px;
        background: linear-gradient(145deg, #1a1d29, #1e2130);
        border-radius: 25px;
        box-shadow: 
            9px 9px 18px rgba(0, 0, 0, 0.4),
            -9px -9px 18px rgba(42, 48, 60, 0.4);
        letter-spacing: 1px;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #c9d1d9;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding: 15px 20px;
        background: linear-gradient(145deg, #1a1d29, #1e2130);
        border-radius: 15px;
        box-shadow: 
            6px 6px 12px rgba(0, 0, 0, 0.4),
            -6px -6px 12px rgba(42, 48, 60, 0.3);
    }
    
    /* Dark neumorphic prediction boxes */
    .prediction-box {
        padding: 30px;
        border-radius: 25px;
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .fraud-yes {
        background: linear-gradient(145deg, #2d1f1f, #3d2626);
        box-shadow: 
            9px 9px 18px rgba(0, 0, 0, 0.5),
            -9px -9px 18px rgba(55, 35, 35, 0.3),
            inset 2px 2px 4px rgba(239, 83, 80, 0.1);
    }
    
    .fraud-no {
        background: linear-gradient(145deg, #1f2d1f, #263d26);
        box-shadow: 
            9px 9px 18px rgba(0, 0, 0, 0.5),
            -9px -9px 18px rgba(35, 55, 35, 0.3),
            inset 2px 2px 4px rgba(102, 187, 106, 0.1);
    }
    
    /* Dark neumorphic info box */
    .info-box {
        background: linear-gradient(145deg, #1a1d29, #1e2130);
        padding: 20px;
        border-radius: 20px;
        margin: 15px 0;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.4),
            -8px -8px 16px rgba(42, 48, 60, 0.3);
        color: #c9d1d9;
        border: none;
    }
    
    /* Dark neumorphic buttons */
    .stButton > button {
        background: linear-gradient(145deg, #1a1d29, #1e2130);
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        color: #e0e6ed;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.5),
            -8px -8px 16px rgba(42, 48, 60, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 
            6px 6px 12px rgba(0, 0, 0, 0.5),
            -6px -6px 12px rgba(42, 48, 60, 0.4);
        transform: translateY(2px);
        background: linear-gradient(145deg, #1e2130, #22253a);
    }
    
    .stButton > button:active {
        box-shadow: 
            inset 4px 4px 8px rgba(0, 0, 0, 0.6),
            inset -4px -4px 8px rgba(42, 48, 60, 0.3);
    }
    
    /* Dark neumorphic input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        background: linear-gradient(145deg, #16181f, #1a1d29) !important;
        border: none !important;
        border-radius: 12px;
        padding: 12px 15px;
        color: #e0e6ed !important;
        box-shadow: 
            inset 4px 4px 8px rgba(0, 0, 0, 0.5),
            inset -4px -4px 8px rgba(42, 48, 60, 0.2);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #6e7681 !important;
    }
    
    /* Dark neumorphic select dropdown */
    .stSelectbox > div > div > div {
        background: linear-gradient(145deg, #16181f, #1a1d29) !important;
        color: #e0e6ed !important;
    }
    
    /* Dropdown menu styling */
    div[data-baseweb="select"] > div {
        background: linear-gradient(145deg, #16181f, #1a1d29) !important;
        color: #e0e6ed !important;
        border: none !important;
    }
    
    /* Dark neumorphic sliders */
    .stSlider > div > div > div > div {
        background: linear-gradient(145deg, #16181f, #1a1d29);
        border-radius: 10px;
        box-shadow: 
            inset 3px 3px 6px rgba(0, 0, 0, 0.5),
            inset -3px -3px 6px rgba(42, 48, 60, 0.2);
    }
    
    .stSlider > div > div > div > div > div {
        background: linear-gradient(145deg, #4a9eff, #357abd) !important;
        border-radius: 50%;
        box-shadow: 
            4px 4px 10px rgba(0, 0, 0, 0.6),
            -4px -4px 10px rgba(74, 158, 255, 0.3),
            inset 1px 1px 2px rgba(255, 255, 255, 0.2);
    }
    
    /* Slider track fill */
    .stSlider > div > div > div > div[role="slider"]::before {
        background: linear-gradient(90deg, #357abd, #4a9eff) !important;
    }
    
    /* Dark neumorphic expander */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #1a1d29, #1e2130);
        border-radius: 12px;
        box-shadow: 
            6px 6px 12px rgba(0, 0, 0, 0.4),
            -6px -6px 12px rgba(42, 48, 60, 0.3);
        color: #c9d1d9;
        font-weight: 600;
        border: none !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(145deg, #1e2130, #22253a);
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        background: linear-gradient(145deg, #16181f, #1a1d29);
        border: none;
    }
    
    /* Dark neumorphic metrics */
    div[data-testid="stMetricValue"] {
        color: #e0e6ed;
        font-weight: 700;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1a1d29, #1e2130);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.5),
            -8px -8px 16px rgba(42, 48, 60, 0.3);
    }
    
    /* Dark neumorphic file uploader */
    .stFileUploader > div {
        background: linear-gradient(145deg, #1a1d29, #1e2130);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.5),
            -8px -8px 16px rgba(42, 48, 60, 0.3);
    }
    
    .stFileUploader label {
        color: #c9d1d9 !important;
    }
    
    /* Dark neumorphic download button */
    .stDownloadButton > button {
        background: linear-gradient(145deg, #1a1d29, #1e2130);
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        color: #e0e6ed;
        font-weight: 600;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.5),
            -8px -8px 16px rgba(42, 48, 60, 0.4);
    }
    
    .stDownloadButton > button:hover {
        box-shadow: 
            6px 6px 12px rgba(0, 0, 0, 0.5),
            -6px -6px 12px rgba(42, 48, 60, 0.4);
        background: linear-gradient(145deg, #1e2130, #22253a);
    }
    
    /* Dark neumorphic dataframe */
    .stDataFrame {
        background: linear-gradient(145deg, #1a1d29, #1e2130);
        border-radius: 15px;
        padding: 10px;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.5),
            -8px -8px 16px rgba(42, 48, 60, 0.3);
    }
    
    /* Dataframe styling */
    .stDataFrame table {
        color: #c9d1d9 !important;
    }
    
    .stDataFrame thead tr th {
        background-color: #2d3139 !important;
        color: #e0e6ed !important;
    }
    
    .stDataFrame tbody tr:hover {
        background-color: #24272f !important;
    }
    
    /* Labels and text */
    label {
        color: #c9d1d9 !important;
        font-weight: 500 !important;
    }
    
    /* Remove default Streamlit styling that conflicts */
    .element-container {
        background: transparent !important;
    }
    
    /* Dark neumorphic alert boxes */
    .stAlert {
        background: linear-gradient(145deg, #1a1d29, #1e2130);
        border-radius: 15px;
        border: none;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.5),
            -8px -8px 16px rgba(42, 48, 60, 0.3);
        color: #c9d1d9;
    }
    
    /* Success message dark neumorphic */
    .stSuccess {
        background: linear-gradient(145deg, #1f2d1f, #263d26) !important;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.5),
            -8px -8px 16px rgba(35, 55, 35, 0.3);
        color: #81c784 !important;
    }
    
    /* Error message dark neumorphic */
    .stError {
        background: linear-gradient(145deg, #2d1f1f, #3d2626) !important;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.5),
            -8px -8px 16px rgba(55, 35, 35, 0.3);
        color: #ef5350 !important;
    }
    
    /* Warning message dark neumorphic */
    .stWarning {
        background: linear-gradient(145deg, #2d2620, #3d3426) !important;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.5),
            -8px -8px 16px rgba(55, 50, 35, 0.3);
        color: #ffb74d !important;
    }
    
    /* Info message dark neumorphic */
    .stInfo {
        background: linear-gradient(145deg, #1f252d, #26333d) !important;
        box-shadow: 
            8px 8px 16px rgba(0, 0, 0, 0.5),
            -8px -8px 16px rgba(35, 48, 55, 0.3);
        color: #64b5f6 !important;
    }
    
    /* Column dividers with depth */
    .row-widget {
        padding: 10px;
    }
    
    /* Spinner with dark neumorphic effect */
    .stSpinner > div {
        border-color: #4a9eff transparent transparent transparent !important;
    }
    
    /* Markdown text color */
    .stMarkdown {
        color: #c9d1d9;
    }
    
    /* Headers in markdown */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #e0e6ed;
    }
    
    /* Links */
    a {
        color: #58a6ff !important;
    }
    
    a:hover {
        color: #79c0ff !important;
    }
    
    /* Code blocks */
    code {
        background-color: #2d3139 !important;
        color: #79c0ff !important;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    /* Horizontal rule */
    hr {
        border-color: #30363d !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(145deg, #1a1d29, #1e2130);
        border-radius: 15px;
        padding: 5px;
        box-shadow: 
            inset 4px 4px 8px rgba(0, 0, 0, 0.4),
            inset -4px -4px 8px rgba(42, 48, 60, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8b949e;
        border-radius: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #1e2130, #22253a);
        color: #e0e6ed !important;
        box-shadow: 
            4px 4px 8px rgba(0, 0, 0, 0.5),
            -4px -4px 8px rgba(42, 48, 60, 0.3);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: linear-gradient(145deg, #16181f, #1a1d29);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(145deg, #2d3139, #383e47);
        border-radius: 5px;
        box-shadow: 
            inset 2px 2px 4px rgba(0, 0, 0, 0.4),
            inset -2px -2px 4px rgba(60, 68, 80, 0.3);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(145deg, #383e47, #434a55);
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: linear-gradient(145deg, #1a1d29, #1e2130);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 
            6px 6px 12px rgba(0, 0, 0, 0.4),
            -6px -6px 12px rgba(42, 48, 60, 0.3);
    }
    
    .stRadio label {
        color: #c9d1d9 !important;
    }
    
    /* Checkbox */
    .stCheckbox > label {
        color: #c9d1d9 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #357abd, #4a9eff) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Expected feature order (without target variable)
FEATURES = [
    'InscClaimAmtReimbursed', 'AttendingPhysician', 'DeductibleAmtPaid', 
    'Gender', 'Race', 'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
    'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
    'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
    'ChronicCond_Depression', 'ChronicCond_Diabetes',
    'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
    'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke',
    'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
    'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt'
]

MODEL_PATH = 'model.pkl'
PREPROC_PATH = 'preprocessor.pkl'


@st.cache_resource
def load_artifacts():
    """Load the preprocessor and model"""
    preprocessor = None
    model = None
    
    try:
        if os.path.exists(PREPROC_PATH):
            with open(PREPROC_PATH, 'rb') as f:
                preprocessor = pickle.load(f)
            #st.success("✅ Preprocessor loaded successfully!")
        else:
            st.error(f"❌ Preprocessor file not found: {PREPROC_PATH}")
    except Exception as e:
        st.error(f"❌ Error loading preprocessor: {str(e)}")
    
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            #st.success("✅ Model loaded successfully!")
        else:
            st.error(f"❌ Model file not found: {MODEL_PATH}")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
    
    return preprocessor, model


def prepare_input(data_dict):
    """Convert input dictionary to DataFrame with correct feature order"""
    df = pd.DataFrame([data_dict])
    
    # Ensure all features are present
    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing features
    
    # Reorder columns to match training
    df = df[FEATURES]
    return df


def predict_fraud(preprocessor, model, df):
    """Make prediction using preprocessor and model"""
    try:
        # Preprocess the data
        X_processed = preprocessor.transform(df)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_processed)[0]
            probability = proba[1] if len(proba) >= 2 else None
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None


def display_prediction_result(prediction, probability):
    """Display prediction result with dark neumorphic styling"""
    if prediction is None:
        st.error("Unable to make prediction")
        return
    
    # Determine fraud status
    is_fraud = (prediction == 1 or prediction == 'Yes')
    
    # Display result with dark neumorphic design
    if is_fraud:
        st.markdown(f"""
            <div class="prediction-box fraud-yes">
                <h2 style="color: #ef5350; margin: 0; font-weight: 700; font-size: 2rem;"> POTENTIAL FRAUD DETECTED</h2>
                <p style="font-size: 1.3rem; margin-top: 15px; color: #c9d1d9; font-weight: 600;">
                    Fraud Probability: <strong style="color: #ef5350;">{probability*100:.2f}%</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-box fraud-no">
                <h2 style="color: #66bb6a; margin: 0; font-weight: 700; font-size: 2rem;"> NO FRAUD DETECTED</h2>
                <p style="font-size: 1.3rem; margin-top: 15px; color: #c9d1d9; font-weight: 600;">
                    Fraud Probability: <strong style="color: #66bb6a;">{probability*100:.2f}%</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="main-header">Healthcare Fraud Detection System</div>', 
                unsafe_allow_html=True)
    
    # Load model and preprocessor
    with st.spinner("Loading model and preprocessor..."):
        preprocessor, model = load_artifacts()
    
    if model is None or preprocessor is None:
        st.error("Cannot proceed without model and preprocessor. Please ensure both files exist.")
        return
    
    # Sidebar
    st.sidebar.title(" Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Single Prediction", "Batch Prediction", "Feature Information", "About"]
    )
    
    if app_mode == "Single Prediction":
        single_prediction_mode(preprocessor, model)
    elif app_mode == "Batch Prediction":
        batch_prediction_mode(preprocessor, model)
    elif app_mode == "Feature Information":
        feature_information_page()
    else:
        about_page()


def single_prediction_mode(preprocessor, model):
    """Single prediction interface with form inputs"""
    st.markdown('<div class="sub-header">Enter Claim Details</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <strong style="color: #e0e6ed; font-size: 1.1rem;">ℹ Instructions:</strong> 
            <span style="color: #c9d1d9;">Fill in the claim information below to check for potential fraud.
            All fields are required for accurate prediction.</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header"> Financial Information</p>', unsafe_allow_html=True)
        insc_claim_amt = st.slider(
            "Insurance Claim Amount Reimbursed ($)",
            min_value=0, max_value=125000, value=80, step=10
        )
        deductible_amt = st.slider(
            "Deductible Amount Paid ($)",
            min_value=0.0, max_value=1068.0, value=0.0, step=10.0
        )
        ip_annual_reimbursement = st.slider(
            "IP Annual Reimbursement Amount ($)",
            min_value=-8000, max_value=161470, value=0, step=100
        )
        ip_annual_deductible = st.slider(
            "IP Annual Deductible Amount ($)",
            min_value=0, max_value=38272, value=0, step=50
        )
        op_annual_reimbursement = st.slider(
            "OP Annual Reimbursement Amount ($)",
            min_value=-70, max_value=102960, value=1170, step=100
        )
        op_annual_deductible = st.slider(
            "OP Annual Deductible Amount ($)",
            min_value=0, max_value=13840, value=340, step=50
        )
    
    with col2:
        st.markdown('<p class="sub-header"> Patient Information</p>', unsafe_allow_html=True)
        attending_physician = st.text_input(
            "Attending Physician ID",
            value="PHY330576",
            help="Enter physician ID (e.g., PHY330576)"
        )
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        race = st.selectbox(
            "Race", 
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1: "White", 2: "Black", 3: "Other", 4: "Hispanic", 5: "Asian"}[x]
        )
        renal_disease = st.selectbox(
            "Renal Disease Indicator",
            options=["0", "Y"],
            format_func=lambda x: "No" if x == "0" else "Yes"
        )
        
        st.markdown('<p class="sub-header"> Chronic Conditions</p>', unsafe_allow_html=True)
        chronic_cond_map = {
            "Alzheimer": 'ChronicCond_Alzheimer',
            "Heart Failure": 'ChronicCond_Heartfailure',
            "Kidney Disease": 'ChronicCond_KidneyDisease',
            "Cancer": 'ChronicCond_Cancer',
            "Obstructive Pulmonary": 'ChronicCond_ObstrPulmonary',
            "Depression": 'ChronicCond_Depression',
            "Diabetes": 'ChronicCond_Diabetes',
            "Ischemic Heart": 'ChronicCond_IschemicHeart',
            "Osteoporosis": 'ChronicCond_Osteoporasis',
            "Rheumatoid Arthritis": 'ChronicCond_rheumatoidarthritis',
            "Stroke": 'ChronicCond_stroke'
        }
        
        chronic_conditions = {}
        for display_name, col_name in chronic_cond_map.items():
            chronic_conditions[col_name] = st.selectbox(
                f"{display_name}",
                options=[1, 2],
                format_func=lambda x: "No" if x == 1 else "Yes",
                key=col_name
            )
    
    # Predict button
    st.markdown("---")
    if st.button("Predict Fraud", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'InscClaimAmtReimbursed': insc_claim_amt,
            'AttendingPhysician': attending_physician,
            'DeductibleAmtPaid': deductible_amt,
            'Gender': gender,
            'Race': race,
            'RenalDiseaseIndicator': renal_disease,
            'IPAnnualReimbursementAmt': ip_annual_reimbursement,
            'IPAnnualDeductibleAmt': ip_annual_deductible,
            'OPAnnualReimbursementAmt': op_annual_reimbursement,
            'OPAnnualDeductibleAmt': op_annual_deductible
        }
        input_data.update(chronic_conditions)
        
        # Prepare dataframe
        df = prepare_input(input_data)
        
        # Make prediction
        with st.spinner("Analyzing claim data..."):
            prediction, probability = predict_fraud(preprocessor, model, df)
        
        # Display result
        if prediction is not None and probability is not None:
            display_prediction_result(prediction, probability)
            
            # Show input summary
            with st.expander(" View Input Summary"):
                st.dataframe(df, use_container_width=True)


def batch_prediction_mode(preprocessor, model):
    """Batch prediction interface for CSV upload"""
    st.markdown('<div class="sub-header">Batch Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <strong style="color: #e0e6ed; font-size: 1.1rem;">ℹ Instructions:</strong> 
            <span style="color: #c9d1d9;">Upload a CSV file with claim data. 
            The file should contain all required feature columns.</span>
        </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing claim records"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f" File uploaded successfully! Found {len(df)} records.")
            
            # Show preview
            with st.expander(" Preview Data (First 5 rows)"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Check for required features
            missing_features = [f for f in FEATURES if f not in df.columns]
            
            if missing_features:
                st.warning(f" Missing features: {', '.join(missing_features)}")
                st.info("Missing features will be filled with default values (0)")
            
            # Remove target column if present
            if 'PotentialFraud' in df.columns:
                actual_labels = df['PotentialFraud'].copy()
                df = df.drop(columns=['PotentialFraud'])
                has_actual = True
            else:
                has_actual = False
            
            # Ensure all features are present
            for feature in FEATURES:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Reorder columns
            df = df[FEATURES]
            
            # Predict button
            if st.button(" Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    try:
                        # Preprocess
                        X_processed = preprocessor.transform(df)
                        
                        # Predict
                        predictions = model.predict(X_processed)
                        
                        # Get probabilities
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(X_processed)[:, 1]
                        else:
                            probabilities = None
                        
                        # Create results dataframe
                        results_df = df.copy()
                        results_df['Predicted_Fraud'] = ['Yes' if p == 1 or p == 'Yes' else 'No' 
                                                          for p in predictions]
                        if probabilities is not None:
                            results_df['Fraud_Probability'] = probabilities
                        
                        if has_actual:
                            results_df['Actual_Fraud'] = actual_labels
                            results_df['Correct'] = (results_df['Predicted_Fraud'] == results_df['Actual_Fraud'])
                        
                        # Display summary statistics
                        st.markdown("###  Prediction Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        fraud_count = (results_df['Predicted_Fraud'] == 'Yes').sum()
                        no_fraud_count = (results_df['Predicted_Fraud'] == 'No').sum()
                        
                        with col1:
                            st.metric("Total Records", len(results_df))
                        with col2:
                            st.metric("Predicted Fraud", fraud_count, delta=f"{fraud_count/len(results_df)*100:.1f}%")
                        with col3:
                            st.metric("Predicted No Fraud", no_fraud_count, delta=f"{no_fraud_count/len(results_df)*100:.1f}%")
                        
                        # If actual labels are available, show accuracy
                        if has_actual:
                            accuracy = (results_df['Correct'].sum() / len(results_df)) * 100
                            st.metric("Accuracy", f"{accuracy:.2f}%")
                        
                        # Show results
                        st.markdown("###  Prediction Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f" Error during batch prediction: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f" Error reading CSV file: {str(e)}")
            st.exception(e)




def feature_information_page():
    """Feature information page with detailed descriptions"""
    st.markdown('<div class="sub-header"> Feature Information</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <strong style="color: #e0e6ed; font-size: 1.1rem;">ℹ About Features:</strong> 
            <span style="color: #c9d1d9;">This page provides detailed information about all 21 features used by the fraud detection model. 
            Understanding these features will help you provide accurate input data for better predictions.</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different feature categories
    tab1, tab2, tab3 = st.tabs([" Financial Features", " Patient Demographics", " Chronic Conditions"])
    
    with tab1:
        st.markdown("### Financial Features (6 features)")
        st.markdown("These features represent the financial aspects of insurance claims and annual costs.")
        
        # Insurance Claim Amount Reimbursed
        with st.expander(" Insurance Claim Amount Reimbursed", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                **Description:**  
                The amount reimbursed by insurance for a specific claim.
                
                **What it means:**  
                This represents the actual payout made by the insurance company to cover medical expenses. 
                Higher amounts might indicate more complex procedures or potential over-billing.
                
                **Fraud Indicator:**  
                - Unusually high reimbursement amounts (>$10,000) may signal fraudulent claims
                - Claims with round numbers (e.g., exactly $5,000) might be suspicious
                - Patterns of consistently high reimbursements warrant investigation
                """)
            with col2:
                st.markdown("""
                **Statistics:**
                - **Min:** $0
                - **Max:** $125,000
                - **Average:** $979
                - **Median:** $80
                - **Common Range:** $40 - $300
                """)
        
        # Deductible Amount Paid
        with st.expander(" Deductible Amount Paid", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                **Description:**  
                The out-of-pocket amount paid by the patient before insurance coverage kicks in.
                
                **What it means:**  
                This is the portion of medical costs that the patient must pay themselves. Most values are $0, 
                indicating insurance covered the full amount after deductible was met.
                
                **Fraud Indicator:**  
                - Consistently zero deductibles across multiple claims might indicate billing manipulation
                - Maximum deductible ($1,068) paid repeatedly could signal coordinated fraud
                - Deductible patterns inconsistent with policy terms
                """)
            with col2:
                st.markdown("""
                **Statistics:**
                - **Min:** $0
                - **Max:** $1,068
                - **Average:** $78
                - **Median:** $0
                - **75th percentile:** $0
                """)
        
        # IP Annual Reimbursement Amount
        with st.expander(" IP (Inpatient) Annual Reimbursement Amount", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                **Description:**  
                Total amount reimbursed for inpatient (hospital stay) services over a year.
                
                **What it means:**  
                Inpatient care includes hospitalizations, surgeries, and overnight stays. Higher amounts 
                indicate more serious medical conditions or extended hospital stays.
                
                **Fraud Indicator:**  
                - Extremely high annual amounts (>$50,000) without corresponding diagnosis codes
                - Multiple high-cost inpatient stays in a short period
                - Negative values indicate refunds or billing corrections (potential fraud reversal)
                """)
            with col2:
                st.markdown("""
                **Statistics:**
                - **Min:** -$8,000
                - **Max:** $161,470
                - **Average:** $5,177
                - **Median:** $0
                - **75th percentile:** $5,640
                """)
        
        # IP Annual Deductible Amount
        with st.expander(" IP (Inpatient) Annual Deductible Amount", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                **Description:**  
                Total deductible paid by patient for inpatient services over a year.
                
                **What it means:**  
                This represents the patient's annual out-of-pocket costs for hospital stays and inpatient care 
                before insurance coverage begins.
                
                **Fraud Indicator:**  
                - Deductible amounts exceeding typical policy maximums ($38,272 is suspiciously high)
                - Zero deductibles with high reimbursements might indicate billing fraud
                - Inconsistent deductible patterns across similar procedures
                """)
            with col2:
                st.markdown("""
                **Statistics:**
                - **Min:** $0
                - **Max:** $38,272
                - **Average:** $566
                - **Median:** $0
                - **75th percentile:** $1,068
                """)
        
        # OP Annual Reimbursement Amount
        with st.expander(" OP (Outpatient) Annual Reimbursement Amount", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                **Description:**  
                Total amount reimbursed for outpatient (no overnight stay) services over a year.
                
                **What it means:**  
                Outpatient care includes doctor visits, tests, minor procedures, and treatments that don't 
                require hospitalization. This is typically lower than inpatient costs.
                
                **Fraud Indicator:**  
                - Unusually high outpatient costs (>$50,000) may indicate billing for unnecessary services
                - Negative values suggest billing corrections or fraud detection
                - Patterns of maximum reimbursements across multiple patients
                """)
            with col2:
                st.markdown("""
                **Statistics:**
                - **Min:** -$70
                - **Max:** $102,960
                - **Average:** $2,278
                - **Median:** $1,170
                - **Common Range:** $460 - $2,590
                """)
        
        # OP Annual Deductible Amount
        with st.expander(" OP (Outpatient) Annual Deductible Amount", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                **Description:**  
                Total deductible paid by patient for outpatient services over a year.
                
                **What it means:**  
                Patient's annual out-of-pocket costs for outpatient care. Higher values indicate more 
                frequent medical services or high-deductible insurance plans.
                
                **Fraud Indicator:**  
                - Deductibles consistently at policy maximums across many patients
                - Zero deductibles with high service utilization (potential billing errors)
                - Unusual patterns in deductible vs. reimbursement ratios
                """)
            with col2:
                st.markdown("""
                **Statistics:**
                - **Min:** $0
                - **Max:** $13,840
                - **Average:** $650
                - **Median:** $340
                - **Common Range:** $120 - $790
                """)
    
    with tab2:
        st.markdown("### Patient Demographics (4 features)")
        st.markdown("These features describe patient characteristics and their healthcare providers.")
        
        # Attending Physician
        with st.expander(" Attending Physician", expanded=False):
            st.markdown("""
            **Description:**  
            Unique identifier for the physician who provided care (e.g., PHY330576).
            
            **What it means:**  
            This tracks which doctor is treating the patient. There are 81,953 unique physicians in the dataset, 
            with some doctors treating significantly more patients than others.
            
            **Fraud Indicator:**  
            - Physicians with unusually high billing amounts compared to peers
            - Doctors with high percentages of fraud claims in their patient portfolio
            - Patterns of identical claims across different patients from same physician
            - Geographic anomalies (physician billing for patients in distant locations)
            
            **Top Physicians by Volume:**
            - PHY330576: 2,534 claims (most frequent)
            - PHY350277: 1,628 claims
            - PHY412132: 1,321 claims
            """)
        
        # Gender
        with st.expander("Gender", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                **Description:**  
                Patient's gender encoded as: **1 = Male**, **2 = Female**
                
                **What it means:**  
                Gender can influence healthcare needs and costs due to biological differences, 
                gender-specific conditions, and treatment patterns.
                
                **Fraud Indicator:**  
                - Gender-inappropriate procedures (e.g., prostate surgery billed for females)
                - Claims for pregnancy-related services for male patients
                - Patterns of fraud may correlate with gender in certain fraud schemes
                """)
            with col2:
                st.markdown("""
                **Distribution:**
                - **Average:** 1.58
                - More females (2) than males (1)
                - **Male (1):** ~42%
                - **Female (2):** ~58%
                """)
        
        # Race
        with st.expander(" Race", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                **Description:**  
                Patient's race/ethnicity encoded as:
                - **1 = White**
                - **2 = Black**
                - **3 = Other**
                - **4 = Hispanic**
                - **5 = Asian**
                
                **What it means:**  
                Race can correlate with health disparities, access to care, and disease prevalence. 
                This helps model understand demographic patterns in healthcare utilization.
                
                **Fraud Indicator:**  
                - Not directly indicative of fraud
                - May help identify geographic fraud patterns
                - Can reveal targeting of specific communities by fraudulent providers
                """)
            with col2:
                st.markdown("""
                **Distribution:**
                - **Average:** 1.25
                - Heavily weighted toward White (1)
                - **White (1):** ~75%
                - **Others:** ~25%
                """)
        
        # Renal Disease Indicator
        with st.expander(" Renal Disease Indicator", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                **Description:**  
                Indicates whether patient has end-stage renal disease (kidney failure).
                - **0 = No renal disease**
                - **Y = Yes, has renal disease**
                
                **What it means:**  
                Patients with renal disease require dialysis or kidney transplants, resulting in very high 
                medical costs. This is a critical indicator for legitimate high-cost claims.
                
                **Fraud Indicator:**  
                - Claims for dialysis without renal disease indicator
                - Renal disease coded but no corresponding treatments billed
                - Sudden appearance of renal disease in records without prior kidney issues
                - Overcharging for dialysis services
                """)
            with col2:
                st.markdown("""
                **Distribution:**
                - **No (0):** 446,693 (80%)
                - **Yes (Y):** 109,120 (20%)
                
                **Impact:**  
                Renal patients have significantly higher legitimate costs
                """)
    
    with tab3:
        st.markdown("### Chronic Conditions (11 features)")
        st.markdown("Binary indicators for chronic health conditions. **1 = No**, **2 = Yes**")
        
        chronic_conditions = {
            " Alzheimer's Disease": {
                "description": "Progressive brain disorder affecting memory, thinking, and behavior.",
                "fraud_indicators": [
                    "Claims for Alzheimer's medication without diagnosis",
                    "Expensive treatments for early-stage patients",
                    "Duplicate billing for memory care services"
                ],
                "prevalence": "40% of patients (2 = Yes)",
                "avg": 1.60
            },
            " Heart Failure": {
                "description": "Chronic condition where heart cannot pump blood effectively.",
                "fraud_indicators": [
                    "Multiple hospitalizations without supporting diagnostics",
                    "Expensive cardiac procedures billed repeatedly",
                    "Heart failure coded without echocardiogram or other tests"
                ],
                "prevalence": "41% of patients",
                "avg": 1.41
            },
            " Kidney Disease": {
                "description": "Chronic kidney disease (distinct from end-stage renal disease).",
                "fraud_indicators": [
                    "Kidney disease claims without lab work (creatinine, GFR)",
                    "Advanced treatments without documented disease progression",
                    "Billing for kidney-related procedures inconsistent with stage"
                ],
                "prevalence": "59% of patients",
                "avg": 1.59
            },
            " Cancer": {
                "description": "Any form of cancer diagnosis.",
                "fraud_indicators": [
                    "Chemotherapy billed without cancer diagnosis",
                    "Multiple rounds of expensive treatments without remission checks",
                    "Cancer medications prescribed for non-cancer conditions",
                    "Billing for experimental treatments as standard care"
                ],
                "prevalence": "85% of patients",
                "avg": 1.85
            },
            " Obstructive Pulmonary Disease (COPD)": {
                "description": "Chronic lung disease including emphysema and chronic bronchitis.",
                "fraud_indicators": [
                    "COPD medications without diagnosis or pulmonary function tests",
                    "Oxygen therapy billed without documented need",
                    "Unnecessary breathing treatments and equipment"
                ],
                "prevalence": "69% of patients",
                "avg": 1.69
            },
            " Depression": {
                "description": "Chronic mental health condition affecting mood and functioning.",
                "fraud_indicators": [
                    "Antidepressants prescribed without documented diagnosis",
                    "Excessive therapy sessions beyond insurance limits",
                    "Billing for services not actually rendered"
                ],
                "prevalence": "57% of patients",
                "avg": 1.57
            },
            " Diabetes": {
                "description": "Chronic condition affecting blood sugar regulation.",
                "fraud_indicators": [
                    "Diabetes supplies billed in excessive quantities",
                    "Insulin prescribed without diabetes diagnosis",
                    "Expensive diabetes medications when cheaper alternatives exist",
                    "Claims without required HbA1c monitoring"
                ],
                "prevalence": "30% of patients",
                "avg": 1.30
            },
            " Ischemic Heart Disease": {
                "description": "Reduced blood flow to heart muscle (coronary artery disease).",
                "fraud_indicators": [
                    "Cardiac stents or procedures billed without supporting tests",
                    "Multiple interventions without documented progression",
                    "Ischemic heart disease coded without stress tests or angiograms"
                ],
                "prevalence": "24% of patients",
                "avg": 1.24
            },
            " Osteoporosis": {
                "description": "Bone density loss leading to fracture risk.",
                "fraud_indicators": [
                    "Osteoporosis medications without bone density scans",
                    "Treatments billed for young patients (uncommon)",
                    "Expensive biologics prescribed as first-line treatment"
                ],
                "prevalence": "68% of patients",
                "avg": 1.68
            },
            " Rheumatoid Arthritis": {
                "description": "Autoimmune disease causing joint inflammation.",
                "fraud_indicators": [
                    "Expensive biologics prescribed without failed first-line treatments",
                    "Rheumatoid arthritis diagnosis without supporting lab tests (RF, anti-CCP)",
                    "Excessive infusion treatments billed"
                ],
                "prevalence": "69% of patients",
                "avg": 1.69
            },
            "Brain Stroke": {
                "description": "Brain damage from interrupted blood flow.",
                "fraud_indicators": [
                    "Stroke rehabilitation billed without documented stroke event",
                    "Long-term expensive therapies without progress notes",
                    "Stroke diagnosis without imaging (CT/MRI)"
                ],
                "prevalence": "90% of patients",
                "avg": 1.90
            }
        }
        
        for condition_name, info in chronic_conditions.items():
            with st.expander(f"{condition_name}", expanded=False):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"""
                    **Description:**  
                    {info['description']}
                    
                    **Fraud Indicators:**
                    """)
                    for indicator in info['fraud_indicators']:
                        st.markdown(f"- {indicator}")
                    
                    st.markdown("""
                    **Encoding:**  
                    1 = Patient does NOT have this condition  
                    2 = Patient HAS this condition
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Statistics:**
                    - **Average:** {info['avg']}
                    - **Prevalence:** {info['prevalence']}
                    
                    **Impact:**  
                    Higher costs and service utilization expected
                    """)
    
    # Summary section
    st.markdown("---")
    st.markdown("###  Key Takeaways")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #1a1d29, #1e2130); 
                    padding: 20px; border-radius: 15px; 
                    box-shadow: 8px 8px 16px rgba(0, 0, 0, 0.5), 
                                -8px -8px 16px rgba(42, 48, 60, 0.3);">
            <h4 style="color: #4a9eff; margin-top: 0;"> Financial Red Flags</h4>
            <ul style="color: #c9d1d9; font-size: 0.9rem;">
                <li>Unusually high reimbursements</li>
                <li>Round-number billing</li>
                <li>Inconsistent deductibles</li>
                <li>Negative amounts (refunds)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #1a1d29, #1e2130); 
                    padding: 20px; border-radius: 15px; 
                    box-shadow: 8px 8px 16px rgba(0, 0, 0, 0.5), 
                                -8px -8px 16px rgba(42, 48, 60, 0.3);">
            <h4 style="color: #66bb6a; margin-top: 0;">👤 Demographic Patterns</h4>
            <ul style="color: #c9d1d9; font-size: 0.9rem;">
                <li>Provider billing patterns</li>
                <li>Gender-inappropriate procedures</li>
                <li>Geographic anomalies</li>
                <li>Targeted fraud schemes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #1a1d29, #1e2130); 
                    padding: 20px; border-radius: 15px; 
                    box-shadow: 8px 8px 16px rgba(0, 0, 0, 0.5), 
                                -8px -8px 16px rgba(42, 48, 60, 0.3);">
            <h4 style="color: #ef5350; margin-top: 0;"> Medical Inconsistencies</h4>
            <ul style="color: #c9d1d9; font-size: 0.9rem;">
                <li>Treatments without diagnoses</li>
                <li>Missing required tests</li>
                <li>Excessive service utilization</li>
                <li>Inappropriate medications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def about_page():
    """About page with system information"""
    st.markdown('<div class="sub-header">About This System</div>', unsafe_allow_html=True)
    
    st.markdown("""
        ###  Healthcare Fraud Detection System
        
        This application uses machine learning to detect potential fraud in healthcare insurance claims.
        
        ####  Features
        - **Single Prediction**: Analyze individual claims for fraud detection
        - **Batch Prediction**: Process multiple claims from CSV files
        - **Real-time Analysis**: Get instant predictions with probability scores
        
        ####  Model Information
        - **Algorithm**: Bagging Classifier
        - **Preprocessing**: RobustScaler for numerical features, OrdinalEncoder for categorical features
        - **Features**: 21 input features including financial amounts, patient demographics, and chronic conditions
        
        ####  Required Features
        The model uses the following features for prediction:
    """)
    
    # Display features in a nice format
    feature_categories = {
        " Financial Features": [
            "InscClaimAmtReimbursed", "DeductibleAmtPaid", 
            "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
            "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt"
        ],
        " Patient Demographics": [
            "AttendingPhysician", "Gender", "Race", "RenalDiseaseIndicator"
        ],
        " Chronic Conditions": [
            "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", 
            "ChronicCond_KidneyDisease", "ChronicCond_Cancer",
            "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
            "ChronicCond_Diabetes", "ChronicCond_IschemicHeart",
            "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis",
            "ChronicCond_stroke"
        ]
    }
    
    for category, features in feature_categories.items():
        with st.expander(category):
            for feature in features:
                st.write(f"- {feature}")
    
    st.markdown("---")
    st.markdown("""
        ####  Privacy & Security
        - All predictions are made locally on your machine
        - No data is sent to external servers
        - Input data is not stored after prediction
        
        ####  Support
        For questions or issues, please contact your system administrator.
    """)


if __name__ == '__main__':
    main()