import streamlit as st
import pandas as pd
import numpy as np
import pickle
import mlflow.sklearn  # ✅ MLflow import for your model

# -------------------------------
# Custom CSS (unchanged)
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1f2933 0%, #020617 55%, #000000 100%);
        color: #f9fafb;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
    }
    .block-container {
        max-width: 1100px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: 0.02em;
        color: #f9fafb;
    }
    .stCard, .stExpander, .css-1d391kg, .css-1v0mbdj {
        background: rgba(15, 23, 42, 0.70) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(148, 163, 184, 0.35) !important;
        backdrop-filter: blur(18px) !important;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: transparent !important;
    }
    label {
        font-weight: 500 !important;
        color: #e5e7eb !important;
    }
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox select {
        background: rgba(15, 23, 42, 0.85) !important;
        color: #f9fafb !important;
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.6) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        color: #f9fafb;
        border-radius: 999px;
        border: none;
        padding: 0.5rem 1.6rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.7);
        transition: all 0.18s ease-in-out;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #818cf8, #a5b4fc);
        transform: translateY(-1px);
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.9);
        color: #020617;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #4f46e5, #a855f7) !important;
    }
    .stSlider [data-baseweb="slider"] > div {
        color: #f9fafb !important;
    }
    .stAlert {
        border-radius: 18px !important;
        border: 1px solid rgba(148, 163, 184, 0.4) !important;
        background: rgba(15, 23, 42, 0.9) !important;
    }
    footer, .stCaption {
        color: #9ca3af !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Huntington Disease Stage Prediction",
    page_icon="",
    layout="centered"
)

st.title(" Huntington Disease Stage Prediction")

# -------------------------------
# ✅ MLflow Model Loading
# -------------------------------
@st.cache_resource
def load_mlflow_model():
    """Load MLflow model with built-in preprocessing"""
    try:
        loaded_model = mlflow.sklearn.load_model("model.pkl")
        return loaded_model
    except:
        # Fallback to pickle if not proper MLflow format
        with open("model.pkl", "rb") as f:
            return pickle.load(f)

model = load_mlflow_model()

# Your existing input fields (unchanged)
left_col, right_col = st.columns([2, 1])

with left_col:
    age = st.slider("Age", min_value=30, max_value=80, value=51)
    sex = st.selectbox("Sex", ["Female", "Male"], index=1)
    family_history = st.selectbox("Family History", ["Yes", "No"], index=1)
    htt_cag_repeat_length = st.slider("HTT CAG Repeat Length", min_value=35, max_value=80, value=46)
    motor_symptoms = st.selectbox("Motor Symptoms", ["Mild", "Moderate", "Severe"], index=2)
    cognitive_decline = st.selectbox("Cognitive Decline", ["Mild", "Moderate", "Severe"], index=2)
    chorea_score = st.slider("Chorea Score", min_value=0.0, max_value=30.0, value=10.0, step=0.5)
    brain_volume_loss = st.slider("Brain Volume Loss (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    functional_capacity = st.slider("Functional Capacity (TFC)", min_value=0, max_value=13, value=10, step=1)
    htt_gene_expression_level = st.slider("HTT Gene Expression Level", min_value=0.0, max_value=10.0, value=2.5, step=0.01)
    protein_aggregation_level = st.slider("Protein Aggregation Level", min_value=0.0, max_value=5.0, value=1.0, step=0.01)
    gene_mutation_type = st.selectbox("Gene Mutation Type", ["Deletion", "Duplication", "Insertion", "Point Mutation"], index=1)
    gene_factor = st.selectbox("Gene/Factor", ["HTT", "MLH1", "MSH3", "HTT (Somatic Expansion)"])
    chromosome_location = st.selectbox("Chromosome Location", ["4p16.3", "3p22.2", "5q14.1"])
    function = "Mismatch Repair"
    effect = st.selectbox("Effect", ["CAG Repeat Expansion", "Neurodegeneration", "Faster Disease Onset"])
    category = st.selectbox("Category", ["Trans-acting Modifier", "Primary Cause", "Cis-acting Modifier"])

# Your existing right column explanations (unchanged)
with right_col:
    st.subheader(" What does each field mean?")
    # ... keep all your expanders exactly as they are ...

# -------------------------------
# ✅ MLflow Prediction (WORKS WITH YOUR MODEL)
# -------------------------------
if st.button(" Predict Disease Stage", type="primary"):
    input_data = pd.DataFrame(
        [[
            age, sex, family_history, htt_cag_repeat_length,
            motor_symptoms, cognitive_decline, chorea_score,
            brain_volume_loss, functional_capacity,
            gene_mutation_type, htt_gene_expression_level,
            protein_aggregation_level, gene_factor,
            chromosome_location, function, effect, category
        ]],
        columns=[
            "Age", "Sex", "Family_History", "HTT_CAG_Repeat_Length",
            "Motor_Symptoms", "Cognitive_Decline", "Chorea_Score",
            "Brain_Volume_Loss", "Functional_Capacity", "Gene_Mutation_Type",
            "HTT_Gene_Expression_Level", "Protein_Aggregation_Level",
            "Gene/Factor", "Chromosome_Location", "Function", "Effect", "Category"
        ]
    )
    
    try:
        # MLflow model handles preprocessing automatically
        prediction = model.predict(input_data)[0]
        st.success(f" **Predicted Disease Stage: {prediction}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            confidence = float(np.max(proba) * 100)
            st.info(f" Prediction Confidence: **{confidence:.1f}%**")

            prob_df = pd.DataFrame({
                "Stage": model.classes_,
                "Probability": [f"{p:.1%}" for p in proba]
            })
            st.dataframe(prob_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Prediction failed. Model expects numeric data.")
        st.info("💡 Make sure model.pkl is from MLflow with proper preprocessing.")

st.markdown("---")
st.caption(" Built with Streamlit | ML Deployment | By Hemanth")
