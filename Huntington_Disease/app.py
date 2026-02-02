import streamlit as st
import pandas as pd
import numpy as np
import pickle



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
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Huntington Disease Stage Prediction")

@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model, None  # Return None for preprocessor

model, preprocessor = load_artifacts()  # preprocessor will be None


left_col, right_col = st.columns([2, 1])


with left_col:
    age = st.slider("Age", min_value=30, max_value=80, value=51)

    sex = st.selectbox("Sex", ["Female", "Male"], index=1)

    family_history = st.selectbox("Family History", ["Yes", "No"], index=1)

    htt_cag_repeat_length = st.slider(
        "HTT CAG Repeat Length",
        min_value=35, max_value=80, value=46
    )

    motor_symptoms = st.selectbox(
        "Motor Symptoms",
        ["Mild", "Moderate", "Severe"],
        index=2
    )

    cognitive_decline = st.selectbox(
        "Cognitive Decline",
        ["Mild", "Moderate", "Severe"],
        index=2
    )

    chorea_score = st.slider(
        "Chorea Score",
        min_value=0.0, max_value=30.0, value=10.0, step=0.5
    )

    brain_volume_loss = st.slider(
        "Brain Volume Loss (%)",
        min_value=0.0, max_value=15.0, value=5.0, step=0.1
    )

    functional_capacity = st.slider(
        "Functional Capacity (TFC)",
        min_value=0, max_value=13, value=10, step=1
    )

    htt_gene_expression_level = st.slider(
        "HTT Gene Expression Level",
        min_value=0.0, max_value=10.0, value=2.5, step=0.01
    )

    protein_aggregation_level = st.slider(
        "Protein Aggregation Level",
        min_value=0.0, max_value=5.0, value=1.0, step=0.01
    )

    gene_mutation_type = st.selectbox(
        "Gene Mutation Type",
        ["Deletion", "Duplication", "Insertion", "Point Mutation"],
        index=1
    )

    gene_factor = st.selectbox(
        "Gene/Factor",
        ["HTT", "MLH1", "MSH3", "HTT (Somatic Expansion)"]
    )

    chromosome_location = st.selectbox(
        "Chromosome Location",
        ["4p16.3", "3p22.2", "5q14.1"]
    )

    function = "Mismatch Repair"

    effect = st.selectbox(
        "Effect",
        ["CAG Repeat Expansion", "Neurodegeneration", "Faster Disease Onset"]
    )

    category = st.selectbox(
        "Category",
        ["Trans-acting Modifier", "Primary Cause", "Cis-acting Modifier"]
    )

with right_col:
    st.subheader("ℹ️ What does each field mean?")

    with st.expander("Age"):
        st.markdown(
            "How old the patient is in years. Older age can change when and how strongly symptoms appear."
        )

    with st.expander("Sex"):
        st.markdown(
            "Whether the patient is male or female. In some diseases, symptoms can differ slightly between boys and girls."
        )

    with st.expander("Family History"):
        st.markdown(
            "Shows if close family members also have Huntington's disease. If **Yes**, the chance of having it is higher."
        )

    with st.expander("HTT CAG Repeat Length"):
        st.markdown(
            "A count of how many times the letters **CAG** repeat in the HTT gene. Larger numbers usually mean earlier and more severe disease."
        )

    with st.expander("Motor Symptoms"):
        st.markdown(
            "- **Mild**: Small problems with movement.\n"
            "- **Moderate**: Clear movement problems, but still can do many tasks.\n"
            "- **Severe**: Strong shaking or difficulty walking and moving."
        )

    with st.expander("Cognitive Decline"):
        st.markdown(
            "How much thinking and memory are affected:\n"
            "- **Mild**: Slight trouble remembering or planning.\n"
            "- **Moderate**: Noticeable problems in daily life.\n"
            "- **Severe**: Big difficulties with thinking, memory, and decisions."
        )

    with st.expander("Chorea Score"):
        st.markdown(
            "A number that tells how bad the 'dance‑like' movements are. **0** means no extra movements; higher numbers mean more movement problems."
        )

    with st.expander("Brain Volume Loss"):
        st.markdown(
            "How much the brain has shrunk, in percent. More loss usually means more damage to brain cells."
        )

    with st.expander("Functional Capacity (TFC)"):
        st.markdown(
            "How well the person can do everyday activities like walking, eating, working or dressing. Higher scores mean more independence."
        )

    with st.expander("HTT Gene Expression Level"):
        st.markdown(
            "How much of the Huntington (HTT) protein the body is making. Higher levels can be linked to more cell stress and damage."
        )

    with st.expander("Protein Aggregation Level"):
        st.markdown(
            "How much of the harmful HTT protein is clumping together inside brain cells. More clumps usually means more damage."
        )

    with st.expander("Gene Mutation Type"):
        st.markdown(
            "What kind of change happened in the DNA instruction:\n"
            "- **Deletion**: A piece is missing.\n"
            "- **Duplication**: A piece is copied twice.\n"
            "- **Insertion**: An extra piece is added.\n"
            "- **Point Mutation**: One letter is changed."
        )

    with st.expander("Gene/Factor"):
        st.markdown("""
        This tells **which DNA instruction is involved** in the disease:
        - **HTT** – The main Huntington disease gene; changes here directly cause the illness.  
        - **MLH1** – A DNA–repair helper gene that can change how strong or how early the disease is.  
        - **MSH3** – Another DNA–repair helper that can affect how much the CAG repeat grows.  
        - **HTT (Somatic Expansion)** – The same HTT gene, but this name is used when its CAG repeats keep growing inside brain cells over time.
        """)

    with st.expander("Chromosome Location"):
        st.markdown("""
        **What this means:**  
        This is the **DNA address** where an important gene is found.  
        Each code tells which chromosome and which small region on it.
        - **4p16.3**  
        The address of the **HTT gene**, the main gene that causes Huntington's disease when its CAG repeat is too long.  
        - **3p22.2**  
        The address of the **MLH1 gene**, a DNA‑repair gene that can change how the disease behaves.  
        - **5q14.1**  
        The address of the **MSH3 gene**, another DNA‑repair gene that can affect how much the CAG repeat grows.
        """)

    with st.expander("Function"):
        st.markdown(
            "What the gene normally does. For example, **Mismatch Repair** genes fix mistakes in DNA, helping keep cells healthy."
        )

    with st.expander("Effect"):
        st.markdown(
            "What bad result the change causes, such as:\n"
            "- **CAG Repeat Expansion**: the CAG word in DNA grows too long.\n"
            "- **Neurodegeneration**: brain cells slowly die.\n"
            "- **Faster Disease Onset**: symptoms start earlier in life."
        )

    with st.expander("Category"):
        st.markdown(
            "- **Primary Cause**: the main gene that causes the disease.\n"
            "- **Trans‑acting Modifier**: a gene far away that changes how severe the disease is.\n"
            "- **Cis‑acting Modifier**: a nearby gene that also changes how severe the disease is."
        )

# Stage mapping for user-friendly display (LabelEncoder alphabetical order)
stage_names = {
    0: "Early",              
    1: "Late",               
    2: "Middle",             
    3: "Pre-Symptomatic"     
}

# ✅ FIXED: Single button with unique key
if st.button("Predict Disease Stage", type="primary", key="unique_predict_btn"):
    
    # Create input DataFrame
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

    # ✅ Encode categorical variables to match training data
    categorical_mappings = {
        "Sex": {"Female": 0, "Male": 1},
        "Family_History": {"No": 0, "Yes": 1},
        "Motor_Symptoms": {"Mild": 0, "Moderate": 1, "Severe": 2},
        "Cognitive_Decline": {"Mild": 0, "Moderate": 1, "Severe": 2},
        "Gene_Mutation_Type": {"Deletion": 0, "Duplication": 1, "Insertion": 2, "Point Mutation": 3},
        "Gene/Factor": {"HTT": 0, "HTT (Somatic Expansion)": 1, "MLH1": 2, "MSH3": 3},
        "Chromosome_Location": {"3p22.2": 0, "4p16.3": 1, "5q14.1": 2},
        "Function": {"Mismatch Repair": 0},
        "Effect": {"CAG Repeat Expansion": 0, "Faster Disease Onset": 1, "Neurodegeneration": 2},
        "Category": {"Cis-acting Modifier": 0, "Primary Cause": 1, "Trans-acting Modifier": 2}
    }
    
    # Apply encodings
    for col, mapping in categorical_mappings.items():
        if col in input_data.columns:
            input_data[col] = input_data[col].map(mapping)
    
    # Make prediction with encoded data
    prediction_num = model.predict(input_data)[0]
    
    # ✅ Convert numeric prediction to readable string
    stage_display = stage_names.get(prediction_num, f"Stage {prediction_num}")
    
    # Results display
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.success(f"**{stage_display}**")
        st.caption(f"Model Output: {prediction_num}")
    
    with col2:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            confidence = float(np.max(proba) * 100)
            
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Probability table with correct stage names
            prob_df = pd.DataFrame({
                "Disease Stage": [stage_names.get(int(i), f"Stage {i}") for i in range(4)],
                "Probability": [f"{p:.1%}" for p in proba]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
    
    # Clinical interpretation based on numeric prediction
    if prediction_num == 3:
        st.info("✅ Low risk - Early intervention recommended")
    elif prediction_num == 2:
        st.warning("⚠️ Moderate risk - Regular monitoring advised")
    elif prediction_num in [0, 1]:
        st.error("🚨 High risk - Immediate clinical attention needed")

st.markdown("---")
st.caption("🔬 Built with Streamlit | ML Deployment | By Hemanth")
