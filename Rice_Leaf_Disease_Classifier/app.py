import streamlit as st
import pickle
import numpy as np
import cv2
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1f2933 0%, #020617 55%, #000000 100%);
    color: #f9fafb;
}
.metric-card {
    padding: 1.5rem;
    border-radius: 1rem;
}
.info-box {
    padding: 1.2rem;
    border-radius: 0.8rem;
    margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('Rice_Leaf_Disease_Classifier/rice_leaf_diseases.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

def extract_features(image):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img = cv2.resize(src=img_bgr, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)
    return img.flatten().reshape(1, -1)

def main():
    st.title("Rice Leaf Disease Classifier")
    
    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        st.subheader("Upload Rice Leaf Image")
        uploaded_file = st.file_uploader("", type=['png','jpg','jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            
            # AUTO PREDICT - No button needed
            with st.spinner('Processing (300x300 flatten)...'):
                features = extract_features(image)
                
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0]
                
                # Main prediction card
                top_conf = np.max(probability) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style='color: #10b981;'>Disease: <strong>{prediction}</strong></h2>
                    <p>Confidence: <strong>{top_conf:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability table
                prob_df = pd.DataFrame({
                    'Disease': ['Bacterialblight', 'Brownspot', 'Leafsmut'],
                    'Probability': [f"{p:.1%}" for p in probability]
                })
                st.dataframe(prob_df, use_container_width=True)

    with col2:
        st.header("Model Information")
        
        # Model Used
        with st.expander("Model Used"):
            st.markdown("""
            <div class="info-box">
                <h4>DecisionTreeClassifier(random_state=42, class_weight='balanced')</h4>
                <p><strong>Parameters:</strong></p>
                • max_depth: None (unlimited)<br>
                • class_weight: 'balanced'<br>
                • random_state: 42<br>
                • Features: 270,000 (300×300×3)
            </div>
            """, unsafe_allow_html=True)
        
        # Preprocessing Pipeline
        with st.expander("Preprocessing Pipeline"):
            st.markdown("""
            <div class="info-box">
                <h4>Image → Features Pipeline</h4>
                <ol>
                    <li><strong>Load Image</strong> (PNG/JPG)</li>
                    <li><strong>RGB → BGR</strong> (OpenCV format)</li>
                    <li><strong>Resize</strong> 300×300 (INTER_LINEAR)</li>
                    <li><strong>Flatten</strong> → 270,000 features</li>
                    <li><strong>Predict</strong> with Decision Tree</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        # Dataset
        with st.expander("Dataset"):
            st.markdown("""
            <div class="info-box">
                <h4>Rice Leaf Diseases Dataset</h4>
                <p><strong>Total: 4,684 images</strong></p>
                """, unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a: st.metric("Brownspot", "1,620")
            with col_b: st.metric("Bacterialblight", "1,604")
            with col_c: st.metric("Leafsmut", "1,460")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # About the Diseases + Sample Images
        with st.expander("About the Diseases"):
            st.markdown("""
            <div class="info-box">
                <h4>Disease Descriptions & Samples</h4>
            """, unsafe_allow_html=True)
            
            # Brownspot
            st.markdown("**Brownspot** - Small, circular brown spots on leaves")
            st.image("Rice_Leaf_Disease_Classifier/brownspot_orig_098.jpg", width=200)
            
            # Bacterialblight  
            st.markdown("**Bacterial Blight** - Yellowish streaks turning grayish white")
            st.image("Rice_Leaf_Disease_Classifier/BACTERAILBLIGHT3_191.jpg", width=200)
            
            # Leafsmut
            st.markdown("**Leaf Smut** - Black powdery masses on leaf surfaces")
            st.image("Rice_Leaf_Disease_Classifier/BLAST1_075.jpg", width=200)
            
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with Streamlit | Rice Disease Detection | Hemanth")

if __name__ == "__main__":
    main()



