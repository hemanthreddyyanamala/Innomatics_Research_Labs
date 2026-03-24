import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Movie Genre Classification", page_icon="", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1f2933 0%, #020617 55%, #000000 100%);
    color: #f9fafb;
}
.metric-card {
    background: rgba(79, 70, 229, 0.1);
    padding: 1.5rem;
    border-radius: 1rem;
    border-left: 4px solid #4f46e5;
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
    try:
        with open('Movie_genre.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessing.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Loading error: {str(e)}")
        st.stop()

def main():
    st.title("Movie Genre Classifier")
    
    model, vectorizer = load_model()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Enter Movie Description")
        text_input = st.text_area("", height=150, placeholder="Paste movie plot...")

        if st.button("Predict", type="primary", use_container_width=True):
            if not text_input.strip():
                st.warning("Enter text!")
            else:
                try:
                    X = vectorizer.transform([text_input])

                    # Predict
                    pred = model.predict(X)[0]
                    proba = model.predict_proba(X)[0] * 100  # shape: (n_classes,)

                    # ===== SHOW TOP CONFIDENCE HERE =====
                    top_conf = proba.max()
                    st.success(f"Confidence: {top_conf:.2f}%")

                    # Main card
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style='color: #10b981;'>Genre: <strong>{pred}</strong></h2>
                    </div>
                    """, unsafe_allow_html=True)

                    # Full table
                    prob_df = pd.DataFrame({
                        'Genre': model.classes_,
                        'Score': [f"{p:.2f}%" for p in proba]
                    }).sort_values('Score', ascending=False)
                    st.dataframe(prob_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")


    with col2:
        st.header("About The Model")
        
        with st.expander("Model Type"):
            st.markdown("""
            <div class="info-box">
                <p><strong>Multinomial Naive Bayes</strong></p>
                <p>
                • Good for text classification<br>
                • Uses word frequencies (TF‑IDF)<br>
                • Fast and works well with many genres
                </p>
            </div>
            """, unsafe_allow_html=True)

        
        with st.expander("Vectorizer"):
            st.markdown("""
            <div class="info-box">
                <h4>Vectorizer</h4>
                <p><strong>TF-IDF</strong> (Term Frequency–Inverse Document Frequency)</p>
                <small>N-grams: (1, 1)</small>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Supported Genres"):
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            genres = list(model.classes_)
            for i in range(0, len(genres), 5):
                cols = st.columns(5)
                cols[0].write(f"• {genres[i]}")
                if i + 1 < len(genres):
                    cols[1].write(f"• {genres[i+1]}")
                if i + 2 < len(genres):
                    cols[2].write(f"• {genres[i+2]}")
                if i + 3 < len(genres):
                    cols[3].write(f"• {genres[i+3]}")
                if i + 4 < len(genres):
                    cols[4].write(f"• {genres[i+4]}")
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("How It Works"):
            st.markdown("""
            <div class="info-box">
                <h4>How It Works</h4>
                <p>
                1️⃣ Text is converted to TF-IDF vectors<br>
                2️⃣ Model predicts genre probability<br>
                3️⃣ Returns top matching genre + confidence
                </p>
            </div>
            """, unsafe_allow_html=True)


    st.markdown("---")
    st.caption("Built with Streamlit | ML Deployment | Hemanth")

if __name__ == "__main__":
    main()
