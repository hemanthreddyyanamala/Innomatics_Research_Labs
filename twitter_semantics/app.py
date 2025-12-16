import streamlit as st
import pickle

st.title("Toxicity Detection App 🚀")

# Load saved files
with open(r"C:\Users\heman\OneDrive\Desktop\AFTER BTECH\innomatics\ml\twitter_deployement\Tfidf_vect.pkl", "rb") as f:
    tfv = pickle.load(f)

with open(r"C:\Users\heman\OneDrive\Desktop\AFTER BTECH\innomatics\ml\twitter_deployement\model.pkl", "rb") as f:
    model = pickle.load(f)

text = st.text_input("Enter a review")

if st.button("Predict"):
    if text:
        text_vec = tfv.transform([text])
        prediction = model.predict(text_vec)[0]

        # Output as 'Toxic' or 'Not Toxic'
        output = "Toxic" if prediction == 1 else "Not Toxic"
        st.success(f"Prediction: {output}")
    else:
        st.warning("Please enter some text")