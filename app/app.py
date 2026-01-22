import streamlit as st
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "spam_classifier.pkl"

# Load model and vectorizer
model, vectorizer = joblib.load(model_path)

# UI
st.title("SMS Spam Classifier")
st.write("Enter an SMS message and the model will classify it as Spam or Ham.")

user_input = st.text_area("Enter message:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_text = vectorizer.transform([user_input.lower()])
        prediction = model.predict(transformed_text)[0]

        if prediction == "spam":
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is HAM (Not Spam)")
