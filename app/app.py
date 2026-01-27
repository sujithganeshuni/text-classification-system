import streamlit as st
import joblib
from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "spam_classifier.pkl"

# Load model and vectorizer
model, vectorizer = joblib.load(model_path)

# App title
st.title("SMS Spam Classifier")
st.write("Enter an SMS message and the model will classify it as Spam or Ham.")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_area("Enter message:")

# Input validation + prediction
if st.button("Classify"):
    text = user_input.strip()

    # Validation
    if text == "":
        st.warning("Please enter a message.")
    elif len(text.split()) < 3:
        st.warning("Message is too short to classify reliably.")
    else:
        transformed_text = vectorizer.transform([text.lower()])
        prediction = model.predict(transformed_text)[0]

        if prediction == "spam":
            result = "SPAM"
            st.error("🚨 This message is SPAM")
        else:
            result = "HAM"
            st.success("✅ This message is HAM (Not Spam)")

        # Save to history
        st.session_state.history.append({
            "Message": text,
            "Prediction": result
        })

# Show prediction history
if st.session_state.history:
    st.subheader("Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)