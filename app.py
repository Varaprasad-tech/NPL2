import streamlit as st
import pickle
import os
from src.preprocess import preprocess_text

# -------- PATH SETUP -------- #

base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, 'models', 'sentiment_model.pkl')
vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer.pkl')

# -------- LOAD MODEL -------- #

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# -------- UI DESIGN -------- #

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

st.title("💬 Sentiment Analysis App")
st.write("Analyze whether a review is Positive, Negative, or Neutral")

# Input box
user_input = st.text_area("✍️ Enter your review:")

# Button
if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Preprocess
        cleaned = preprocess_text(user_input)

        # Transform
        vectorized = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vectorized)[0]

        # Display result
        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        elif prediction == "negative":
            st.error("😡 Negative Sentiment")
        else:
            st.info("😐 Neutral Sentiment")

# Footer
st.write("---")
st.write("🚀 Built using NLP, TF-IDF & Logistic Regression")