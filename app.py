# appp.py — Multi-label Toxicity Detection with Streamlit

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import pad_sequences

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load trained multi-label LSTM model
model = load_model("multilabel_lstm_model.keras")

# Define label names
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediction function for a single comment
def predict_labels(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=150, padding='post')
    pred = model.predict(padded)[0]
    return dict(zip(LABELS, pred))

# Streamlit UI
st.set_page_config(page_title="🧠 Multi-Label Toxic Comment Detector", layout="centered")
st.title("💬 Multi-Label Toxic Comment Classifier")
st.write("Enter a comment to detect multiple types of toxicity or upload a CSV file.")

# Real-time input
st.subheader("✏️ Predict from a Single Comment")
user_input = st.text_area("Type a comment here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        results = predict_labels(user_input)
        st.markdown("### 🔍 Prediction Results:")
        for label, score in results.items():
            emoji = "🛑" if score >= 0.5 else "✅"
            st.write(f"{emoji} **{label}**: `{score:.2f}`")

# CSV Upload Section
st.markdown("---")
st.subheader("📁 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV with a column named `comment_text`", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if 'comment_text' not in df.columns:
            st.error("❌ CSV must contain a `comment_text` column.")
        else:
            # Clean and preprocess
            df['cleaned'] = df['comment_text'].apply(clean_text)
            seqs = tokenizer.texts_to_sequences(df['cleaned'])
            padded = pad_sequences(seqs, maxlen=150, padding='post')
            preds = model.predict(padded)

            # Add each label prediction
            for i, label in enumerate(LABELS):
                df[label] = preds[:, i]

            # Optional: Thresholded binary predictions
            for label in LABELS:
                df[label + '_label'] = df[label].apply(lambda x: "Toxic" if x >= 0.5 else "Non-Toxic")

            # Display
            st.success("✅ Batch predictions complete!")
            st.dataframe(df[['comment_text'] + LABELS + [l + "_label" for l in LABELS]])

            ### Visualize toxicity label distribution
            st.subheader("📊 Toxicity Label Distribution")
            st.divider()

            # Count number of toxic predictions (threshold ≥ 0.5) per label
            label_counts = {
                label: (df[label] >= 0.5).sum()
                for label in LABELS
            }

            # Filter out labels with zero counts for a cleaner chart
            filtered_counts = {k: v for k, v in label_counts.items() if v > 0}

            # Handle empty case
            if not filtered_counts:
                st.info("No toxic labels detected above the threshold in this file.")
            else:
                fig, ax = plt.subplots()
                ax.pie(
                    filtered_counts.values(),
                    labels=filtered_counts.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=plt.cm.tab20.colors[:len(filtered_counts)]
                )
                ax.axis('equal')  # Perfect circle
                st.pyplot(fig)

            

            # Download button
            result_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", result_csv, "multilabel_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
