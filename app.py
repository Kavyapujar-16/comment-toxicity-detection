import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import pad_sequences

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the functional model
model = load_model("lstm_model_final.keras")

# Text clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Predict single comment
def predict_comment(comment):
    cleaned = clean_text(comment)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=150, padding='post')
    prediction = model.predict(padded)[0][0]
    return prediction

# UI Section
st.title("💬 Toxic Comment Classifier")
st.write("Predict toxicity of a single comment or upload a CSV for batch processing.")

# Real-time prediction
comment = st.text_area("Enter a comment:")

if st.button("Predict"):
    if comment.strip() == "":
        st.warning("Please enter a comment.")
    else:
        prob = predict_comment(comment)
        label = "🛑 Toxic" if prob >= 0.5 else "✅ Non-Toxic"
        st.markdown(f"### Prediction: {label}")
        st.write(f"Confidence: **{prob:.2f}**")

# CSV Upload Section
st.markdown("---")
st.subheader("📁 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV file with a column named `comment_text`", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if 'comment_text' not in df.columns:
            st.error("❌ Column `comment_text` not found in uploaded file.")
        else:
            # Batch processing
            df['cleaned'] = df['comment_text'].apply(clean_text)
            sequences = tokenizer.texts_to_sequences(df['cleaned'])
            padded = pad_sequences(sequences, maxlen=150, padding='post')
            preds = model.predict(padded)
            df['toxicity_score'] = preds
            df['prediction'] = df['toxicity_score'].apply(lambda x: "Toxic" if x >= 0.5 else "Non-Toxic")

            # Show result
            st.success("✅ Predictions completed!")
            st.dataframe(df[['comment_text', 'toxicity_score', 'prediction']])

            # Pie Chart
            st.subheader("🧩 Toxicity Distribution (Pie Chart)")
            toxic_count = df['prediction'].value_counts().get('Toxic', 0)
            nontoxic_count = df['prediction'].value_counts().get('Non-Toxic', 0)

            labels = ['Toxic', 'Non-Toxic']
            sizes = [toxic_count, nontoxic_count]
            colors = ['#ff4d4d', '#66bb6a']

            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            # Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Result CSV", csv, "toxicity_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")
