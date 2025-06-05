# app.py — Final Clean Version

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import base64
from keras.models import load_model
from keras.utils import pad_sequences

# Page config
st.set_page_config(page_title="🧠 Multi-Label Toxic Comment Detector", layout="wide")

# Background styling
def set_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    page_bg_img = f"""
    <style>
    body {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }}

    .stApp {{
        background-color: rgba(0, 0, 0, 0.0);
        display: flex;
        justify-content: flex-start;
        padding-left: 5vw;
    }}

    h1, h2, h3, h4, h5, h6, p, label {{
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
    }}

    .stTextInput > div > div > input,
    textarea {{
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: black !important;
        border-radius: 10px;
    }}

    .stButton button {{
        background-color: #ffffffcc;
        color: black;
        border-radius: 10px;
        font-weight: bold;
    }}

    .stDataFrame, .stDownloadButton {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: black !important;
        border-radius: 10px;
    }}

    .pie-container {{
        display: flex;
        justify-content: flex-start;
        margin-top: 1rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Apply the background
set_bg_from_local("C:/Users/kavya/project4/background.png")

# Load tokenizer and model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("multilabel_lstm_model.keras")
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Predict a single comment
def predict_labels(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=150, padding='post')
    pred = model.predict(padded)[0]
    return dict(zip(LABELS, pred))

# Interface
st.title("💬 Multi-Label Toxic Comment Classifier")
st.write("Enter a comment to detect multiple types of toxicity or upload a CSV file.")

# Single comment input
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

# Batch file input
st.markdown("---")
st.subheader("📁 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV with a column named `comment_text`", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if 'comment_text' not in df.columns:
            st.error("❌ CSV must contain a `comment_text` column.")
        else:
            df['cleaned'] = df['comment_text'].apply(clean_text)
            seqs = tokenizer.texts_to_sequences(df['cleaned'])
            padded = pad_sequences(seqs, maxlen=150, padding='post')
            preds = model.predict(padded)

            for i, label in enumerate(LABELS):
                df[label] = preds[:, i]

            for label in LABELS:
                df[label + '_label'] = df[label].apply(lambda x: "Toxic" if x >= 0.5 else "Non-Toxic")

            st.success("✅ Batch predictions complete!")
            st.dataframe(df[['comment_text'] + LABELS + [l + "_label" for l in LABELS]])

            import io
            import matplotlib
            matplotlib.use('Agg')  # Prevent backend errors

            # 📊 Left-aligned, clean, small and sharp pie chart
            st.markdown("### 📊 Toxicity Label Distribution")

            label_counts = {
                label: (df[label] >= 0.5).sum()
                for label in LABELS
            }
            filtered_counts = {k: v for k, v in label_counts.items() if v > 0}

            if not filtered_counts:
                st.info("No toxic labels detected above the threshold in this file.")
            else:
                # Step 1: Create pie chart
                fig, ax = plt.subplots(figsize=(5,5), dpi=300, facecolor='none')
                ax.pie(
                    filtered_counts.values(),
                    labels=filtered_counts.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=plt.cm.Set2.colors[:len(filtered_counts)],
                    textprops={'fontsize': 10, 'color': 'white'}  # Keep text visible
                )
                ax.axis('equal')
                fig.patch.set_alpha(0.0)

                # Step 2: Save chart to memory
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", bbox_inches='tight', transparent=True)
                buffer.seek(0)
                encoded = base64.b64encode(buffer.read()).decode()

                # Step 3: Show as inline small image, aligned to left
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin-top: 0px; margin-bottom: -30px;">
                        <img src="data:image/png;base64,{encoded}" width="450"/>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Add spacing after pie chart
            st.markdown('<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)

            # Prepare downloadable CSV
            result_csv = df.to_csv(index=False).encode('utf-8')

            # Custom download button style
            custom_css = """
            <style>
                .custom-download-button > button {
                    background-color: #007acc;
                    color: black;
                    font-weight: bold;
                    border-radius: 8px;
                    border: none;
                    padding: 10px 16px;
                }
            </style>
            """
            st.markdown(custom_css, unsafe_allow_html=True)

            # Render the button
            with st.container():
                st.markdown('<div class="custom-download-button">', unsafe_allow_html=True)
                st.download_button("📥 Download Results", result_csv, "multilabel_predictions.csv", "text/csv")
                st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
