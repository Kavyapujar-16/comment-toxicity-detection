
# 💬 Comment Toxicity Detection with Deep Learning and Streamlit

This project is a real-time and batch toxicity classification system using Deep Learning (LSTM) and deployed via Streamlit. It analyzes text comments and predicts whether the input is toxic or non-toxic.

## 🚀 Features

- Real-time comment classification
- Batch prediction via CSV upload
- Toxicity score for each comment
- Pie chart visualization for batch results
- Downloadable result CSV
- Clean UI powered by Streamlit

## 🧠 Model Info

- LSTM (Keras Functional API)
- Trained on multi-label comment dataset (`toxic` label used)
- Tokenizer saved using `pickle`
- Model saved in `.keras` format

## 📁 Folder Structure

```
├── app.py
├── tokenizer.pkl
├── lstm_model_final.keras
├── requirements.txt
├── README.md
```

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

```bash
streamlit run app.py
```

## 📄 Sample CSV Format

Your uploaded CSV file should have a column named:

```csv
comment_text
This is an amazing post!
You're the worst person ever.
Thanks a lot for this.
```

## 📊 Demo Preview

- Enter a comment to see real-time prediction
- Upload a `.csv` file with comments
- View prediction table and pie chart
- Download results as CSV

## 📹 Demo Video

> A 1–2 minute video demonstrating:
> - Single comment prediction
> - Batch upload and results
> - Pie chart and download button



