# 🛡️ Phishing Email Detection System

A robust, machine-learning-powered NLP application built with Python to analyze, preprocess, and classify emails into **Phishing** or **Legitimate** categories.

---

## 📁 Project Structure

```text
├── data/
│   └── phishing_email.csv  # The target dataset (600 rows of user/fallback data)
├── models/
│   ├── model.pkl           # Trained Logistic Regression classifier
│   └── vectorizer.pkl      # Serialized TF-IDF feature extractor
├── preprocess.py           # NLP pipelines (NLTK Tokenization, Stopwords, Lemmatizer)
├── train.py                # Model training, validation split, and evaluation script
├── predict.py              # Low-latency single-text inference wrapper
├── ui.py                   # Streamlit Dashboard interactive frontend
└── requirements.txt        # Production-ready Python dependencies
```

---

## 🚀 Installation & Local Execution (VS Code / CLI)

To run this application locally on your computer inside VS Code, follow these steps:

### 1. Clone or Copy the Files
Ensure all five Python/Text files are located inside your chosen local project folder.

### 2. Set Up a Virtual Environment
Open your terminal in VS Code and run the following commands to create and activate an isolated environment:
```bash
# Create Virtual Environment
python -m venv venv

# Activate Environment
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Dependencies
Install the pre-configured packaging list using pip:
```bash
pip install -r requirements.txt
```
*(This will automatically configure Streamlit, Scikit-Learn, NLTK, Pandas, Numpy, and Seaborn)*

### 4. Run the Training Pipeline
Generate your model and feature vectorizer binary files (`models/model.pkl`, `models/vectorizer.pkl`) by training on the baseline dataset:
```bash
python train.py
```

### 5. Launch the Streamlit Interactive Dashboard
Fire up the local interactive web interface:
```bash
streamlit run ui.py
```
Your browser should open automatically to `http://localhost:8501` showcasing the single-text checking system, batch processing capability, and diagnostic graphs.

---

## 📊 Features & Functional Highlights
* **Single-Text Checker**: Paste raw email strings to evaluate secure versus suspicious risk levels with accurate confidence probability.
* **Batch Processing**: Upload any spreadsheet dataset (CSV format), auto-select target feature columns, run classifications in bulk, and download prediction output records.
* **Modern NLP Preprocessing**: Text processing with lemmatization via NLTK to reduce words to their base dictionary forms, and clean regex symbol-stripping.