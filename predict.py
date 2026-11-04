import os
import joblib
from preprocess import preprocess_email

# Setup paths
MODEL_PATH = os.path.join('models', 'model.pkl')
VECTORIZER_PATH = os.path.join('models', 'vectorizer.pkl')

def load_artifacts():
    """
    Loads and returns the model and vectorizer.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model or Vectorizer file missing. Please execute train.py first.")
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict_email(text):
    """
    Processes raw input email and produces classification output.
    """
    try:
        model, vectorizer = load_artifacts()
    except FileNotFoundError as e:
        return "Error", 0.0, str(e)
        
    cleaned = preprocess_email(text)
    if not cleaned:
        return "Legitimate", 0.0, "Empty or invalid string provided."
        
    vectorized = vectorizer.transform([cleaned])
    
    # Get predictions and probabilities
    pred = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0]
    
    # Format output classification label
    label = "Phishing" if pred == 1 else "Legitimate"
    confidence = prob[1] if pred == 1 else prob[0]
    
    return label, confidence, "Success"