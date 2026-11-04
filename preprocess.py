import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Automatically download required NLTK resources including punkt_tab for tokenization compatibility
for resource in ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

def preprocess_email(text):
    """
    Cleans and preprocesses input text for classification.
    """
    # Handle missing or non-string inputs safely
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Load stopwords and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Filter tokens and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words
    ]
    
    # Return the cleaned text as a single space-separated string
    return " ".join(cleaned_tokens)