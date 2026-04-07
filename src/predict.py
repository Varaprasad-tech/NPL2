import pickle
import os
from src.preprocess import preprocess_text

# Path setup
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(base_dir, 'models', 'sentiment_model.pkl')
vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer.pkl')

# Load model
model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])
    return model.predict(vectorized)[0]