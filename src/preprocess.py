import string
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Initialize
stop_words = set(ENGLISH_STOP_WORDS)
# Preserve negation words because they are important for sentiment polarity
negation_words = {'no', 'not', 'nor', 'never'}
stop_words = stop_words.difference(negation_words)

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)          # remove HTML
    text = re.sub(r'http\S+|www\S+', '', text) # remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\d+', '', text)            # remove numbers
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Convert rating to sentiment
def get_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

if __name__ == "__main__":
    import os
    import pandas as pd

    # -------- MAIN PROCESS -------- #

    # Load dataset (IMPORTANT: correct path)
    df = pd.read_csv(r'data/raw/7817_1.csv')

    # Select needed columns
    df = df[['reviews.text', 'reviews.rating']]
    df.columns = ['text', 'rating']

    # Remove nulls and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Convert sentiment
    df['sentiment'] = df['rating'].apply(get_sentiment)

    # Apply preprocessing
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Keep only required columns
    df = df[['text', 'clean_text', 'sentiment']]

    # Create processed folder if not exists
    os.makedirs('../data/processed', exist_ok=True)

    # Save file
    output_path = '../data/processed/clean_dataset.csv'
    df.to_csv(output_path, index=False)

    print("Preprocessing completed!")
    print("File saved at:", output_path)
    print("Sample data:")
    print(df.head())
