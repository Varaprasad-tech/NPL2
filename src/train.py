import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -------- PATH SETUP -------- #

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(base_dir, 'data', 'processed', 'clean_dataset.csv')
model_dir = os.path.join(base_dir, 'models')

os.makedirs(model_dir, exist_ok=True)

print("Loading dataset from:", data_path)

# -------- LOAD DATA -------- #

df = pd.read_csv(data_path)

# -------- CLEANING FIXES -------- #

df.dropna(subset=['clean_text'], inplace=True)
df['clean_text'] = df['clean_text'].astype(str)
df = df[df['clean_text'].str.strip() != ""]

print("Dataset size after cleaning:", df.shape)

# -------- ⚖️ KEEP FULL DATASET -------- #

print("\nKeeping all available cleaned examples to improve reliability.")
print(df['sentiment'].value_counts())

# -------- FEATURES & LABELS -------- #

X = df['clean_text']
y = df['sentiment']

# -------- TF-IDF -------- #

print("\nApplying TF-IDF...")

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

# -------- TRAIN TEST SPLIT -------- #

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# -------- MODEL TRAINING -------- #

print("Training model...")

model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train, y_train)

# -------- EVALUATION -------- #

print("\nModel Performance:\n")

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# -------- SAVE MODEL -------- #

model_path = os.path.join(model_dir, 'sentiment_model.pkl')
vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')

pickle.dump(model, open(model_path, 'wb'))
pickle.dump(vectorizer, open(vectorizer_path, 'wb'))

print("\nModel saved successfully!")
print("Model path:", model_path)