import pandas as pd
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "raw" / "spam"
model_path = BASE_DIR / "models" / "spam_classifier.pkl"

# Load data
df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"])
df["text"] = df["text"].str.lower()

# Features
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Split
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train best model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model + vectorizer
joblib.dump((model, vectorizer), model_path)

print("Model saved at:", model_path)
