import matplotlib
matplotlib.use("Agg")

import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "raw" / "spam"
model_path = BASE_DIR / "models" / "spam_classifier.pkl"

# Load dataset
df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"])
df["text"] = df["text"].str.lower()

# Load trained model and vectorizer
model, vectorizer = joblib.load(model_path)

# Convert text to features
X = vectorizer.transform(df["text"])
y = df["label"]

# Train-test split (same logic as training)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Predictions
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["spam", "ham"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["spam", "ham"])

# Plot
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Spam Classification (Naive Bayes)")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
print("Confusion matrix saved as confusion_matrix.png")

