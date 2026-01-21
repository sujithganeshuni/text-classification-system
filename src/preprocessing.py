import re
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data"/"raw"/"spam"

df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"])

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)
print(df[["text","clean_text"]].head())