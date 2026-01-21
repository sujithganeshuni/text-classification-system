import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / 'data'/'raw'/'spam'

df=pd.read_csv(data_path,sep="\t",header=None,names=["label","text"])

df["text"]=df["text"].str.lower()

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["text"])

print(X.shape)
