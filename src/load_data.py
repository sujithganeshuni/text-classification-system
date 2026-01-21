import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "raw" / "spam"

# Load tab-separated text file
df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"])

print(df.head())
print(df["label"].value_counts())
