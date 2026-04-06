import pandas as pd
import json
from collections import Counter

print("Starting data loading...\n")

# -------------------------------
# Dataset 1: Davidson et al. 2017
# -------------------------------
print("[Dataset 1] Davidson Hate Speech & Offensive Language")
davidson = pd.read_csv("data/raw/labeled_data.csv")
print(f"Shape: {davidson.shape}")
print(f"Columns: {davidson.columns.tolist()}")
print("Missing values:")
print(davidson.isnull().sum())

if "class" in davidson.columns:
    print("\nClass distribution:")
    print(davidson["class"].value_counts())

if {"tweet", "class"}.issubset(davidson.columns):
    print("\nFirst 3 rows:")
    print(davidson[["tweet", "class"]].head(3))

print("\n" + "-" * 60 + "\n")

# -------------------------------
# Dataset 2: GESIS Sexism Dataset
# -------------------------------
print("[Dataset 2] GESIS Sexism Dataset")
gesis = pd.read_csv("data/raw/sexism_data.csv")
print(f"Shape: {gesis.shape}")
print(f"Columns: {gesis.columns.tolist()}")
print("Missing values:")
print(gesis.isnull().sum())

if "sexist" in gesis.columns:
    print("\nClass distribution:")
    print(gesis["sexist"].value_counts())

if {"text", "sexist"}.issubset(gesis.columns):
    print("\nFirst 3 rows:")
    print(gesis[["text", "sexist"]].head(3))

print("\n" + "-" * 60 + "\n")

# -------------------------------
# Dataset 3: MMHS150K
# -------------------------------
print("[Dataset 3] MMHS150K")
with open("data/raw/MMHS150K_GT.json", "r", encoding="utf-8") as f:
    data = json.load(f)

mmhs = pd.DataFrame.from_dict(data, orient="index")
print(f"Shape: {mmhs.shape}")
print(f"Columns: {mmhs.columns.tolist()}")
print("Missing values:")
print(mmhs.isnull().sum())

label_map = {
    0: "Not Hate",
    1: "Racist",
    2: "Sexist",
    3: "Homophobe",
    4: "Religion",
    5: "Other Hate",
}

def majority_vote(labels):
    return Counter(labels).most_common(1)[0][0]

if "labels" in mmhs.columns:
    mmhs["label"] = mmhs["labels"].apply(majority_vote).map(label_map)
    print("\nLabel distribution (majority vote):")
    print(mmhs["label"].value_counts())

if {"tweet_text", "label"}.issubset(mmhs.columns):
    print("\nFirst 3 rows:")
    print(mmhs[["tweet_text", "label"]].head(3))

print("\nAll 3 datasets loaded successfully.")
