import os
import re
import json
import pandas as pd
from collections import Counter

os.makedirs("data/processed", exist_ok=True)

print("Starting preprocessing...\n")

# -------------------------------
# TEXT CLEANING FUNCTION
# -------------------------------
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)                # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)           # keep hashtag word, remove #
    text = re.sub(r"[^\w\s]", "", text)             # remove punctuation
    text = re.sub(r"\d+", "", text)                 # remove numbers
    text = re.sub(r"\s+", " ", text).strip()        # remove extra spaces
    return text

def majority_vote(labels):
    return Counter(labels).most_common(1)[0][0]

# -------------------------------
# STEP 1: DAVIDSON DATASET
# -------------------------------
print("[Step 1] Processing Davidson dataset")
davidson = pd.read_csv("data/raw/labeled_data.csv")
davidson = davidson[["tweet", "class"]].copy()

davidson_map = {
    0: "Other Hate",
    1: "Offensive Language",
    2: "Not Hate",
}
davidson["label"] = davidson["class"].map(davidson_map)
davidson = davidson[["tweet", "label"]].copy()
davidson.columns = ["tweet_text", "label"]

print(f"Loaded {davidson.shape[0]} rows")
print(davidson["label"].value_counts(), "\n")

# -------------------------------
# STEP 2: GESIS DATASET
# -------------------------------
print("[Step 2] Processing GESIS dataset")
gesis = pd.read_csv("data/raw/sexism_data.csv")
gesis = gesis[["text", "sexist"]].copy()
gesis.columns = ["tweet_text", "sexist"]
gesis["label"] = gesis["sexist"].map({True: "Sexist", False: "Not Hate"})
gesis = gesis[["tweet_text", "label"]].copy()

print(f"Loaded {gesis.shape[0]} rows")
print(gesis["label"].value_counts(), "\n")

# -------------------------------
# STEP 3: MMHS150K DATASET
# -------------------------------
print("[Step 3] Processing MMHS150K dataset")
with open("data/raw/MMHS150K_GT.json", "r", encoding="utf-8") as f:
    data = json.load(f)

mmhs = pd.DataFrame.from_dict(data, orient="index")

label_map = {
    0: "Not Hate",
    1: "Racist",
    2: "Sexist",
    3: "Homophobe",
    4: "Religion",
    5: "Other Hate",
}

mmhs["final_label"] = mmhs["labels"].apply(majority_vote)
mmhs["label"] = mmhs["final_label"].map(label_map)

# Exclude Homophobe due to context-dependent annotation
mmhs = mmhs[mmhs["label"] != "Homophobe"]
mmhs = mmhs[["tweet_text", "label"]].copy()

print(f"Loaded {mmhs.shape[0]} rows")
print(mmhs["label"].value_counts(), "\n")

# -------------------------------
# STEP 4: COMBINE DATASETS
# -------------------------------
print("[Step 4] Combining datasets")
combined = pd.concat([davidson, gesis, mmhs], ignore_index=True)
print(f"Combined rows before cleaning: {combined.shape[0]}\n")

# -------------------------------
# STEP 5: CLEAN TEXT
# -------------------------------
print("[Step 5] Cleaning tweet text")
combined["tweet_text"] = combined["tweet_text"].apply(clean_tweet)

before = len(combined)

combined = combined.dropna(subset=["tweet_text", "label"])
combined = combined.drop_duplicates(subset=["tweet_text"])
combined = combined[combined["tweet_text"].str.len() > 0]

removed = before - len(combined)
print(f"Removed {removed} rows (missing / duplicate / empty)")
print(f"Rows after cleaning: {combined.shape[0]}\n")

# -------------------------------
# STEP 6: CAP NOT HATE CLASS
# -------------------------------
print("[Step 6] Reducing class imbalance")

not_hate_rows = combined[combined["label"] == "Not Hate"]
rest_rows = combined[combined["label"] != "Not Hate"]

if len(not_hate_rows) > 15000:
    not_hate_rows = not_hate_rows.sample(n=15000, random_state=42)

combined = pd.concat([not_hate_rows, rest_rows], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

print("Class distribution after balancing:")
print(combined["label"].value_counts(), "\n")

# -------------------------------
# STEP 7: ADD FEATURES
# -------------------------------
print("[Step 7] Adding structural features")
combined["tweet_length"] = combined["tweet_text"].apply(len)
combined["word_count"] = combined["tweet_text"].apply(lambda x: len(x.split()))

print("Tweet length summary:")
print(combined["tweet_length"].describe(), "\n")

# -------------------------------
# STEP 8: SAVE OUTPUT
# -------------------------------
output_path = "data/processed/combined_dataset.csv"
combined.to_csv(output_path, index=False)

print(f"Final dataset shape: {combined.shape}")
print(f"Saved cleaned dataset to: {output_path}")
