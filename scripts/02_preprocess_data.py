import os
import re
import json
import pandas as pd
from collections import Counter

os.makedirs("data/processed", exist_ok=True)

print("Starting preprocessing...\n")

# -------------------------------
# Helper functions
# -------------------------------
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)         # remove URLs
    text = re.sub(r"@\w+", "", text)                   # remove mentions
    text = re.sub(r"[^\w\s]", "", text)                # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()           # remove extra spaces
    return text

def majority_vote(labels):
    return Counter(labels).most_common(1)[0][0]

# -------------------------------
# Step 1: Davidson
# -------------------------------
print("[Step 1] Processing Davidson dataset")
davidson = pd.read_csv("data/raw/labeled_data.csv")
davidson = davidson[["tweet", "class"]].copy()
davidson["label"] = davidson["class"].map({
    0: "Other Hate",
    1: "Offensive Language",
    2: "Not Hate"
})
davidson = davidson[["tweet", "label"]].copy()
davidson.columns = ["tweet_text", "label"]
davidson["source"] = "Davidson et al. 2017"

print(f"Loaded {davidson.shape[0]} rows")
print(davidson["label"].value_counts(), "\n")

# -------------------------------
# Step 2: GESIS
# -------------------------------
print("[Step 2] Processing GESIS dataset")
gesis = pd.read_csv("data/raw/sexism_data.csv")
gesis = gesis[["text", "sexist"]].copy()
gesis.columns = ["tweet_text", "sexist"]
gesis["label"] = gesis["sexist"].map({True: "Sexist", False: "Not Hate"})
gesis = gesis[["tweet_text", "label"]].copy()
gesis["source"] = "GESIS Sexism (2021)"

print(f"Loaded {gesis.shape[0]} rows")
print(gesis["label"].value_counts(), "\n")

# -------------------------------
# Step 3: MMHS150K
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

# Exclude Homophobe to match your current label setup
mmhs = mmhs[mmhs["label"] != "Homophobe"]
mmhs = mmhs[["tweet_text", "label"]].copy()
mmhs["source"] = "MMHS150K (2020)"

print(f"Loaded {mmhs.shape[0]} rows")
print(mmhs["label"].value_counts(), "\n")

# -------------------------------
# Step 4: Combine
# -------------------------------
print("[Step 4] Combining datasets")
combined = pd.concat([davidson, gesis, mmhs], ignore_index=True)
print(f"Combined rows before cleaning: {combined.shape[0]}\n")

# -------------------------------
# Step 5: Clean text
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
# Step 6: Cap Not Hate at 15000
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
# Step 7: Add features
# -------------------------------
print("[Step 7] Adding features")
combined["tweet_length"] = combined["tweet_text"].apply(len)
combined["word_count"] = combined["tweet_text"].apply(lambda x: len(x.split()))

# Hate vs not hate binary
combined["hate_binary"] = combined["label"].apply(
    lambda x: "Not Hate" if x == "Not Hate" else "Hate"
)

# Keywords used for the keyword comparison plot
keywords = [
    "nigger", "white trash", "retard", "retarded", "faggot",
    "race card", "redneck", "cunt", "twat", "hillbilly",
    "sjw", "buildthewall", "nigga", "dyke"
]

def contains_keyword(text, keyword):
    return keyword in text

keyword_flags = {}
for kw in keywords:
    col_name = f"kw_{kw.replace(' ', '_')}"
    keyword_flags[col_name] = combined["tweet_text"].apply(lambda x: contains_keyword(x, kw))

for col, values in keyword_flags.items():
    combined[col] = values

print("Tweet length summary:")
print(combined["tweet_length"].describe(), "\n")

# -------------------------------
# Step 8: Save
# -------------------------------
output_path = "data/processed/combined_dataset.csv"
combined.to_csv(output_path, index=False)

print(f"Final dataset shape: {combined.shape}")
print(f"Saved cleaned dataset to: {output_path}")




