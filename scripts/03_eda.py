import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create folders if they don't exist
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

print("Starting EDA...\n")

# -------------------------------
# Load processed dataset
# -------------------------------
df = pd.read_csv("data/processed/combined_dataset.csv")

print("Dataset loaded.")
print(f"Shape: {df.shape}\n")

# -------------------------------
# 1. Class Distribution (Q2)
# -------------------------------
print("Creating class distribution plot...")

plt.figure()
sns.countplot(data=df, x="label")
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("outputs/figures/class_distribution.png")
plt.close()

# Save table
class_counts = df["label"].value_counts()
class_counts.to_csv("outputs/tables/class_distribution.csv")

# -------------------------------
# 2. Category Overlap (Q3)
# -------------------------------
print("Creating category overlap plot...")

plt.figure()
sns.kdeplot(data=df, x="tweet_length", hue="label", fill=True)
plt.title("Tweet Length Distribution by Category")
plt.tight_layout()
plt.savefig("outputs/figures/category_overlap.png")
plt.close()

# -------------------------------
# 3. Harmful vs Not (Q1)
# -------------------------------
print("Creating harmful vs non-harmful plot...")

# Create column if not exists
df["is_harmful"] = df["label"].apply(
    lambda x: 0 if x == "Not Hate" else 1
)

plt.figure()
sns.kdeplot(data=df, x="tweet_length", hue="is_harmful", fill=True)
plt.title("Harmful vs Non-Harmful Length Distribution")
plt.tight_layout()
plt.savefig("outputs/figures/harmful_vs_not.png")
plt.close()

# -------------------------------
# 4. Length Summary Table
# -------------------------------
print("Creating summary table...")

summary = df.groupby("label")["tweet_length"].describe()
summary.to_csv("outputs/tables/length_summary.csv")

print("\nEDA completed successfully.")
