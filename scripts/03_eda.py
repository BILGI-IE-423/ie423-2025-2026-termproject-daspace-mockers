import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

print("Starting EDA...\n")

# -------------------------------
# Load processed dataset
# -------------------------------
df = pd.read_csv("data/processed/combined_dataset.csv")

# Consistent label order to match your plots
label_order = [
    "Offensive Language",
    "Not Hate",
    "Racist",
    "Other Hate",
    "Sexist",
    "Religion"
]

# -------------------------------
# 1. Label Distribution
# -------------------------------
print("Creating label distribution plot...")

label_counts = df["label"].value_counts().reindex(label_order)
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=label_counts.index, y=label_counts.values)
plt.title("Label Distribution in Combined Dataset", fontsize=18, weight="bold")
plt.xlabel("Label", fontsize=14)
plt.ylabel("Number of Tweets", fontsize=14)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

for i, value in enumerate(label_counts.values):
    ax.text(i, value + max(label_counts.values) * 0.01, f"{value:,}",
            ha="center", va="bottom", fontsize=10)

plt.savefig("outputs/figures/01_label_distribution.png", dpi=300)
plt.close()

# Save summary table
label_summary = pd.DataFrame({
    "label": label_counts.index,
    "count": label_counts.values,
    "percentage": (label_counts.values / len(df) * 100).round(2)
})
label_summary.to_csv("outputs/tables/01_label_summary.csv", index=False)

# -------------------------------
# 2. Tweet Length Distribution by Label
# -------------------------------
print("Creating tweet length distribution plot...")

plt.figure(figsize=(12, 6))
for label in ["Racist", "Offensive Language", "Not Hate", "Sexist", "Other Hate", "Religion"]:
    subset = df[df["label"] == label]
    sns.kdeplot(subset["tweet_length"], label=label, linewidth=2)

plt.title("Tweet Length Distribution by Label", fontsize=16, weight="bold")
plt.xlabel("Tweet Length (characters)", fontsize=13)
plt.ylabel("Density", fontsize=13)
plt.xlim(0, 200)
plt.legend(title="Label")
plt.tight_layout()
plt.savefig("outputs/figures/02_tweet_length_distribution.png", dpi=300)
plt.close()

# -------------------------------
# 3. Word Count Distribution by Label
# -------------------------------
print("Creating word count boxplot...")

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="label", y="word_count", order=label_order)
plt.title("Word Count Distribution by Label", fontsize=18, weight="bold")
plt.xlabel("Label", fontsize=14)
plt.ylabel("Word Count", fontsize=14)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("outputs/figures/03_word_count_boxplot.png", dpi=300)
plt.close()

# -------------------------------
# 4. Dataset Source Contribution Pie Chart
# -------------------------------
print("Creating dataset source contribution pie chart...")

source_counts = df["source"].value_counts().reindex([
    "Davidson et al. 2017",
    "GESIS Sexism (2021)",
    "MMHS150K (2020)"
])

plt.figure(figsize=(8, 8))
plt.pie(
    source_counts.values,
    labels=source_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    textprops={"fontsize": 12}
)
plt.title("Raw Tweet Contribution by Dataset Source", fontsize=18, weight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/04_dataset_source_pie.png", dpi=300)
plt.close()

# -------------------------------
# 5. Keyword Hate vs Not Hate Distribution
# -------------------------------
print("Creating keyword hate vs not hate distribution plot...")

keywords = [
    "nigger", "white trash", "retard", "retarded", "faggot",
    "race card", "redneck", "cunt", "twat", "hillbilly",
    "sjw", "buildthewall", "nigga", "dyke"
]

keyword_results = []

for kw in keywords:
    mask = df["tweet_text"].str.contains(kw, case=False, regex=False, na=False)
    subset = df[mask]
    total = len(subset)

    if total > 0:
        hate_pct = (subset["hate_binary"] == "Hate").mean() * 100
        not_hate_pct = (subset["hate_binary"] == "Not Hate").mean() * 100
    else:
        hate_pct = 0
        not_hate_pct = 0

    keyword_results.append({
        "keyword": kw,
        "Hate": hate_pct,
        "Not Hate": not_hate_pct
    })

keyword_df = pd.DataFrame(keyword_results)

plt.figure(figsize=(12, 6))
plt.bar(keyword_df["keyword"], keyword_df["Hate"], label="Hate")
plt.bar(keyword_df["keyword"], keyword_df["Not Hate"],
        bottom=keyword_df["Hate"], label="Not Hate")

plt.title("Keyword Hate vs Not Hate Distribution", fontsize=16)
plt.ylabel("%", fontsize=13)
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/05_keyword_distribution.png", dpi=300)
plt.close()

# Save keyword table
keyword_df.to_csv("outputs/tables/05_keyword_distribution.csv", index=False)

print("\nEDA completed successfully.")
print("Saved figures in outputs/figures/")
print("Saved tables in outputs/tables/")




