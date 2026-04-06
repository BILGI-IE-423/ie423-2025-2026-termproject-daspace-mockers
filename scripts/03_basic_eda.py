# ============================================================
# 03_basic_eda.py
# Exploratory Data Analysis & Visualizations
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # for saving without display
import seaborn as sns
import os

# Create output folder if it doesn't exist
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)

# ─────────────────────────────────────────
# LOAD COMBINED DATASET
# ─────────────────────────────────────────
df = pd.read_csv('data/processed/combined_dataset.csv')
print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")

# ─────────────────────────────────────────
# FIGURE 1: Label Distribution
# ─────────────────────────────────────────

label_counts = df['label'].value_counts()
colors = ['#e74c3c', '#3498db', '#e67e22',
          '#2ecc71', '#9b59b6', '#1abc9c']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(label_counts.index, label_counts.values, color=colors)
ax.set_title('Label Distribution in Combined Dataset', fontsize=14, fontweight='bold')
ax.set_xlabel('Label', fontsize=12)
ax.set_ylabel('Number of Tweets', fontsize=12)
ax.set_xticklabels(label_counts.index, rotation=15, ha='right')

# Add count labels on top of bars
for bar, count in zip(bars, label_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 200,
            f'{count:,}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/figures/01_label_distribution.png', dpi=150)
plt.close()
print("Saved: outputs/figures/01_label_distribution.png")

# ─────────────────────────────────────────
# FIGURE 2: Tweet Length Distribution by Label
# ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
labels = df['label'].unique()
for label in labels:
    subset = df[df['label'] == label]['tweet_length']
    subset.plot(kind='kde', ax=ax, label=label, linewidth=2)

ax.set_title('Tweet Length Distribution by Label', fontsize=14, fontweight='bold')
ax.set_xlabel('Tweet Length (characters)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(title='Label', fontsize=10)
ax.set_xlim(0, 200)

plt.tight_layout()
plt.savefig('outputs/figures/02_tweet_length_distribution.png', dpi=150)
plt.close()
print("Saved: outputs/figures/02_tweet_length_distribution.png")

# ─────────────────────────────────────────
# FIGURE 3: Word Count Distribution by Label
# ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='word_count', by='label', ax=ax,
           patch_artist=True, showfliers=False)
ax.set_title('Word Count Distribution by Label', fontsize=14, fontweight='bold')
ax.set_xlabel('Label', fontsize=12)
ax.set_ylabel('Word Count', fontsize=12)
plt.suptitle('')  # remove default boxplot title
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')

plt.tight_layout()
plt.savefig('outputs/figures/03_word_count_boxplot.png', dpi=150)
plt.close()
print("Saved: outputs/figures/03_word_count_boxplot.png")

# ─────────────────────────────────────────
# FIGURE 4: Dataset Source Contribution
# ─────────────────────────────────────────

source_counts = {
    'Davidson et al. 2017': 24783,
    'GESIS Sexism (2021)': 13631,
    'MMHS150K (2020)': 144897
}

fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(source_counts.values(),
       labels=source_counts.keys(),
       autopct='%1.1f%%',
       colors=['#3498db', '#e74c3c', '#2ecc71'],
       startangle=140)
ax.set_title('Raw Tweet Contribution by Dataset Source',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/04_dataset_source_pie.png', dpi=150)
plt.close()
print("Saved: outputs/figures/04_dataset_source_pie.png")

# ─────────────────────────────────────────
# TABLE: Summary Statistics
# ─────────────────────────────────────────

summary = df.groupby('label').agg(
    Count=('tweet_text', 'count'),
    Avg_Length=('tweet_length', 'mean'),
    Avg_Words=('word_count', 'mean')
).round(2)

summary['Percentage'] = (summary['Count'] /
                         summary['Count'].sum() * 100).round(2)
summary = summary.sort_values('Count', ascending=False)

print("\nSummary Table:")
print(summary.to_string())

summary.to_csv('outputs/tables/01_label_summary.csv')
print("\nSaved: outputs/tables/01_label_summary.csv")


print("EDA complete. All figures saved to outputs/figures/")
