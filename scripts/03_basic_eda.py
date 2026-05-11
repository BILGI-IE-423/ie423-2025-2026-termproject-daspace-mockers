import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)

df = pd.read_csv('data/processed/combined_dataset.csv')

# ── 1. Class distribution bar chart ─────────────────────────────────────────
plt.figure(figsize=(10, 5))
df['parrot_label'].value_counts().plot(kind='bar', color='steelblue')
plt.title('Emotion Class Distribution (Parrot Mapping)')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/figures/01_class_distribution.png')
plt.show()

# ── 2. Comment length KDE by label ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
for label in df['parrot_label'].unique():
    subset = df[df['parrot_label'] == label]
    subset['comment_length'].plot(kind='kde', label=label)
plt.title('Comment Length Distribution by Emotion')
plt.xlabel('Comment Length (characters)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/02_comment_length_kde.png')
plt.show()

# ── 3. Word count boxplot by label ───────────────────────────────────────────
plt.figure(figsize=(10, 5))
df.boxplot(column='word_count', by='parrot_label')
plt.title('Word Count by Emotion')
plt.suptitle('')
plt.xlabel('Emotion')
plt.ylabel('Word Count')
plt.tight_layout()
plt.savefig('outputs/figures/03_word_count_boxplot.png')
plt.show()

# ── 4. Summary table ─────────────────────────────────────────────────────────
summary = df.groupby('parrot_label').agg(
    count=('text', 'count'),
    avg_length=('comment_length', 'mean'),
    avg_words=('word_count', 'mean')
).round(2)

summary.to_csv('outputs/tables/01_label_summary.csv')
print(summary)
