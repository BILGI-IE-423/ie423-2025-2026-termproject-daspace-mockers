import pandas as pd
import re
import os

# ── Load raw data ────────────────────────────────────────────────────────────
df = pd.read_csv('data/raw/go_emotions_raw.csv')
print(f"Loaded: {df.shape}")

# ── Text cleaning ────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text)
    text = text.encode('ascii', 'ignore').decode('ascii')  # remove encoding artifacts
    text = re.sub(r'http\S+', '', text)                    # remove URLs
    text = re.sub(r'\s+', ' ', text).strip()               # normalize whitespace
    text = text.lower()                                     # lowercase
    return text

df['text'] = df['text'].apply(clean_text)

# ── Remove empty rows after cleaning ────────────────────────────────────────
before = len(df)
df = df[df['text'].str.strip() != ''].reset_index(drop=True)
print(f"Removed {before - len(df)} empty rows after cleaning")

# ── Add features ─────────────────────────────────────────────────────────────
df['comment_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

# ── Save processed data ──────────────────────────────────────────────────────
df.to_csv('data/processed/combined_dataset.csv', index=False)
print(f"Saved processed dataset: {df.shape}")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\nClass distribution:")
print(df['parrot_label'].value_counts())
print(f"\nAverage comment length: {df['comment_length'].mean():.1f} characters")
print(f"Average word count: {df['word_count'].mean():.1f} words")
