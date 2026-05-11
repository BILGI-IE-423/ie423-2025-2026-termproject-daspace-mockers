import pandas as pd
from datasets import load_dataset
import os

# ── Create folder structure ──────────────────────────────────────────────────
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)

# ── Load GoEmotions (raw) ────────────────────────────────────────────────────
print("Loading GoEmotions dataset...")
dataset = load_dataset("google-research-datasets/go_emotions", "raw")
df = pd.DataFrame(dataset['train'])
print(f"Raw dataset shape: {df.shape}")
print(f"Unique comments: {df['id'].nunique()}")

# ── Aggregate by comment ID (one row per comment) ────────────────────────────
emotion_cols = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

df_agg = df.groupby('id').agg(
    text=('text', 'first'),
    **{col: (col, 'max') for col in emotion_cols}
).reset_index()

print(f"Aggregated dataset shape: {df_agg.shape}")

# ── Apply Parrot mapping ─────────────────────────────────────────────────────
parrot_mapping = {
    'joy':      ['joy', 'amusement', 'excitement', 'optimism', 'pride', 'relief', 'gratitude', 'approval'],
    'love':     ['love', 'admiration', 'caring', 'desire'],
    'surprise': ['surprise', 'realization', 'confusion', 'curiosity'],
    'anger':    ['anger', 'annoyance', 'disapproval', 'disgust', 'embarrassment'],
    'sadness':  ['sadness', 'disappointment', 'grief', 'remorse'],
    'fear':     ['fear', 'nervousness'],
    'neutral':  ['neutral']
}

def assign_parrot_label(row):
    for parrot_class, emotions in parrot_mapping.items():
        if any(row[e] == 1 for e in emotions):
            return parrot_class
    return None

df_agg['parrot_label'] = df_agg.apply(assign_parrot_label, axis=1)

# ── Drop unlabeled rows ──────────────────────────────────────────────────────
df_agg = df_agg[df_agg['parrot_label'].notna()].reset_index(drop=True)
print(f"After dropping unlabeled: {len(df_agg)} rows")

# ── Save raw aggregated ──────────────────────────────────────────────────────
df_agg[['text', 'parrot_label']].to_csv('data/raw/go_emotions_raw.csv', index=False)
print("Saved to data/raw/go_emotions_raw.csv")

# ── Print class distribution ─────────────────────────────────────────────────
print("\nClass distribution:")
print(df_agg['parrot_label'].value_counts())
