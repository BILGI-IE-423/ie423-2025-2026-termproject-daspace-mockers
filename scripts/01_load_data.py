import pandas as pd
import json
from collections import Counter
import os
import shutil

# Create folder structure
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)

# Auto-move datasets to data/raw/ if found in root
files_to_move = ['labeled_data.csv', 'sexism_data.csv', 'MMHS150K_GT.json']
for file in files_to_move:
    if os.path.exists(file) and not os.path.exists(f'data/raw/{file}'):
        shutil.move(file, f'data/raw/{file}')
        print(f"Moved {file} → data/raw/")

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)

print("Starting dataset loading...\n")

# -------------------------------
# Dataset 1: Davidson
# -------------------------------
davidson = pd.read_csv('data/raw/labeled_data.csv')
print("\n[Dataset 1] Davidson")
print("Shape:", davidson.shape)
print("Columns:", davidson.columns.tolist())
print("Missing values:\n", davidson.isnull().sum())

if 'class' in davidson.columns:
    print("\nClass distribution:")
    print(davidson['class'].value_counts())

# -------------------------------
# Dataset 2: GESIS
# -------------------------------
gesis = pd.read_csv('data/raw/sexism_data.csv')
print("\n[Dataset 2] GESIS")
print("Shape:", gesis.shape)
print("Columns:", gesis.columns.tolist())
print("Missing values:\n", gesis.isnull().sum())

if 'sexist' in gesis.columns:
    print("\nClass distribution:")
    print(gesis['sexist'].value_counts())

# -------------------------------
# Dataset 3: MMHS150K
# -------------------------------
with open('data/raw/MMHS150K_GT.json', 'r') as f:
    data = json.load(f)

mmhs = pd.DataFrame.from_dict(data, orient='index')

label_map = {
    0: 'Not Hate',
    1: 'Racist',
    2: 'Sexist',
    3: 'Homophobe',
    4: 'Religion',
    5: 'Other Hate'
}

def majority_vote(labels):
    return Counter(labels).most_common(1)[0][0]

mmhs['label'] = mmhs['labels'].apply(majority_vote).map(label_map)

print("\n[Dataset 3] MMHS150K")
print("Shape:", mmhs.shape)
print("Columns:", mmhs.columns.tolist())
print("Missing values:\n", mmhs.isnull().sum())
print("\nLabel distribution:")
print(mmhs['label'].value_counts())

print("\nAll datasets loaded successfully.")
