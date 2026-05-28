import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import os

os.makedirs('visuals', exist_ok=True)

# ── Load and prepare data ────────────────────────────────────────────────────
df = pd.read_csv('data/processed/combined_dataset.csv')
X = df['text']
y = df['parrot_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

labels = sorted(y_test.unique())

# ── Helper: plot confusion matrix ────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, title, filename):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'visuals/{filename}')
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# COMBATTING FEAR — Minority Class Experiments
# ══════════════════════════════════════════════════════════════════════════════

# ── Experiment 1: SMOTE ──────────────────────────────────────────────────────
print("=== Experiment 1: SMOTE ===")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
print(f"After SMOTE — training size: {len(y_train_smote)}")

lr_smote = LogisticRegression(random_state=42, max_iter=1000)
lr_smote.fit(X_train_smote, y_train_smote)
y_pred_lr_smote = lr_smote.predict(X_test_tfidf)
print("\n=== LR + SMOTE ===")
print(classification_report(y_test, y_pred_lr_smote))
plot_confusion_matrix(y_test, y_pred_lr_smote, 'Confusion Matrix — LR + SMOTE', '09_cm_lr_smote.png')

svm_smote = LinearSVC(random_state=42, max_iter=2000)
svm_smote.fit(X_train_smote, y_train_smote)
y_pred_svm_smote = svm_smote.predict(X_test_tfidf)
print("\n=== SVM + SMOTE ===")
print(classification_report(y_test, y_pred_svm_smote))
plot_confusion_matrix(y_test, y_pred_svm_smote, 'Confusion Matrix — SVM + SMOTE', '10_cm_svm_smote.png')

# ── Experiment 2: LSA (Dimensionality Reduction) ────────────────────────────
print("\n=== Experiment 2: LSA ===")
svd = TruncatedSVD(n_components=300, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)
print(f"Reduced shape: {X_train_svd.shape}")

lr_svd = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_svd.fit(X_train_svd, y_train)
y_pred_lr_svd = lr_svd.predict(X_test_svd)
print("\n=== LR + LSA ===")
print(classification_report(y_test, y_pred_lr_svd))
plot_confusion_matrix(y_test, y_pred_lr_svd, 'Confusion Matrix — LR + LSA', '11_cm_lr_lsa.png')

svm_svd = LinearSVC(random_state=42, max_iter=2000)
svm_svd.fit(X_train_svd, y_train)
y_pred_svm_svd = svm_svd.predict(X_test_svd)
print("\n=== SVM + LSA ===")
print(classification_report(y_test, y_pred_svm_svd, zero_division=0))
plot_confusion_matrix(y_test, y_pred_svm_svd, 'Confusion Matrix — SVM + LSA', '12_cm_svm_lsa.png')

# ── Experiment 3: Aggressive Class Weights ───────────────────────────────────
print("\n=== Experiment 3: Aggressive Class Weights ===")
custom_weights = {
    'joy': 1, 'surprise': 2, 'anger': 2, 'love': 2,
    'neutral': 3, 'sadness': 5, 'fear': 20
}

lr_weighted = LogisticRegression(class_weight=custom_weights, random_state=42, max_iter=1000)
lr_weighted.fit(X_train_tfidf, y_train)
y_pred_lr_weighted = lr_weighted.predict(X_test_tfidf)
print("\n=== LR + Aggressive Weights ===")
print(classification_report(y_test, y_pred_lr_weighted))
plot_confusion_matrix(y_test, y_pred_lr_weighted, 'Confusion Matrix — LR + Aggressive Weights', '13_cm_lr_weighted.png')

# ── Experiment 4: Computed Class Weights ─────────────────────────────────────
print("\n=== Experiment 4: Computed Class Weights ===")
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
computed_weights = dict(zip(classes, weights))
print(f"Computed weights: {computed_weights}")

lr_computed = LogisticRegression(class_weight=computed_weights, random_state=42, max_iter=1000)
lr_computed.fit(X_train_tfidf, y_train)
y_pred_lr_computed = lr_computed.predict(X_test_tfidf)
print("\n=== LR + Computed Weights ===")
print(classification_report(y_test, y_pred_lr_computed))

# ── All experiments summary chart ────────────────────────────────────────────
experiments = [
    'LR Baseline', 'SVM Baseline', 'kNN Baseline',
    'LR + SMOTE', 'SVM + SMOTE',
    'LR + LSA', 'SVM + LSA',
    'LR + Aggressive Weights', 'LR + Computed Weights',
    'LR + GridSearch (Best)'
]

macro_f1_scores = [0.37, 0.36, 0.17, 0.37, 0.35, 0.30, 0.25, 0.39, 0.37, 0.40]

plt.figure(figsize=(14, 6))
colors = ['coral' if s == max(macro_f1_scores) else 'steelblue' for s in macro_f1_scores]
plt.bar(experiments, macro_f1_scores, color=colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Macro F1 Score')
plt.title('COMBATTING FEAR — All Experiments Macro F1 Comparison')
plt.ylim(0, 0.6)
plt.tight_layout()
plt.savefig('visuals/14_all_experiments_comparison.png')
plt.show()
