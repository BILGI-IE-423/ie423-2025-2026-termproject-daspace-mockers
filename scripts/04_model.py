import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import os

os.makedirs('visuals', exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv('data/processed/combined_dataset.csv')
X = df['text']
y = df['parrot_label']

# ── Train/test split (stratified) ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── TF-IDF vectorization ─────────────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ── Helper: plot confusion matrix ────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, title, filename):
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'visuals/{filename}')
    plt.show()

# ── Baseline models ──────────────────────────────────────────────────────────
# Logistic Regression
lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)
print("\n=== Logistic Regression (Baseline) ===")
print(classification_report(y_test, y_pred_lr))
plot_confusion_matrix(y_test, y_pred_lr, 'Confusion Matrix — Logistic Regression', '04_cm_lr_baseline.png')

# SVM
svm = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)
print("\n=== SVM (Baseline) ===")
print(classification_report(y_test, y_pred_svm))
plot_confusion_matrix(y_test, y_pred_svm, 'Confusion Matrix — SVM', '05_cm_svm_baseline.png')

# kNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_tfidf, y_train)
y_pred_knn = knn.predict(X_test_tfidf)
print("\n=== kNN (Baseline) ===")
print(classification_report(y_test, y_pred_knn))
plot_confusion_matrix(y_test, y_pred_knn, 'Confusion Matrix — kNN', '06_cm_knn_baseline.png')

# ── Best model: LR + GridSearch + Aggressive Class Weights ───────────────────
custom_weights = {
    'joy': 1, 'surprise': 2, 'anger': 2, 'love': 2,
    'neutral': 3, 'sadness': 5, 'fear': 20
}

param_grid = {
    'C': [0.1, 0.5, 1, 5, 10],
    'class_weight': ['balanced', custom_weights]
}

grid = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid, cv=5, scoring='f1_macro', n_jobs=-1
)
grid.fit(X_train_tfidf, y_train)
print(f"\nBest params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")

y_pred_best = grid.predict(X_test_tfidf)
print("\n=== Best Model (LR + GridSearch) ===")
print(classification_report(y_test, y_pred_best))
plot_confusion_matrix(y_test, y_pred_best, 'Confusion Matrix — Best LR (GridSearch)', '07_cm_lr_best.png')

# ── Model comparison chart ───────────────────────────────────────────────────
models = ['LR Baseline', 'SVM Baseline', 'kNN Baseline', 'LR Best (GridSearch)']
accuracy = [0.46, 0.49, 0.30, 0.55]
macro_f1 = [0.37, 0.36, 0.17, 0.40]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, accuracy, width, label='Accuracy', color='steelblue')
ax.bar(x + width/2, macro_f1, width, label='Macro F1', color='coral')
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('visuals/08_model_comparison.png')
plt.show()
