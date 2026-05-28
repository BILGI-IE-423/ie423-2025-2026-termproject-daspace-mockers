# Required: pip install deap
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from deap import base, creator, tools, algorithms
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
emotion_classes = ['anger', 'fear', 'joy', 'love', 'neutral', 'sadness', 'surprise']

# ── Fitness function ─────────────────────────────────────────────────────────
def evaluate(individual):
    weights = {emotion_classes[i]: max(individual[i], 0.1) for i in range(7)}
    clf = LogisticRegression(C=0.5, class_weight=weights, random_state=42, max_iter=500)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X_train_tfidf, y_train):
        X_tr, X_val = X_train_tfidf[train_idx], X_train_tfidf[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='macro'))
    return (np.mean(scores),)

# ── GA setup ─────────────────────────────────────────────────────────────────
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.1, 25.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=7)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

# ── Run GA ───────────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

population = toolbox.population(n=10)
NGEN = 5
best_per_gen = []

print("Running Genetic Algorithm...")
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.3)
    fits = list(map(toolbox.evaluate, offspring))
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    best = tools.selBest(population, k=1)[0]
    best_per_gen.append(best.fitness.values[0])
    print(f"Gen {gen+1} | Best Macro F1: {best.fitness.values[0]:.4f}")

# ── Best weights found ───────────────────────────────────────────────────────
best_individual = tools.selBest(population, k=1)[0]
best_weights = {emotion_classes[i]: round(max(best_individual[i], 0.1), 3) for i in range(7)}
print(f"\nBest weights found: {best_weights}")

# ── Train final model with GA weights ────────────────────────────────────────
lr_ga = LogisticRegression(C=0.5, class_weight=best_weights, random_state=42, max_iter=1000)
lr_ga.fit(X_train_tfidf, y_train)
y_pred_ga = lr_ga.predict(X_test_tfidf)

print("\n=== LR + Genetic Algorithm Weights ===")
print(classification_report(y_test, y_pred_ga))

# ── Confusion matrix ─────────────────────────────────────────────────────────
cm_ga = confusion_matrix(y_test, y_pred_ga, labels=labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_ga, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title('Confusion Matrix — LR + GA Weights')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('visuals/15_cm_lr_ga.png')
plt.show()

# ── GA convergence plot ───────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(range(1, NGEN+1), best_per_gen, marker='o', color='steelblue')
plt.title('GA Convergence — Best Macro F1 per Generation')
plt.xlabel('Generation')
plt.ylabel('Macro F1')
plt.tight_layout()
plt.savefig('visuals/16_ga_convergence.png')
plt.show()
