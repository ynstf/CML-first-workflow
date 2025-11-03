# train_experiments.py
import os
import json
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------
# 0) Helpers
# ---------------------------
def safe_metric(fn, y_true, y_pred):
    """
    Try binary average first (suitable for Titanic). If it fails (multiclass),
    fall back to weighted average.
    """
    try:
        return float(fn(y_true, y_pred, average="binary"))
    except Exception:
        return float(fn(y_true, y_pred, average="weighted"))

# ---------------------------
# 1) Load & preprocess Titanic (same as train.py)
# ---------------------------
print("🔁 Loading Titanic dataset and preprocessing...")
df = sns.load_dataset("titanic").dropna(subset=["survived"])
df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]

# avoid chained-assignment warnings
df = df.copy()
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# One-hot encode categorical columns (same as train.py)
df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

X = df.drop("survived", axis=1).values
y = df["survived"].values

# Split (same random_state/test_size as in your other scripts)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"Data shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

# ---------------------------
# 2) Define experiments
# ---------------------------
experiments = [
    # RandomForest with different n_estimators
    {
        "name": "RandomForest_50",
        "model": RandomForestClassifier(n_estimators=50, random_state=42),
        "params": {"n_estimators": 50},
    },
    {
        "name": "RandomForest_100",
        "model": RandomForestClassifier(n_estimators=100, random_state=42),
        "params": {"n_estimators": 100},
    },
    {
        "name": "RandomForest_200",
        "model": RandomForestClassifier(n_estimators=200, random_state=42),
        "params": {"n_estimators": 200},
    },
    # SVM with different kernels
    {
        "name": "SVM_linear",
        "model": SVC(kernel="linear", random_state=42),
        "params": {"kernel": "linear"},
    },
    {
        "name": "SVM_rbf",
        "model": SVC(kernel="rbf", random_state=42),
        "params": {"kernel": "rbf"},
    },
    {
        "name": "SVM_poly",
        "model": SVC(kernel="poly", degree=3, random_state=42),
        "params": {"kernel": "poly", "degree": 3},
    },
]

# ---------------------------
# 3) Prepare folders
# ---------------------------
os.makedirs("experiments", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------
# 4) Train experiments
# ---------------------------
results = []
print("=" * 60)
print("Entraînement de la matrice d'expériences")
print("=" * 60)

for exp in experiments:
    name = exp["name"]
    print(f"\nEntraînement: {name}")

    model = exp["model"]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # compute metrics with safe fallback for average parameter
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = safe_metric(precision_score, y_test, y_pred)
    recall = safe_metric(recall_score, y_test, y_pred)
    f1 = safe_metric(f1_score, y_test, y_pred)

    metrics = {
        "name": name,
        "algorithm": model.__class__.__name__,
        "params": exp.get("params", {}),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "test_size": int(len(X_test)),
        "timestamp": datetime.now().isoformat(),
    }

    # Save model
    model_path = f"models/{name}.pkl"
    joblib.dump(model, model_path)

    # Save metrics individually
    metrics_path = f"experiments/{name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    results.append(metrics)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Modèle sauvegardé: {model_path}")

print("\n" + "=" * 60)
print(f"Entraînement terminé: {len(results)} modèles")
print("=" * 60)

# Save aggregated results
with open("experiments/all_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Résultats sauvegardés dans experiments/all_results.json")
