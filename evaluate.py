# evaluate.py (adapted for Titanic project)
import os
from pathlib import Path
import json

# 0) forcer un backend qui n'a pas besoin d'écran
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("📂 cwd        :", os.getcwd())
print("📂 script dir :", BASE_DIR)
print("📂 reports    :", REPORTS_DIR)

# 1) test d'écriture simple
test_file = REPORTS_DIR / "DEBUG_EVALUATE_RAN.txt"
test_file.write_text("evaluate.py a bien été exécuté jusqu'en bas.\n")
print(f"📝 Fichier texte créé : {test_file}")

# 2) Load and preprocess the Titanic data in the same way as train.py
print("🔁 Chargement et preprocessing des données Titanic (comme dans train.py)...")
df = sns.load_dataset("titanic").dropna(subset=["survived"])
df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]

# same fills as in train.py (avoid chained-assignment warnings by assigning column)
df = df.copy()
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# One-hot encode categorical columns exactly as train.py did
df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

# Split features and labels (same random_state/test_size as train.py)
X = df.drop("survived", axis=1)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Data prepared. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# 3) Load model
model_path = BASE_DIR / "models" / "titanic_model.pkl"
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
print("🔎 loading model from :", model_path)
model = joblib.load(model_path)
print("✅ model loaded")

# 4) Predict & compute metrics
y_pred = model.predict(X_test)
accuracy = float(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(f"✅ prediction done. accuracy={accuracy:.4f}")

# Save metrics.json (overwrite / create)
metrics = {
    "accuracy": accuracy,
    "test_size": int(len(X_test)),
    "n_features": int(X_test.shape[1]),
    "model_path": str(model_path.name),
}
with open(BASE_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"✅ metrics saved to {BASE_DIR / 'metrics.json'}")

# 5) Confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Survived", "Survived"],
    yticklabels=["Not Survived", "Survived"],
)
plt.title("Confusion Matrix - Titanic")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
cm_path = REPORTS_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"✅ confusion matrix saved to {cm_path}")

# 6) Feature importance for classifiers that expose it (or coefficients for linear models)
fi_path = REPORTS_DIR / "feature_importance.png"
if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
    # get feature names from X_test to map importances
    feature_names = list(X_test.columns)
    if hasattr(model, "feature_importances_"):
        fi = np.array(model.feature_importances_)
        title = "Feature importance (tree-based)"
    else:
        # logistic regression: coef_ shape (n_classes, n_features) or (1, n_features)
        coef = np.array(model.coef_)
        if coef.ndim == 1 or coef.shape[0] == 1:
            fi = np.abs(coef).ravel()
        else:
            # multi-class: sum absolute coef across classes
            fi = np.sum(np.abs(coef), axis=0)
        title = "Feature importance (|coef| for linear model)"

    # sort descending
    idx = np.argsort(fi)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(fi)), fi[idx])
    plt.xticks(range(len(fi)), [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fi_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✅ feature importance saved to {fi_path}")
else:
    print("ℹ️ modèle sans feature_importances_ ni coef_ → on saute la figure 'feature_importance'.")

print("🎉 Terminé.")
