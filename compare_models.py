# compare_models.py
import os
import json

# headless backend for plotting on CI
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------------------------
# 1) Load aggregated results
# ---------------------------
results_path = "experiments/all_results.json"
if not os.path.exists(results_path):
    raise FileNotFoundError(f"{results_path} not found. Run train_experiments.py first.")

with open(results_path, "r") as f:
    results = json.load(f)

if not results:
    raise ValueError("No experiment results found in experiments/all_results.json")

df = pd.DataFrame(results)

# sort by accuracy descending
df_sorted = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

print("=" * 70)
print("COMPARAISON DES MODÈLES")
print("=" * 70)

# print comparison table
print("\nTableau de comparaison (trié par accuracy):\n")
print(df_sorted[["name", "algorithm", "accuracy", "f1_score", "precision", "recall"]].to_string(index=False))

# ---------------------------
# 2) Best model
# ---------------------------
best_model = df_sorted.iloc[0]
best_model_info = {
    "best_model": best_model["name"],
    "algorithm": best_model["algorithm"],
    "params": best_model.get("params", {}),
    "accuracy": float(best_model["accuracy"]),
    "f1_score": float(best_model["f1_score"]),
    "precision": float(best_model["precision"]),
    "recall": float(best_model["recall"]),
}

os.makedirs("experiments", exist_ok=True)
with open("experiments/best_model.json", "w") as f:
    json.dump(best_model_info, f, indent=2)

print("\nMeilleur modèle sauvegardé dans experiments/best_model.json")

# ---------------------------
# 3) Create report figures
# ---------------------------
os.makedirs("reports", exist_ok=True)

# Visualization 1: Accuracy comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=df_sorted, x="name", y="accuracy", palette="viridis")
plt.title("Comparaison des Accuracy par Modèle", fontsize=14, fontweight="bold")
plt.xlabel("Modèle", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.savefig("reports/accuracy_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# Visualization 2: All metrics comparison (melted)
metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
df_melted = df.melt(id_vars=["name"], value_vars=metrics_to_plot, var_name="metric", value_name="score")

plt.figure(figsize=(14, 7))
sns.barplot(data=df_melted, x="name", y="score", hue="metric", palette="Set2")
plt.title("Comparaison de Toutes les Métriques", fontsize=14, fontweight="bold")
plt.xlabel("Modèle", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Métrique", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.savefig("reports/all_metrics_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# Visualization 3: Performance heatmap
df_heatmap = df_sorted[["name", "accuracy", "precision", "recall", "f1_score"]].set_index("name")
plt.figure(figsize=(10, 8))
sns.heatmap(df_heatmap.T, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={"label": "Score"}, linewidths=0.5)
plt.title("Heatmap des Performances", fontsize=14, fontweight="bold")
plt.xlabel("Modèle", fontsize=12)
plt.ylabel("Métrique", fontsize=12)
plt.tight_layout()
plt.savefig("reports/performance_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

print("\nVisualisations générées:")
print("  - reports/accuracy_comparison.png")
print("  - reports/all_metrics_comparison.png")
print("  - reports/performance_heatmap.png")

print("\n" + "=" * 70)
print("Comparaison terminée!")
print("=" * 70)
