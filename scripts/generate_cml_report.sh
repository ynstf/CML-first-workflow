#!/bin/bash
set -e

# Create report.md
echo "# 🏆 Rapport de Comparaison de Modèles - Exercice 3" > report.md
echo "" >> report.md

echo "## 🎯 Meilleur Modèle Sélectionné" >> report.md
echo "" >> report.md
echo '```json' >> report.md
[ -f experiments/best_model.json ] && cat experiments/best_model.json >> report.md || echo "{}" >> report.md
echo '```' >> report.md
echo "" >> report.md

echo "## 📊 Comparaison des Accuracy" >> report.md
echo "" >> report.md
[ -f reports/accuracy_comparison.png ] \
    && echo "![Accuracy](./reports/accuracy_comparison.png)" >> report.md \
    || echo "(missing accuracy_comparison.png)" >> report.md
echo "" >> report.md

echo "## 📈 Comparaison de Toutes les Métriques" >> report.md
echo "" >> report.md
[ -f reports/all_metrics_comparison.png ] \
    && echo "![All metrics](./reports/all_metrics_comparison.png)" >> report.md \
    || echo "(missing all_metrics_comparison.png)" >> report.md
echo "" >> report.md

echo "## 🔥 Heatmap des Performances" >> report.md
echo "" >> report.md
[ -f reports/performance_heatmap.png ] \
    && echo "![Heatmap](./reports/performance_heatmap.png)" >> report.md \
    || echo "(missing heatmap)" >> report.md
echo "" >> report.md

echo "## 📋 Résultats Détaillés" >> report.md
echo "" >> report.md
echo "<details>" >> report.md
echo "<summary>Cliquez pour voir tous les résultats</summary>" >> report.md
echo "" >> report.md
echo '```json' >> report.md
[ -f experiments/all_results.json ] && cat experiments/all_results.json >> report.md || echo "[]" >> report.md
echo '```' >> report.md
echo "</details>" >> report.md

echo "✅ report.md generated."
