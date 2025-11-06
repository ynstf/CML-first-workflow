echo "# 🏆 Rapport de Comparaison de Modèles - Exercice 3" >> report.md
echo "" >> report.md

echo "## 🎯 Meilleur Modèle Sélectionné" >> report.md
echo "" >> report.md

cat experiments/best_model.json >> report.md

echo "" >> report.md

echo "## 📊 Comparaison des Accuracy" >> report.md
echo "" >> report.md
cml publish reports/accuracy_comparison.png --md >> report.md
echo "" >> report.md

echo "## 📈 Comparaison de Toutes les Métriques" >> report.md
echo "" >> report.md
cml publish reports/all_metrics_comparison.png --md >> report.md
echo "" >> report.md

echo "## 🔥 Heatmap des Performances" >> report.md
echo "" >> report.md
cml publish reports/performance_heatmap.png --md >> report.md
echo "" >> report.md

echo "## 📋 Résultats Détaillés" >> report.md
echo "" >> report.md
echo "<details>" >> report.md
echo "<summary>Cliquez pour voir tous les résultats</summary>" >> report.md
echo "" >> report.md
echo '```json' >> report.md
cat experiments/all_results.json >> report.md
echo '```' >> report.md
echo "</details>" >> report.md

cml comment create report.md
