# Protocole d'Évaluation

## Métriques Principales

| Métrique | Description |
|----------|-------------|
| **Accuracy** | (TP + TN) / Total |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1-Score** | 2 × (P × R) / (P + R) |

## Objectifs de Performance

- Model A (Baseline): F1 ≥ 0.70
- Model B (Deep): ΔF1 ≥ +0.05 vs Model A

## Procédure d'Évaluation

1. Charger le meilleur checkpoint de chaque modèle
2. Évaluer sur le test set (1,200 images)
3. Calculer toutes les métriques
4. Générer les matrices de confusion
5. Comparer les performances

## Ablation Studies (Optionnel)

| Expérience | Variable | Objectif |
|------------|----------|----------|
| Ablation 1 | Model B sans BatchNorm | Impact de BatchNorm |
| Ablation 2 | Model B sans skip connections | Impact des skips |
| Ablation 3 | Sans data augmentation | Impact de l'augmentation |
