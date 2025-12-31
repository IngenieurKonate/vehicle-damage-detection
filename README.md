# Vehicle Damage Detection

Détection automatique de dommages sur véhicules par CNN From Scratch.

## Vue d'Ensemble du Projet

Ce projet vise à concevoir, implémenter et comparer **deux architectures CNN from scratch** pour la détection automatique de dommages visuels sur véhicules (rayures, bosses, fissures). L'objectif est de démontrer une **compréhension profonde** des concepts de Deep Learning à travers une démarche de conception originale.

### Problématique

> Dans quelle mesure une architecture CNN conçue from scratch, s'inspirant des principes de VGG et ResNet, peut-elle détecter efficacement les dommages visuels sur véhicules ?

### Objectifs du Projet

| ID | Objectif | Priorité | Critère de Succès |
|----|----------|----------|-------------------|
| O1 | Concevoir une architecture baseline (VGG-like) from scratch | Critique | Architecture fonctionnelle, F1 ≥ 0.70 |
| O2 | Concevoir une architecture deep avec skip connections | Critique | Architecture fonctionnelle, ΔF1 ≥ +0.05 vs baseline |
| O3 | Comparer scientifiquement les deux architectures | Critique | Analyse comparative documentée |
| O4 | Produire un rapport académique rigoureux | Important | Justification de chaque choix architectural |
| O5 | Créer une présentation PowerPoint professionnelle | Important | Slides clairs, visuels, défense des choix |
| O6 | Développer une application Flask de démonstration | Secondaire | Interface web fonctionnelle pour prédiction |

### Portée (Scope)

**In-Scope :**
- Classification binaire : `damaged` vs `undamaged`
- Deux architectures CNN conçues from scratch
- Pipeline complet : prétraitement → entraînement → évaluation
- Notebooks reproductibles et documentés
- Rapport académique et présentation PowerPoint
- **[Secondaire]** Application Flask de démonstration

### Architectures

| Modèle | Description | Paramètres |
|--------|-------------|------------|
| **Model A** | Baseline CNN inspiré de VGG (convolutions 3×3 empilées) | ~6.5M |
| **Model B** | Deep CNN avec skip connections inspiré de ResNet | ~11M |

## Structure du Projet

```
vehicle-damage-detection/              # Code local (VS Code + Git)
│
├── README.md
├── PRD.md                             # Document de référence
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── configs/                           # Configuration YAML
│   ├── config.yaml
│   ├── model_a_config.yaml
│   └── model_b_config.yaml
│
├── notebooks/                         # Notebooks Jupyter (exécutés sur Colab)
│   ├── 00_setup_colab.ipynb          # Setup Colab + vérification GPU
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_baseline.ipynb
│   ├── 04_train_deep.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_analysis.ipynb
│
├── src/                               # Code source
│   ├── data/                         # Dataset, transforms, utils
│   ├── models/                       # BaselineCNN, DeepCNN, components
│   ├── training/                     # Trainer, callbacks, losses
│   ├── evaluation/                   # Metrics, visualization
│   └── utils/                        # Config, seed, paths, logging
│
├── app/                               # [Secondaire] Application Flask
├── scripts/                           # Scripts CLI
├── docs/                              # Documentation
├── presentation/                      # Slides PowerPoint
└── tests/                             # Tests unitaires
```

### Structure Google Drive

Les données volumineuses sont stockées sur Google Drive :

```
My Drive/ENSA_Deep_Learning/
├── datasets/
│   ├── raw/                          # CarDD + Stanford Cars
│   └── processed/                    # Dataset combiné (train/val/test)
├── checkpoints/
│   ├── model_a/
│   └── model_b/
└── outputs/
    ├── figures/
    └── logs/
```

## Données

Le projet combine deux datasets pour créer un dataset équilibré :

| Dataset | Classe | Images | Source |
|---------|--------|--------|--------|
| **CarDD** | damaged | 4,000 | [USTC](https://cardd-ustc.github.io/) |
| **Stanford Cars** | undamaged | 4,000 (échantillon) | [Kaggle 224×224](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-images-in-224x224) |
| **Total** | - | **8,000** | 50% / 50% |

### Split des Données

| Split | Images | Damaged | Undamaged |
|-------|--------|---------|-----------|
| Train | 5,600 (70%) | 2,800 | 2,800 |
| Validation | 1,200 (15%) | 600 | 600 |
| Test | 1,200 (15%) | 600 | 600 |

## Installation et Usage

### Option 1 : Google Colab (Recommandé)

1. Ouvrir `notebooks/00_setup_colab.ipynb` dans Colab
2. Monter Google Drive
3. Cloner le repo et configurer les chemins
4. Exécuter les notebooks dans l'ordre (01 → 06)

### Option 2 : Local

```bash
# Cloner le repository
git clone https://github.com/IngenieurKonate/vehicle-damage-detection.git
cd vehicle-damage-detection

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Pipeline

1. **Setup** : Configuration Colab, montage Drive, vérification GPU
2. **Exploration** : Visualisation des datasets, statistiques
3. **Prétraitement** : Combinaison, échantillonnage, split stratifié
4. **Entraînement Model A** : Baseline VGG-like
5. **Entraînement Model B** : Deep CNN avec skip connections
6. **Évaluation** : Comparaison des performances, matrices de confusion
7. **Analyse** : Étude des erreurs, ablation studies

## Objectifs de Performance

| Métrique | Model A (Baseline) | Model B (Deep) |
|----------|-------------------|----------------|
| F1-Score | ≥ 0.70 | ≥ 0.75 (ΔF1 ≥ +0.05) |

## Technologies

- **Deep Learning** : PyTorch 2.0+
- **Data Science** : NumPy, Pandas, scikit-learn
- **Visualisation** : Matplotlib, Seaborn
- **Environnement** : Google Colab (GPU), Jupyter
- **[Secondaire]** : Flask, ReportLab (génération PDF)

## Livrables

- [x] Code source structuré et documenté
- [ ] Notebooks reproductibles
- [ ] Rapport académique
- [ ] Présentation PowerPoint
- [ ] [Secondaire] Application Flask de démonstration

## Auteurs

- **Bachirou Konate**
- **Sylla Karamo**

## Licence

MIT License - voir [LICENSE](LICENSE)
