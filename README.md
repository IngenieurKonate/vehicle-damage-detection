# Vehicle Damage Detection

CNN from scratch pour la détection automatique de dommages sur véhicules.

## Description

Ce projet implémente et compare deux architectures CNN conçues from scratch pour la classification binaire de dommages sur véhicules (damaged vs undamaged).

- **Model A**: Baseline CNN inspiré de VGG (~6.5M paramètres)
- **Model B**: Deep CNN avec skip connections inspiré de ResNet (~11M paramètres)

## Structure du Projet

```
vehicle-damage-detection/
├── configs/                # Fichiers de configuration YAML
│   ├── config.yaml        # Configuration principale
│   ├── model_a_config.yaml
│   └── model_b_config.yaml
├── data/                   # Données (non versionnées)
│   ├── raw/               # Données brutes (CarDD, Stanford Cars)
│   └── processed/         # Données prétraitées (train/val/test)
├── notebooks/             # Notebooks Jupyter
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_baseline.ipynb
│   ├── 04_train_deep.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_analysis.ipynb
├── src/                   # Code source
│   ├── data/             # Gestion des données
│   ├── models/           # Architectures CNN
│   ├── training/         # Entraînement
│   ├── evaluation/       # Évaluation
│   └── utils/            # Utilitaires
├── app/                   # Application Flask (secondaire)
├── scripts/               # Scripts exécutables
├── checkpoints/           # Modèles sauvegardés
├── outputs/               # Résultats et figures
├── docs/                  # Documentation
├── presentation/          # Présentation PowerPoint
└── tests/                 # Tests unitaires
```

## Installation

```bash
# Cloner le repository
git clone https://github.com/IngenieurKonate/vehicle-damage-detection.git
cd vehicle-damage-detection

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Données

Le projet utilise deux datasets:
- **CarDD**: Véhicules endommagés (~4,000 images) - [USTC](https://cardd-ustc.github.io/)
- **Stanford Cars**: Véhicules non endommagés (~4,000 images échantillonnées) - [Kaggle 224x224](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-images-in-224x224)

Voir [data/README.md](data/README.md) pour les instructions de téléchargement.

## Usage

### Notebooks (Recommandé)
```bash
jupyter notebook notebooks/
```

### Scripts CLI
```bash
# Entraîner Model A (Baseline)
python scripts/train.py --model baseline --config configs/config.yaml

# Entraîner Model B (Deep CNN)
python scripts/train.py --model deep --config configs/config.yaml

# Évaluer un modèle
python scripts/evaluate.py --checkpoint checkpoints/model_b/best.pth

# Prédiction sur une image
python scripts/predict.py --image path/to/image.jpg
```

## Méthodologie

1. **Prétraitement**: Combinaison et équilibrage des datasets (50/50)
2. **Split**: 70% train / 15% val / 15% test (stratifié)
3. **Augmentation**: Flip, rotation, color jitter, crop
4. **Entraînement**: Adam optimizer, early stopping, checkpointing
5. **Évaluation**: Accuracy, Precision, Recall, F1-Score

## Objectifs de Performance

- Model A (Baseline): F1 ≥ 0.70
- Model B (Deep): ΔF1 ≥ +0.05 vs Model A

## Technologies

- Python 3.10+
- PyTorch 2.0+
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- Flask (application démo)

## Auteurs

- Bachirou Konate
- Sylla Karamo

## Licence

MIT License - voir [LICENSE](LICENSE)
