# vehicle-damage-detection
Ce projet vise à concevoir et implémenter un système de détection automatique des dommages sur les véhicules à partir d’images, en s’appuyant sur des techniques de vision par ordinateur et de deep learning. L’objectif technique est de concevoir, implémenter et comparer deux
architectures de réseaux de neurones convolutifs (CNN) entièrement développées from
scratch.


# Objectives
- Détecter des dommages visibles sur des images de véhicules (rayures, bosses, cassures).
- Implémenter from scratch et comparer deux architectures CNN distinctes f.
- Évaluer les performances des modèles à l’aide de métriques pertinentes.
- Mettre en place une chaîne complète de traitement des données et d’expérimentation.

# Structure du projet
   ```
├── data/
│ ├── raw/ # données brutes originales (non modifiées)
│ ├── processed/ # données après prétraitement (redimensionnement, normalisation, etc.)
│ └── README.md # description détaillée des datasets utilisés
├── notebooks/
│ ├── 01_pretraitement.ipynb # notebook de nettoyage et préparation des données
│ ├── 02_cnn_baseline.ipynb # entraînement et évaluation du CNN de base
│ └── 03_cnn_deep.ipynb # entraînement et évaluation du CNN plus profond
├── src/
│ ├── cnn_baseline.py # définition de l’architecture CNN baseline from scratch
│ ├── cnn_deep.py # définition de l’architecture CNN deep from scratch
│ └── utils.py # fonctions utilitaires communes (chargement, métriques, etc.)
├── README.md 
└── requirements.txt # liste des dépendances Python nécessaires
   ```

## Methodology
- Prétraitement des images : nettoyage, redimensionnement et normalisation.
- Conception et implémentation de deux architectures CNN from scratch.
- Entraînement des modèles sur les données préparées.
- Évaluation et comparaison des performances à l’aide de métriques adaptées.
- conception d'outils pour le rapport de véhicule 
- Conception et déployement de la solution finale

## Technologies
- Python
- NumPy
- Framework de Deep Learning (sans modèles pré-entraînés)
- Jupyter Notebook

## How to Run*
1. clonne the ripos
      ```bash
   git colonne https://github.com/IngenieurKonate/vehicle-damage-detection.git
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
