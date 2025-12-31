# 🚗 PRD — Détection Automatique de Dommages sur Véhicules par CNN From Scratch

> **Document de Référence pour l'Implémentation**  

---

## 📋 Table des Matières

1. [Vue d'Ensemble du Projet](#1-vue-densemble-du-projet)
2. [Contexte Académique et Contraintes](#2-contexte-académique-et-contraintes)
3. [Problématique et Hypothèses](#3-problématique-et-hypothèses)
4. [Spécifications des Données](#4-spécifications-des-données)
5. [Architecture Model A — Baseline VGG-like](#5-architecture-model-a--baseline-vgg-like)
6. [Architecture Model B — Deep CNN avec Skip Connections](#6-architecture-model-b--deep-cnn-avec-skip-connections)
7. [Pipeline d'Entraînement](#7-pipeline-dentraînement)
8. [Protocole d'Évaluation](#8-protocole-dévaluation)
9. [Structure du Projet](#9-structure-du-projet)
10. [Spécifications d'Implémentation](#10-spécifications-dimplémentation)
11. [Checklist de Validation](#11-checklist-de-validation)
12. [Glossaire Technique](#12-glossaire-technique)

---

## 1. Vue d'Ensemble du Projet

### 1.1 Résumé Exécutif

Ce projet vise à concevoir, implémenter et comparer **deux architectures CNN from scratch** pour la détection automatique de dommages visuels sur véhicules (rayures, bosses, fissures). L'objectif est de démontrer une **compréhension profonde** des concepts de Deep Learning à travers une démarche de conception originale.

### 1.2 Objectifs du Projet

| ID | Objectif | Priorité | Critère de Succès |
|----|----------|----------|-------------------|
| O1 | Concevoir une architecture baseline (VGG-like) from scratch | 🔴 Critique | Architecture fonctionnelle, F1 ≥ 0.70 |
| O2 | Concevoir une architecture deep avec skip connections | 🔴 Critique | Architecture fonctionnelle, ΔF1 ≥ +0.05 vs baseline |
| O3 | Comparer scientifiquement les deux architectures | 🔴 Critique | Analyse comparative documentée |
| O4 | Produire un rapport académique rigoureux | 🟠 Important | Justification de chaque choix architectural |
| O5 | Créer une présentation PowerPoint professionnelle | 🟠 Important | Slides clairs, visuels, défense des choix |
| O6 | Développer une application Flask de démonstration | 🟢 Secondaire | Interface web fonctionnelle pour prédiction |
| O7 | Implémenter la génération automatique de rapports | 🟢 Secondaire | PDF de diagnostic généré automatiquement |

### 1.3 Portée (Scope)

#### ✅ In-Scope

- Classification binaire : `damaged` vs `undamaged`
- Classification multi-classes (optionnel) : `scratch`, `dent`, `crack`, `shatter`, `undamaged`
- Deux architectures CNN conçues from scratch
- Pipeline complet : prétraitement → entraînement → évaluation
- Notebooks reproductibles et documentés
- Rapport académique et présentation PowerPoint
- **[Secondaire]** Application Flask de démonstration (upload image → prédiction)
- **[Secondaire]** Génération automatique de rapports PDF de diagnostic

#### ❌ Out-of-Scope

- Détection avec bounding boxes (YOLO-style) — hors périmètre initial
- Segmentation sémantique des dommages
- Déploiement en production cloud (API scalable, CI/CD)
- Comparaison avant/après location automatisée
- Transfer learning avec modèles pré-entraînés (interdit académiquement)

---

## 2. Contexte Académique et Contraintes

### 2.1 Exigences du Professeur

> **Citation clé du professeur :**  
> *"Minimum one model implemented by the members of the team from scratch. Take ideas like the VGG block or residual connection and build your own model."*

### 2.2 Ce qui est AUTORISÉ ✅

```python
# ✅ Utilisation de PyTorch/TensorFlow
import torch
import torch.nn as nn

# ✅ Couches de base
nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.Dropout

# ✅ Fonctions d'activation
nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Softmax

# ✅ Pooling
nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d

# ✅ Optimiseurs et Loss
torch.optim.Adam, torch.optim.SGD
nn.CrossEntropyLoss, nn.BCELoss

# ✅ Autograd
# Le calcul automatique des gradients est autorisé

# ✅ Data augmentation
torchvision.transforms.*
```

### 2.3 Ce qui est INTERDIT ❌

```python
# ❌ Import de modèles pré-définis
from torchvision.models import resnet18, vgg16, efficientnet_b0

# ❌ Modèles pré-entraînés
model = resnet18(pretrained=True)  # INTERDIT
model = resnet18(weights=None)     # INTERDIT aussi (architecture pas la nôtre)

# ❌ Hubs de modèles
torch.hub.load('pytorch/vision', 'resnet18')
timm.create_model('efficientnet_b0')
```

### 2.4 Critères de Notation (Implicites)

| Critère | Poids Estimé | Comment l'Atteindre |
|---------|--------------|---------------------|
| Compréhension architecturale | 30% | Justifier CHAQUE choix de couche |
| Originalité de conception | 25% | Architecture propre, pas copier-coller |
| Rigueur expérimentale | 20% | Protocole clair, résultats reproductibles |
| Qualité du code | 15% | Clean code, modulaire, documenté |
| Rapport final et présentation | 10% | Clarté, rigueur, qualité des visuels |

---

## 3. Problématique et Hypothèses

### 3.1 Problématique de Recherche

> **Question principale :**  
> Dans quelle mesure une architecture CNN conçue from scratch, s'inspirant des principes de VGG et ResNet, peut-elle détecter efficacement les dommages visuels sur véhicules ?

> **Questions secondaires :**
> 1. Quel est l'apport mesurable des connexions résiduelles sur cette tâche ?
> 2. Quelle profondeur de réseau est optimale pour ce problème spécifique ?
> 3. Comment la data augmentation influence-t-elle la généralisation ?

### 3.2 Hypothèses Expérimentales

| ID | Hypothèse | Variable Indépendante | Variable Dépendante | Validation |
|----|-----------|----------------------|---------------------|------------|
| H1 | Un CNN VGG-like de 6-8 couches convolutives peut atteindre F1 ≥ 0.70 sur la classification de dommages | Architecture (baseline) | F1-Score | Entraînement Model A |
| H2 | L'ajout de skip connections améliore le F1-Score d'au moins 5 points | Présence de skip connections | F1-Score | Comparaison A vs B |
| H3 | BatchNorm accélère la convergence et améliore la stabilité | Présence de BatchNorm | Loss convergence, variance | Ablation study |
| H4 | L'augmentation de données réduit l'écart train/val loss d'au moins 20% | Data augmentation | Généralisation gap | Comparaison avec/sans augmentation |

### 3.3 Contribution Scientifique Attendue

Ce projet ne vise pas à battre l'état de l'art mais à **démontrer** :

1. **Maîtrise conceptuelle** : comprendre pourquoi certaines architectures fonctionnent
2. **Capacité de conception** : créer une architecture adaptée au problème
3. **Rigueur expérimentale** : comparer objectivement deux approches
4. **Communication scientifique** : expliquer clairement des choix complexes

---

## 4. Spécifications des Données

### 4.1 Stratégie de Dataset : Le Duo Gagnant

Pour réaliser une **classification binaire** (damaged vs undamaged), nous combinons **deux datasets complémentaires** de référence académique.

#### Pourquoi deux datasets ?

Le dataset CarDD contient uniquement des images de véhicules endommagés. Pour entraîner un classificateur binaire, le modèle doit apprendre à distinguer les deux classes. Sans images de véhicules en bon état, le modèle prédirait systématiquement "damaged" (biais total).

```
DATASET COMBINÉ = CarDD (damaged) + Stanford Cars (undamaged)
                      ↓                      ↓
              Classe "DAMAGED"      Classe "UNDAMAGED"
                (4,000 images)       (4,000 images)
```

---

### 4.2 Dataset 1 : CarDD (Véhicules Endommagés)

#### Informations Générales

| Attribut | Valeur |
|----------|--------|
| **Nom complet** | Car Damage Detection Dataset (CarDD) |
| **Source** | USTC (University of Science and Technology of China) |
| **Publication** | IEEE Transactions on Intelligent Transportation Systems, 2023 |
| **Auteurs** | Wang, Xinkuang; Li, Wenjing; Wu, Zhongcheng |
| **Images** | 4,000 images haute résolution |
| **Instances annotées** | ~9,000 (plusieurs dommages par image) |
| **Résolution moyenne** | 684,231 pixels (~13.6× supérieure aux autres datasets) |
| **Taille totale** | ~5 GB (images + annotations + SOD) |
| **Format** | JPEG/PNG, RGB |

#### Structure du Dataset CarDD (téléchargé)

```
CarDD_release/
│
├── 📁 CarDD_COCO/                    # ✅ FORMAT COCO - À UTILISER
│   ├── 📁 annotations/               # ❌ Ignorer (fichiers JSON pour détection)
│   ├── 📁 train2017/                 # ✅ 2,816 images
│   ├── 📁 val2017/                   # ✅ 810 images
│   └── 📁 test2017/                  # ✅ ~374 images
│
└── 📁 CarDD_SOD/                     # ❌ IGNORER ENTIÈREMENT
    ├── 📁 CarDD-TE/                  # (Salient Object Detection - autre tâche)
    ├── 📁 CarDD-TR/
    └── 📁 CarDD-VAL/
```

#### Distribution des Images CarDD

| Split | Nombre d'images | Pourcentage |
|-------|-----------------|-------------|
| **train2017** | 2,816 | 70.4% |
| **val2017** | 810 | 20.25% |
| **test2017** | ~374 | 9.35% |
| **TOTAL** | **~4,000** | 100% |

#### Ce qu'on utilise vs ce qu'on ignore

| Élément | Taille estimée | Utilisation |
|---------|----------------|-------------|
| `CarDD_COCO/train2017/` | ~2 GB | ✅ **UTILISER** |
| `CarDD_COCO/val2017/` | ~600 MB | ✅ **UTILISER** |
| `CarDD_COCO/test2017/` | ~300 MB | ✅ **UTILISER** |
| `CarDD_COCO/annotations/` | ~50 MB | ❌ Ignorer (JSON pour YOLO/Mask R-CNN) |
| `CarDD_SOD/` | ~2 GB | ❌ Ignorer (autre tâche) |

> **Note** : Pour notre classification binaire, seules les **images** sont nécessaires. Les annotations COCO (bounding boxes, masques) et le dossier SOD ne sont pas utilisés car nous ne faisons pas de détection d'objets ni de segmentation.

#### Catégories de Dommages (6 classes)

| Catégorie | Traduction | Description |
|-----------|------------|-------------|
| `dent` | Bosse | Déformation du métal de carrosserie |
| `scratch` | Rayure | Dommage superficiel de la peinture |
| `crack` | Fissure | Fracture profonde du matériau |
| `glass shatter` | Vitre brisée | Pare-brise ou vitres cassés |
| `lamp broken` | Phare cassé | Optiques avant/arrière endommagées |
| `tire flat` | Pneu crevé | Pneumatique à plat |

#### Liens de Référence

| Ressource | URL |
|-----------|-----|
| **Site officiel** | https://cardd-ustc.github.io/ |
| **Paper ArXiv** | https://arxiv.org/abs/2211.00945 |
| **Paper IEEE** | https://ieeexplore.ieee.org/document/10078726 |
| **GitHub** | https://github.com/CarDD-USTC/CarDD-USTC.github.io |
| **Hugging Face** | https://huggingface.co/datasets/harpreetsahota/CarDD |

#### Citation BibTeX

```bibtex
@article{CarDD,
    author={Wang, Xinkuang and Li, Wenjing and Wu, Zhongcheng},
    journal={IEEE Transactions on Intelligent Transportation Systems},
    title={CarDD: A New Dataset for Vision-Based Car Damage Detection},
    year={2023},
    volume={24},
    number={7},
    pages={7202-7214},
    doi={10.1109/TITS.2023.3258480}
}
```

#### Utilisation dans notre projet

- **Rôle** : Fournir la classe `DAMAGED`
- **Sélection** : Toutes les 4,000 images (indépendamment du type de dommage)
- **Label assigné** : `1` (damaged)

---

### 4.3 Dataset 2 : Stanford Cars (Véhicules Non Endommagés)

#### Informations Générales

| Attribut | Valeur |
|----------|--------|
| **Nom complet** | Stanford Cars Dataset |
| **Source** | Stanford AI Lab (Stanford University) |
| **Publication** | 3D Object Representations for Fine-Grained Categorization, 2013 |
| **Auteurs** | Krause, Jonathan; Stark, Michael; Deng, Jia; Fei-Fei, Li |
| **Images totales** | 16,185 images |
| **Classes originales** | 196 (marques/modèles : Tesla Model S 2012, BMW M3 coupe, etc.) |
| **Split original** | 8,144 train / 8,041 test |
| **Taille** | ~2 GB |
| **Format** | JPEG, RGB |

#### Structure du Dataset Stanford Cars (à télécharger)

```
stanford_cars/
│
├── 📁 cars_train/                    # ✅ À UTILISER - 8,144 images
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
│
├── 📁 cars_test/                     # ✅ À UTILISER - 8,041 images
│   ├── 00001.jpg
│   └── ...
│
├── 📄 cars_train_annos.mat           # ❌ Ignorer (labels marques/modèles)
├── 📄 cars_test_annos.mat            # ❌ Ignorer (labels marques/modèles)
├── 📄 cars_meta.mat                  # ❌ Ignorer (métadonnées des 196 classes)
└── 📄 devkit/                        # ❌ Ignorer (outils de développement)
```

#### Distribution des Images Stanford Cars

| Dossier | Nombre d'images | Utilisation |
|---------|-----------------|-------------|
| **cars_train/** | 8,144 | ✅ Source pour échantillonnage |
| **cars_test/** | 8,041 | ✅ Source pour échantillonnage |
| **TOTAL disponible** | **16,185** | Pool total |
| **TOTAL à utiliser** | **4,000** | Échantillon aléatoire (seed=42) |

#### Ce qu'on utilise vs ce qu'on ignore

| Élément | Taille estimée | Utilisation |
|---------|----------------|-------------|
| `cars_train/` | ~1 GB | ✅ **UTILISER** (images uniquement) |
| `cars_test/` | ~1 GB | ✅ **UTILISER** (images uniquement) |
| `*.mat files` | ~10 MB | ❌ Ignorer (annotations marques/modèles) |
| `devkit/` | ~1 MB | ❌ Ignorer (scripts MATLAB) |

> **Note** : Les fichiers `.mat` contiennent les labels des 196 classes (marques et modèles de voitures). Pour notre projet, nous ignorons ces labels car **toutes les images Stanford Cars = classe "undamaged"**. Nous échantillonnons aléatoirement 4,000 images pour équilibrer avec CarDD.

#### Processus d'échantillonnage

```python
# Pseudo-code pour l'échantillonnage
import random

# Charger toutes les images Stanford
all_stanford = list(cars_train/*.jpg) + list(cars_test/*.jpg)  # 16,185 images

# Échantillonner 4,000 pour équilibrer avec CarDD
random.seed(42)  # Reproductibilité
undamaged_images = random.sample(all_stanford, k=4000)

# Toutes labellisées "undamaged"
```

#### Liens de Référence

| Ressource | URL |
|-----------|-----|
| **Site officiel** | https://ai.stanford.edu/~jkrause/cars/car_dataset.html |
| **Kaggle (par classes)** | https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder |
| **Kaggle (full)** | https://www.kaggle.com/datasets/hassiahk/stanford-cars-dataset-full |
| **⭐ Kaggle (224×224)** | https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-images-in-224x224 |
| **TensorFlow Datasets** | https://www.tensorflow.org/datasets/catalog/cars196 |

#### ⭐ Source Recommandée : Kaggle 224×224

> **Télécharger depuis** : https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-images-in-224x224

**Pourquoi cette version ?**

| Raison | Explication |
|--------|-------------|
| **Taille optimale** | Images déjà redimensionnées en 224×224 pixels — exactement la taille d'entrée de nos CNNs |
| **Gain de temps** | Évite le preprocessing de ~16,000 images (redimensionnement coûteux en temps) |
| **Cohérence garantie** | Toutes les images ont strictement la même dimension, pas de surprises |
| **Fichier plus léger** | Téléchargement plus rapide que la version full (~500 MB vs ~2 GB) |
| **Compatibilité PyTorch** | Prêt à être chargé directement dans un DataLoader sans transformation de resize |

**Note** : La version "par classes" organise les images en 196 sous-dossiers (un par marque/modèle), ce qui est inutile pour nous car nous ignorons les marques — toutes les images deviennent simplement "undamaged".

#### Citation BibTeX

```bibtex
@inproceedings{KrauseStarkDengFei-Fei_3DRR2013,
    title={3D Object Representations for Fine-Grained Categorization},
    booktitle={4th International IEEE Workshop on 3D Representation and Recognition (3dRR-13)},
    year={2013},
    address={Sydney, Australia},
    author={Jonathan Krause and Michael Stark and Jia Deng and Li Fei-Fei}
}
```

#### Utilisation dans notre projet

- **Rôle** : Fournir la classe `UNDAMAGED`
- **Sélection** : Échantillon aléatoire de 4,000 images (sur 16,185)
- **Label assigné** : `0` (undamaged)
- **Raison de l'échantillonnage** : Équilibrer les classes (50/50)

---

### 4.4 Dataset Combiné Final

#### Vue d'ensemble

| Métrique | Valeur |
|----------|--------|
| **Total images** | 8,000 |
| **Classe `damaged`** | 4,000 (100% de CarDD) |
| **Classe `undamaged`** | 4,000 (échantillon de Stanford Cars) |
| **Ratio des classes** | 50% / 50% (équilibré) |
| **Taille estimée** | ~7 GB |

#### Justification du Choix

| Critère | Évaluation |
|---------|------------|
| **Qualité académique** | ✅ Deux datasets publiés et reconnus internationalement |
| **Équilibre des classes** | ✅ 50/50 évite le biais de classification |
| **Haute résolution** | ✅ Les deux datasets offrent des images de qualité |
| **Diversité** | ✅ Variété de marques, modèles, angles, conditions |
| **Reproductibilité** | ✅ Datasets publics avec liens stables |

---

### 4.5 Structure des Données (Google Drive)

> **Important** : Les données sont stockées sur Google Drive pour être accessibles depuis Google Colab. Le code source reste en local (VS Code).

#### Architecture Hybride : Code Local + Données Cloud

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ARCHITECTURE DU PROJET                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   💻 LOCAL (VS Code)                   ☁️ GOOGLE DRIVE                  │
│   ──────────────────                   ─────────────────                │
│                                                                         │
│   vehicle-damage-detection/            My Drive/ENSA_Deep_Learning/     │
│   ├── src/                             ├── datasets/                    │
│   ├── notebooks/                       │   ├── raw/                     │
│   ├── configs/                         │   └── processed/               │
│   ├── scripts/                         ├── checkpoints/                 │
│   └── ...                              └── outputs/                     │
│                                                                         │
│   ✅ Code versionné (Git)              ✅ Données persistantes          │
│   ✅ Exécuté sur Colab                 ✅ Accessibles depuis Colab      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Structure Google Drive Complète

```
📁 My Drive/
│
└── 📁 ENSA_Deep_Learning/                        # Dossier projet principal
    │
    ├── 📁 datasets/                              # Toutes les données
    │   │
    │   ├── 📁 raw/                               # Données brutes téléchargées
    │   │   │
    │   │   ├── 📁 CarDD_release/                 # Dataset CarDD (~5 GB)
    │   │   │   ├── 📁 CarDD_COCO/                # ✅ FORMAT À UTILISER
    │   │   │   │   ├── 📁 annotations/           # ❌ Ignorer
    │   │   │   │   ├── 📁 train2017/             # ✅ 2,816 images → damaged
    │   │   │   │   ├── 📁 val2017/               # ✅ 810 images → damaged
    │   │   │   │   └── 📁 test2017/              # ✅ ~374 images → damaged
    │   │   │   │
    │   │   │   └── 📁 CarDD_SOD/                 # ❌ IGNORER ENTIÈREMENT
    │   │   │
    │   │   └── 📁 stanford_cars_224/             # Dataset Stanford (~500 MB)
    │   │       └── 📁 car_data/
    │   │           ├── 📁 train/                 # ~8,144 images (196 sous-dossiers)
    │   │           └── 📁 test/                  # ~8,041 images (196 sous-dossiers)
    │   │
    │   └── 📁 processed/                         # Dataset combiné (généré par script)
    │       ├── 📁 train/                         # 70% = 5,600 images
    │       │   ├── 📁 damaged/                   # 2,800 images
    │       │   └── 📁 undamaged/                 # 2,800 images
    │       │
    │       ├── 📁 val/                           # 15% = 1,200 images
    │       │   ├── 📁 damaged/                   # 600 images
    │       │   └── 📁 undamaged/                 # 600 images
    │       │
    │       └── 📁 test/                          # 15% = 1,200 images
    │           ├── 📁 damaged/                   # 600 images
    │           └── 📁 undamaged/                 # 600 images
    │
    ├── 📁 checkpoints/                           # Modèles sauvegardés (persistants)
    │   ├── 📁 model_a/                           # Checkpoints Model A (VGG-like)
    │   └── 📁 model_b/                           # Checkpoints Model B (Skip connections)
    │
    └── 📁 outputs/                               # Résultats et logs
        ├── 📁 figures/                           # Graphiques, courbes d'apprentissage
        └── 📁 logs/                              # TensorBoard logs
```

#### Chemins d'Accès depuis Colab

| Ressource | Chemin Colab |
|-----------|--------------|
| **Racine Drive** | `/content/drive/MyDrive/` |
| **Projet** | `/content/drive/MyDrive/ENSA_Deep_Learning/` |
| **Datasets raw** | `/content/drive/MyDrive/ENSA_Deep_Learning/datasets/raw/` |
| **Datasets processed** | `/content/drive/MyDrive/ENSA_Deep_Learning/datasets/processed/` |
| **CarDD images** | `/content/drive/MyDrive/ENSA_Deep_Learning/datasets/raw/CarDD_release/CarDD_COCO/` |
| **Stanford images** | `/content/drive/MyDrive/ENSA_Deep_Learning/datasets/raw/stanford_cars_224/car_data/` |
| **Checkpoints** | `/content/drive/MyDrive/ENSA_Deep_Learning/checkpoints/` |
| **Outputs** | `/content/drive/MyDrive/ENSA_Deep_Learning/outputs/` |

#### Note sur le Preprocessing

Le script de préparation des données (exécuté dans Colab) devra :
1. **Monter** Google Drive avec `drive.mount('/content/drive')`
2. **Collecter** les images de `CarDD_COCO/train2017/`, `val2017/`, `test2017/` → toutes = `damaged`
3. **Collecter** les images de `stanford_cars_224/car_data/train/` et `test/` (tous les sous-dossiers) → toutes = `undamaged`
4. **Échantillonner** 4,000 images de Stanford pour équilibrer avec CarDD (seed=42)
5. **Mélanger** et **splitter** en 70/15/15 de manière stratifiée
6. **Copier** les images dans la structure `datasets/processed/`

---

### 4.6 Prétraitement des Images

```python
# Configuration du prétraitement
PREPROCESSING_CONFIG = {
    "input_size": (224, 224),           # Taille d'entrée standard CNN
    "normalization": {
        "mean": [0.485, 0.456, 0.406],  # Statistiques ImageNet (référence)
        "std": [0.229, 0.224, 0.225]
    },
    "color_space": "RGB"
}
```

#### Pipeline de Prétraitement

1. **Chargement** : Lecture de l'image (PIL ou OpenCV)
2. **Redimensionnement** : Resize to 224×224 pixels
3. **Normalisation** : Scale [0, 255] → [0, 1] puis normalisation mean/std
4. **Conversion** : PIL Image → Tensor PyTorch (C, H, W)

---

### 4.7 Augmentation des Données

```python
# Configuration d'augmentation pour l'entraînement
TRAIN_AUGMENTATION = {
    "RandomHorizontalFlip": {"p": 0.5},
    "RandomRotation": {"degrees": 15},
    "ColorJitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.1,
        "hue": 0.05
    },
    "RandomResizedCrop": {
        "size": 224,
        "scale": (0.8, 1.0)
    }
}

# Pas d'augmentation pour validation/test
VAL_TEST_AUGMENTATION = None
```

#### Justification des Augmentations

| Augmentation | Justification |
|--------------|---------------|
| `HorizontalFlip` | Les dommages peuvent apparaître à gauche ou à droite du véhicule |
| `Rotation (±15°)` | Simule les différents angles de prise de vue lors de l'inspection |
| `ColorJitter` | Compense les variations d'éclairage (intérieur, extérieur, nuit) |
| `RandomResizedCrop` | Simule différentes distances entre la caméra et le véhicule |

---

### 4.8 Split des Données

```python
DATA_SPLIT = {
    "train": 0.70,      # 70% pour l'entraînement = 5,600 images
    "val": 0.15,        # 15% pour la validation = 1,200 images
    "test": 0.15,       # 15% pour le test final = 1,200 images
    "random_seed": 42,  # Pour reproductibilité
    "stratified": True  # Maintenir le ratio 50/50 dans chaque split
}
```

#### Distribution Finale

| Split | Total | Damaged | Undamaged |
|-------|-------|---------|-----------|
| **Train** | 5,600 | 2,800 | 2,800 |
| **Validation** | 1,200 | 600 | 600 |
| **Test** | 1,200 | 600 | 600 |
| **TOTAL** | **8,000** | **4,000** | **4,000** |

---

## 5. Architecture Model A — Baseline VGG-like

### 5.1 Philosophie de Conception

> **Principe VGG** : Utiliser des convolutions 3×3 empilées plutôt que de grands kernels.  
> **Avantage** : Même champ réceptif avec moins de paramètres et plus de non-linéarités.

**Pourquoi VGG-like pour la baseline ?**
- Architecture simple et bien comprise
- Facile à implémenter et débugger
- Bon point de référence pour mesurer l'apport des skip connections

### 5.2 Spécifications Architecturales

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL A - BASELINE CNN                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: (batch, 3, 224, 224)                               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ BLOCK 1                                              │   │
│  │ Conv2d(3→32, k=3, p=1) → ReLU                       │   │
│  │ Conv2d(32→32, k=3, p=1) → ReLU                      │   │
│  │ MaxPool2d(2, 2)                                      │   │
│  │ Output: (batch, 32, 112, 112)                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ BLOCK 2                                              │   │
│  │ Conv2d(32→64, k=3, p=1) → ReLU                      │   │
│  │ Conv2d(64→64, k=3, p=1) → ReLU                      │   │
│  │ MaxPool2d(2, 2)                                      │   │
│  │ Output: (batch, 64, 56, 56)                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ BLOCK 3                                              │   │
│  │ Conv2d(64→128, k=3, p=1) → ReLU                     │   │
│  │ Conv2d(128→128, k=3, p=1) → ReLU                    │   │
│  │ MaxPool2d(2, 2)                                      │   │
│  │ Output: (batch, 128, 28, 28)                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ BLOCK 4                                              │   │
│  │ Conv2d(128→256, k=3, p=1) → ReLU                    │   │
│  │ Conv2d(256→256, k=3, p=1) → ReLU                    │   │
│  │ MaxPool2d(2, 2)                                      │   │
│  │ Output: (batch, 256, 14, 14)                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ BLOCK 5                                              │   │
│  │ Conv2d(256→512, k=3, p=1) → ReLU                    │   │
│  │ Conv2d(512→512, k=3, p=1) → ReLU                    │   │
│  │ MaxPool2d(2, 2)                                      │   │
│  │ Output: (batch, 512, 7, 7)                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ CLASSIFIER                                           │   │
│  │ AdaptiveAvgPool2d(1, 1) → Flatten                   │   │
│  │ Linear(512→256) → ReLU → Dropout(0.5)               │   │
│  │ Linear(256→num_classes)                              │   │
│  │ Output: (batch, num_classes)                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Code de Référence

```python
import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    """
    Bloc VGG-style : deux convolutions 3×3 suivies de MaxPool.
    
    Justification architecturale:
    - Deux conv 3×3 = champ réceptif équivalent à une conv 5×5
    - Mais avec moins de paramètres (2×3²×C² vs 5²×C²)
    - Et une non-linéarité supplémentaire (meilleure capacité d'apprentissage)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x


class BaselineCNN(nn.Module):
    """
    Architecture CNN Baseline inspirée de VGG.
    
    Caractéristiques:
    - 5 blocs convolutifs avec doublement progressif des canaux
    - Adaptive pooling pour flexibilité de taille d'entrée
    - Classifier avec dropout pour régularisation
    
    Paramètres totaux estimés: ~6.5M
    """
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            VGGBlock(3, 32),      # 224→112, 3→32 canaux
            VGGBlock(32, 64),     # 112→56, 32→64 canaux
            VGGBlock(64, 128),    # 56→28, 64→128 canaux
            VGGBlock(128, 256),   # 28→14, 128→256 canaux
            VGGBlock(256, 512),   # 14→7, 256→512 canaux
        )
        
        # Global pooling + Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 7×7→1×1
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_num_parameters(self) -> int:
        """Retourne le nombre total de paramètres."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Retourne le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### 5.4 Justification des Choix Architecturaux

| Choix | Pourquoi | Alternative Considérée |
|-------|----------|------------------------|
| Convolutions 3×3 | Petit kernel mais champ réceptif efficace via empilement | 5×5 ou 7×7 (plus de paramètres) |
| Doublement des canaux | Convention standard, capture features de plus en plus abstraites | Croissance linéaire |
| MaxPool 2×2 | Réduction spatiale efficace, invariance locale | AvgPool (moins discriminant) |
| AdaptiveAvgPool | Flexibilité taille d'entrée, réduit overfitting vs FC large | Flatten direct (trop de params) |
| Dropout 0.5 | Régularisation standard pour classifier | Dropout plus faible (moins efficace) |
| ReLU | Simple, efficace, pas de vanishing gradient | LeakyReLU (pas nécessaire ici) |

### 5.5 Analyse du Champ Réceptif

```
Couche          Champ Réceptif    Explication
──────────────────────────────────────────────
Input           1×1               Pixel initial
Block1-Conv1    3×3               Premier kernel
Block1-Conv2    5×5               3 + (3-1) = 5
Block1-Pool     6×6               5 + 1 = 6 (stride 2)
Block2-Conv1    10×10             6×2 + (3-1) = 14? Non: (6-1)×2 + 3
...
Block5-Pool     ~180×180          Couvre une grande partie de l'image 224×224
```

**Conclusion** : Le champ réceptif final permet de capturer des patterns à l'échelle de dommages typiques (quelques cm sur une voiture ≈ 50-150 pixels sur une image 224×224).

---

## 6. Architecture Model B — Deep CNN avec Skip Connections

### 6.1 Philosophie de Conception

> **Principe ResNet** : Les connexions résiduelles permettent d'entraîner des réseaux plus profonds en facilitant le flux de gradients.

**Formulation mathématique** :
```
Output = F(x) + x       (skip connection)
```
Au lieu d'apprendre `H(x)`, le réseau apprend `F(x) = H(x) - x` (le résidu).

**Pourquoi les skip connections ?**
- Atténuent le problème de vanishing gradient
- Permettent l'entraînement de réseaux plus profonds
- L'identité est facile à apprendre si nécessaire (F(x) → 0)

### 6.2 Spécifications Architecturales

```
┌─────────────────────────────────────────────────────────────┐
│                MODEL B - DEEP CNN WITH SKIP CONNECTIONS      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: (batch, 3, 224, 224)                               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ STEM                                                 │   │
│  │ Conv2d(3→64, k=7, s=2, p=3) → BN → ReLU             │   │
│  │ MaxPool2d(3, 2, 1)                                   │   │
│  │ Output: (batch, 64, 56, 56)                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ STAGE 1: 2× ResidualBlock(64→64)                    │   │
│  │ ┌───────────────────────────────────────────────┐   │   │
│  │ │  x ──→ Conv→BN→ReLU→Conv→BN ──→ (+) → ReLU   │   │   │
│  │ │  └──────────────────────────────↗             │   │   │
│  │ └───────────────────────────────────────────────┘   │   │
│  │ Output: (batch, 64, 56, 56)                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ STAGE 2: 2× ResidualBlock(64→128), stride=2 first   │   │
│  │ Output: (batch, 128, 28, 28)                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ STAGE 3: 2× ResidualBlock(128→256), stride=2 first  │   │
│  │ Output: (batch, 256, 14, 14)                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ STAGE 4: 2× ResidualBlock(256→512), stride=2 first  │   │
│  │ Output: (batch, 512, 7, 7)                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ HEAD                                                 │   │
│  │ AdaptiveAvgPool2d(1, 1) → Flatten                   │   │
│  │ Linear(512→num_classes)                              │   │
│  │ Output: (batch, num_classes)                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Code de Référence

```python
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Bloc résiduel basique avec skip connection.
    
    Architecture:
        x → Conv → BN → ReLU → Conv → BN → (+) → ReLU
        └─────────────────────────────────↗
    
    Si downsample=True ou changement de canaux:
        La skip connection passe par une conv 1×1 pour matcher les dimensions.
    
    Justification:
    - Skip connection permet au gradient de "bypass" les convolutions
    - BatchNorm stabilise l'entraînement et accélère la convergence
    - Placement BN après Conv (style original ResNet)
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        downsample: nn.Module = None
    ):
        super().__init__()
        
        # Première convolution (peut réduire la taille spatiale)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Deuxième convolution (maintient la taille)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Branche principale
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection (avec projection si nécessaire)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Addition et activation finale
        out += identity
        out = self.relu(out)
        
        return out


class DeepCNN(nn.Module):
    """
    Architecture Deep CNN avec skip connections inspirée de ResNet.
    
    Caractéristiques:
    - Stem agressif (conv 7×7 stride 2 + maxpool) pour réduction rapide
    - 4 stages avec blocs résiduels
    - Global Average Pooling pour réduire l'overfitting
    - Classifier minimaliste (une seule couche FC)
    
    Différences clés avec Model A:
    - Skip connections pour meilleur flux de gradient
    - BatchNorm pour stabilité
    - Plus profond (18 couches conv vs 10)
    - Moins de paramètres dans le classifier
    
    Paramètres totaux estimés: ~11M
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Stem: réduction rapide de la résolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stages de blocs résiduels
        self.stage1 = self._make_stage(64, 64, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _make_stage(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_blocks: int, 
        stride: int
    ) -> nn.Sequential:
        """
        Crée un stage composé de plusieurs blocs résiduels.
        
        Le premier bloc peut avoir un stride > 1 pour downsampling.
        Les blocs suivants maintiennent la résolution.
        """
        downsample = None
        
        # Projection si changement de dimensions
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        
        # Premier bloc (peut downsample)
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        # Blocs suivants (maintiennent la résolution)
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialisation des poids selon les bonnes pratiques.
        
        - Conv: Kaiming He (adapté pour ReLU)
        - BatchNorm: weight=1, bias=0
        - Linear: Normal avec petit std
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)
        
        # Stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Retourne le nombre total de paramètres."""
        return sum(p.numel() for p in self.parameters())
```

### 6.4 Justification des Choix Architecturaux

| Choix | Pourquoi | Impact Attendu |
|-------|----------|----------------|
| Skip connections | Flux de gradient amélioré, entraînement stable | Convergence plus rapide, possibilité d'aller plus profond |
| BatchNorm | Normalisation des activations, régularisation implicite | Stabilité, accélération |
| Stem 7×7 stride 2 | Réduction rapide de la résolution dès le début | Moins de compute dans les stages suivants |
| Conv 1×1 pour projection | Matcher les dimensions avec minimum de paramètres | Skip connection fonctionnelle même avec changement de taille |
| Global Average Pool | Réduction drastique des paramètres | Moins d'overfitting que FC large |
| Pas de Dropout | BatchNorm fournit déjà une régularisation | Simplicité |

### 6.5 Comparaison Model A vs Model B

| Aspect | Model A (Baseline) | Model B (Deep) |
|--------|-------------------|----------------|
| Profondeur (couches conv) | 10 | 18 |
| Skip connections | ❌ Non | ✅ Oui |
| BatchNorm | ❌ Non | ✅ Oui |
| Params (estimés) | ~6.5M | ~11M |
| Régularisation | Dropout 0.5 | BatchNorm |
| Complexité | Simple | Modérée |
| Risque vanishing gradient | Moyen | Faible |

---

## 7. Pipeline d'Entraînement

### 7.1 Configuration Générale

```python
TRAINING_CONFIG = {
    # Hyperparamètres de base
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    
    # Optimiseur
    "optimizer": "Adam",
    "optimizer_params": {
        "betas": (0.9, 0.999),
        "eps": 1e-8
    },
    
    # Scheduler
    "scheduler": "ReduceLROnPlateau",
    "scheduler_params": {
        "mode": "min",
        "factor": 0.1,
        "patience": 5,
        "min_lr": 1e-6
    },
    
    # Early stopping
    "early_stopping": {
        "patience": 10,
        "min_delta": 1e-4,
        "monitor": "val_loss"
    },
    
    # Checkpointing
    "save_best_only": True,
    "checkpoint_dir": "checkpoints/",
    
    # Reproductibilité
    "random_seed": 42,
    "deterministic": True
}
```

### 7.2 Fonction de Perte

```python
# Pour classification binaire
criterion = nn.CrossEntropyLoss()

# Alternative si classes déséquilibrées
class_weights = torch.tensor([1.0, 2.0])  # Exemple: 2× poids pour 'damaged'
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Justification** : CrossEntropyLoss combine LogSoftmax et NLLLoss, adapté à la classification multi-classes (même binaire avec 2 classes).

### 7.3 Boucle d'Entraînement (Pseudo-code)

```python
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Métriques
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_targets
```

### 7.4 Early Stopping

```python
class EarlyStopping:
    """
    Arrête l'entraînement si la métrique ne s'améliore pas.
    
    Justification:
    - Évite l'overfitting en stoppant au bon moment
    - Économise du temps de calcul
    - Sélectionne automatiquement le meilleur modèle
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop
```

---

## 8. Protocole d'Évaluation

### 8.1 Métriques Principales

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **Accuracy** | (TP + TN) / Total | Performance globale |
| **Precision** | TP / (TP + FP) | "Quand je prédis damaged, ai-je raison ?" |
| **Recall** | TP / (TP + FN) | "Est-ce que je détecte tous les dommages ?" |
| **F1-Score** | 2 × (P × R) / (P + R) | Compromis precision/recall |

**Où** : TP = True Positive (damaged prédit et réel), FP = False Positive, etc.

### 8.2 Code d'Évaluation

```python
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, class_names=['undamaged', 'damaged']):
    """
    Évaluation complète d'un modèle de classification.
    
    Retourne:
    - Dictionnaire de métriques
    - Matrice de confusion
    - Rapport de classification
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'precision_per_class': precision_score(y_true, y_pred, average=None),
        'recall_per_class': recall_score(y_true, y_pred, average=None),
        'f1_per_class': f1_score(y_true, y_pred, average=None)
    }
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    return metrics, cm, report


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Visualise la matrice de confusion."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()


def plot_training_history(history: dict, save_path: str = None):
    """
    Visualise l'historique d'entraînement.
    
    Args:
        history: {'train_loss': [...], 'val_loss': [...], 
                  'train_acc': [...], 'val_acc': [...]}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
```

### 8.3 Comparaison des Modèles

```python
def compare_models(results_a: dict, results_b: dict):
    """
    Compare les performances de Model A et Model B.
    
    Args:
        results_a: Métriques du Model A
        results_b: Métriques du Model B
    
    Returns:
        DataFrame de comparaison
    """
    import pandas as pd
    
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Model A (Baseline)': [
            results_a['accuracy'],
            results_a['precision'],
            results_a['recall'],
            results_a['f1_score']
        ],
        'Model B (Deep)': [
            results_b['accuracy'],
            results_b['precision'],
            results_b['recall'],
            results_b['f1_score']
        ]
    })
    
    comparison['Δ (B - A)'] = comparison['Model B (Deep)'] - comparison['Model A (Baseline)']
    comparison['Improvement (%)'] = (comparison['Δ (B - A)'] / comparison['Model A (Baseline)']) * 100
    
    return comparison
```

### 8.4 Ablation Studies (Optionnel mais Recommandé)

| Expérience | Variable Modifiée | Objectif |
|------------|-------------------|----------|
| Ablation 1 | Model B sans BatchNorm | Mesurer l'apport de BatchNorm |
| Ablation 2 | Model B sans skip connections | Vérifier que les skips aident |
| Ablation 3 | Model A avec BatchNorm | BatchNorm aide-t-il même sans skips ? |
| Ablation 4 | Sans data augmentation | Mesurer l'apport de l'augmentation |

---

## 9. Structure du Projet

### 9.1 Arborescence Recommandée (Code Local)

> **Note** : Les données, checkpoints et outputs sont stockés sur Google Drive (voir section 4.5). Le code source est versionné localement avec Git.

```
vehicle-damage-detection/                # 💻 LOCAL (VS Code + Git)
│
├── README.md                            # Documentation principale
├── PRD.md                               # Ce document
├── requirements.txt                     # Dépendances Python (pour référence)
├── LICENSE                              # Licence MIT
├── .gitignore                           # Fichiers à ignorer
│
├── configs/                             # Fichiers de configuration
│   ├── config.yaml                      # Configuration principale
│   ├── model_a_config.yaml              # Config spécifique Model A
│   └── model_b_config.yaml              # Config spécifique Model B
│
├── notebooks/                           # Notebooks Jupyter (exécutés sur Colab)
│   ├── 00_setup_colab.ipynb             # Setup initial Colab + vérification GPU
│   ├── 01_data_exploration.ipynb        # Exploration des données
│   ├── 02_preprocessing.ipynb           # Création du dataset processed/
│   ├── 03_train_baseline.ipynb          # Entraînement Model A
│   ├── 04_train_deep.ipynb              # Entraînement Model B
│   ├── 05_evaluation.ipynb              # Évaluation et comparaison
│   └── 06_analysis.ipynb                # Analyse des erreurs
│
├── src/                                 # Code source (importé dans notebooks)
│   ├── __init__.py
│   │
│   ├── data/                            # Gestion des données
│   │   ├── __init__.py
│   │   ├── dataset.py                   # Classes Dataset PyTorch
│   │   ├── transforms.py                # Transformations et augmentations
│   │   └── utils.py                     # Utilitaires data
│   │
│   ├── models/                          # Architectures CNN
│   │   ├── __init__.py
│   │   ├── baseline_cnn.py              # Model A (VGG-like)
│   │   ├── deep_cnn.py                  # Model B (Skip connections)
│   │   └── components.py                # Blocs réutilisables (VGGBlock, ResidualBlock)
│   │
│   ├── training/                        # Entraînement
│   │   ├── __init__.py
│   │   ├── trainer.py                   # Classe Trainer
│   │   ├── callbacks.py                 # Early stopping, checkpointing
│   │   └── losses.py                    # Fonctions de perte custom
│   │
│   ├── evaluation/                      # Évaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                   # Calcul des métriques
│   │   └── visualization.py             # Graphiques et plots
│   │
│   └── utils/                           # Utilitaires généraux
│       ├── __init__.py
│       ├── config.py                    # Chargement config YAML
│       ├── seed.py                      # Reproductibilité
│       ├── paths.py                     # Chemins Google Drive (NEW)
│       └── logging.py                   # Logging
│
├── app/                                 # [SECONDAIRE] Application Flask
│   ├── __init__.py
│   ├── app.py                           # Application Flask principale
│   ├── templates/                       # Templates HTML
│   │   ├── index.html
│   │   └── result.html
│   ├── static/
│   │   └── style.css
│   └── utils/
│       ├── predictor.py
│       └── report_generator.py
│
├── scripts/                             # Scripts exécutables
│   ├── prepare_data.py                  # Script préparation données
│   ├── train.py                         # Script d'entraînement
│   ├── evaluate.py                      # Script d'évaluation
│   └── predict.py                       # Script d'inférence
│
├── docs/                                # Documentation additionnelle
│   ├── architecture.md
│   ├── evaluation_protocol.md
│   └── report_template.md
│
├── presentation/                        # Présentation PowerPoint
│   └── slides.pptx
│
└── tests/                               # Tests unitaires (optionnel)
    ├── test_models.py
    └── test_data.py
```

### 9.1.1 Structure Google Drive (Rappel)

```
📁 My Drive/ENSA_Deep_Learning/          # ☁️ GOOGLE DRIVE
│
├── 📁 datasets/
│   ├── 📁 raw/                          # CarDD + Stanford Cars
│   └── 📁 processed/                    # Dataset combiné train/val/test
│
├── 📁 checkpoints/
│   ├── 📁 model_a/                      # Sauvegardes Model A
│   └── 📁 model_b/                      # Sauvegardes Model B
│
└── 📁 outputs/
    ├── 📁 figures/                      # Graphiques générés
    └── 📁 logs/                         # TensorBoard logs
```

### 9.1.2 Fichier paths.py (Chemins centralisés)

```python
# src/utils/paths.py
"""
Chemins centralisés pour Google Drive.
À importer dans tous les notebooks et scripts.
"""

# Racine Google Drive (après mount)
DRIVE_ROOT = "/content/drive/MyDrive"

# Projet
PROJECT_ROOT = f"{DRIVE_ROOT}/ENSA_Deep_Learning"

# Datasets
DATASETS_DIR = f"{PROJECT_ROOT}/datasets"
RAW_DATA_DIR = f"{DATASETS_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATASETS_DIR}/processed"

# Datasets bruts
CARDD_DIR = f"{RAW_DATA_DIR}/CarDD_release/CarDD_COCO"
STANFORD_DIR = f"{RAW_DATA_DIR}/stanford_cars_224/car_data"

# Splits processed
TRAIN_DIR = f"{PROCESSED_DATA_DIR}/train"
VAL_DIR = f"{PROCESSED_DATA_DIR}/val"
TEST_DIR = f"{PROCESSED_DATA_DIR}/test"

# Checkpoints
CHECKPOINTS_DIR = f"{PROJECT_ROOT}/checkpoints"
MODEL_A_CKPT = f"{CHECKPOINTS_DIR}/model_a"
MODEL_B_CKPT = f"{CHECKPOINTS_DIR}/model_b"

# Outputs
OUTPUTS_DIR = f"{PROJECT_ROOT}/outputs"
FIGURES_DIR = f"{OUTPUTS_DIR}/figures"
LOGS_DIR = f"{OUTPUTS_DIR}/logs"
```

### 9.2 requirements.txt

```
# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0

# Data & Preprocessing
Pillow>=9.5.0
scikit-learn>=1.3.0
albumentations>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Configuration
pyyaml>=6.0
python-dotenv>=1.0.0

# Notebooks
jupyter>=1.0.0
ipywidgets>=8.0.0

# Progress & Logging
tqdm>=4.65.0
tensorboard>=2.13.0

# [SECONDAIRE] Application Flask
flask>=3.0.0
werkzeug>=3.0.0

# [SECONDAIRE] Génération de rapports PDF
reportlab>=4.0.0
fpdf2>=2.7.0

# Optional: Experiment tracking
# wandb>=0.15.0
# mlflow>=2.5.0
```

### 9.3 Configuration YAML Principale

```yaml
# configs/config.yaml

# =============================================================================
# CONFIGURATION GÉNÉRALE DU PROJET
# =============================================================================

project:
  name: "vehicle-damage-detection"
  version: "1.0.0"
  description: "CNN from scratch pour détection de dommages véhicules"
  author: "Karamooo"
  
# =============================================================================
# GOOGLE DRIVE PATHS (utilisés dans Colab)
# =============================================================================

drive:
  root: "/content/drive/MyDrive"
  project: "/content/drive/MyDrive/ENSA_Deep_Learning"
  
# =============================================================================
# DONNÉES (sur Google Drive)
# =============================================================================

data:
  # Chemins Google Drive
  datasets_dir: "/content/drive/MyDrive/ENSA_Deep_Learning/datasets"
  raw_dir: "/content/drive/MyDrive/ENSA_Deep_Learning/datasets/raw"
  processed_dir: "/content/drive/MyDrive/ENSA_Deep_Learning/datasets/processed"
  
  # Datasets bruts
  cardd_dir: "/content/drive/MyDrive/ENSA_Deep_Learning/datasets/raw/CarDD_release/CarDD_COCO"
  stanford_dir: "/content/drive/MyDrive/ENSA_Deep_Learning/datasets/raw/stanford_cars_224/car_data"
  
  image:
    size: [224, 224]
    channels: 3
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  
  split:
    train: 0.70
    val: 0.15
    test: 0.15
    seed: 42
    stratified: true
  
  classes:
    - undamaged  # Label 0
    - damaged    # Label 1

# =============================================================================
# AUGMENTATION
# =============================================================================

augmentation:
  train:
    horizontal_flip:
      p: 0.5
    rotation:
      degrees: 15
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.1
      hue: 0.05
    random_resized_crop:
      scale: [0.8, 1.0]
  
  val: null  # Pas d'augmentation pour validation
  test: null # Pas d'augmentation pour test

# =============================================================================
# MODÈLES
# =============================================================================

models:
  baseline:
    name: "BaselineCNN"
    num_classes: 2
    dropout_rate: 0.5
    
  deep:
    name: "DeepCNN"
    num_classes: 2
    
# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

training:
  batch_size: 32
  num_epochs: 100
  num_workers: 2          # Réduit pour Colab
  pin_memory: true
  
  optimizer:
    name: "Adam"
    lr: 0.001
    weight_decay: 0.0001
    betas: [0.9, 0.999]
  
  scheduler:
    name: "ReduceLROnPlateau"
    mode: "min"
    factor: 0.1
    patience: 5
    min_lr: 0.000001
  
  early_stopping:
    patience: 10
    min_delta: 0.0001
    monitor: "val_loss"
  
  checkpointing:
    save_best_only: true
    monitor: "val_loss"
    
# =============================================================================
# ÉVALUATION
# =============================================================================

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - confusion_matrix
    
# =============================================================================
# CHEMINS OUTPUTS (sur Google Drive)
# =============================================================================

paths:
  checkpoints: "/content/drive/MyDrive/ENSA_Deep_Learning/checkpoints"
  model_a_ckpt: "/content/drive/MyDrive/ENSA_Deep_Learning/checkpoints/model_a"
  model_b_ckpt: "/content/drive/MyDrive/ENSA_Deep_Learning/checkpoints/model_b"
  outputs: "/content/drive/MyDrive/ENSA_Deep_Learning/outputs"
  logs: "/content/drive/MyDrive/ENSA_Deep_Learning/outputs/logs"
  figures: "/content/drive/MyDrive/ENSA_Deep_Learning/outputs/figures"

# =============================================================================
# REPRODUCTIBILITÉ
# =============================================================================

seed: 42
deterministic: true
```

### 9.4 Template Première Cellule Notebook (Setup Colab)

Chaque notebook doit commencer par cette cellule de setup :

```python
# ==============================================================================
# SETUP COLAB - À EXÉCUTER EN PREMIER
# ==============================================================================

# 1. Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Vérifier le GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 3. Définir les chemins (depuis paths.py ou directement)
DRIVE_ROOT = "/content/drive/MyDrive"
PROJECT_ROOT = f"{DRIVE_ROOT}/ENSA_Deep_Learning"

# Datasets
RAW_DATA_DIR = f"{PROJECT_ROOT}/datasets/raw"
PROCESSED_DATA_DIR = f"{PROJECT_ROOT}/datasets/processed"
CARDD_DIR = f"{RAW_DATA_DIR}/CarDD_release/CarDD_COCO"
STANFORD_DIR = f"{RAW_DATA_DIR}/stanford_cars_224/car_data"

# Outputs
CHECKPOINTS_DIR = f"{PROJECT_ROOT}/checkpoints"
OUTPUTS_DIR = f"{PROJECT_ROOT}/outputs"
FIGURES_DIR = f"{OUTPUTS_DIR}/figures"

# 4. Ajouter src/ au path (si le code est sur Drive ou cloné)
import sys
# Option A: Si le code est cloné sur Colab
# sys.path.append('/content/vehicle-damage-detection/src')

# Option B: Si le code est sur Drive
# sys.path.append(f'{DRIVE_ROOT}/vehicle-damage-detection/src')

# 5. Installer les packages manquants (si nécessaire)
# !pip install albumentations -q

# 6. Vérifier que les dossiers existent
import os
print("\n📁 Vérification des dossiers:")
print(f"  ✓ Project root: {os.path.exists(PROJECT_ROOT)}")
print(f"  ✓ Raw data: {os.path.exists(RAW_DATA_DIR)}")
print(f"  ✓ CarDD: {os.path.exists(CARDD_DIR)}")
print(f"  ✓ Stanford: {os.path.exists(STANFORD_DIR)}")

print("\n✅ Setup Colab terminé!")
```

---

## 10. Spécifications d'Implémentation

### 10.1 Conventions de Code

```python
# Style: PEP 8 avec les adaptations suivantes

# Imports
import torch                          # Standard library first
import torch.nn as nn                  # Then related packages
from torch.utils.data import DataLoader

import numpy as np                     # Third-party
import pandas as pd
from sklearn.metrics import f1_score

from src.models import BaselineCNN     # Local imports last
from src.data import VehicleDataset

# Type hints obligatoires pour les fonctions publiques
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """
    Docstring Google style.
    
    Args:
        model: Le modèle à entraîner
        dataloader: DataLoader d'entraînement
        criterion: Fonction de perte
        optimizer: Optimiseur
        device: Device (CPU/GPU)
    
    Returns:
        Tuple (loss moyenne, accuracy)
    """
    pass

# Constantes en MAJUSCULES
BATCH_SIZE = 32
NUM_CLASSES = 2
```

### 10.2 Gestion des Erreurs

```python
# Vérifications explicites
def load_image(path: str) -> torch.Tensor:
    """Charge une image avec gestion d'erreur."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image non trouvée: {path}")
    
    try:
        image = Image.open(path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Impossible de charger l'image {path}: {e}")
    
    return image


# Assertions pour le debug
def forward(self, x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 4, f"Expected 4D tensor, got {x.dim()}D"
    assert x.size(1) == 3, f"Expected 3 channels, got {x.size(1)}"
    # ...
```

### 10.3 Logging

```python
import logging

# Configuration du logging
def setup_logging(log_file: str = None):
    """Configure le système de logging."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


# Utilisation
logger = logging.getLogger(__name__)

def train():
    logger.info("Début de l'entraînement")
    logger.info(f"Batch size: {BATCH_SIZE}")
    # ...
    logger.info(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.2f}%")
```

### 10.4 Reproductibilité

```python
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    """
    Fixe toutes les graines aléatoires pour reproductibilité.
    
    Note: Pour une reproductibilité totale sur GPU, ajouter:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### 10.5 Fonctionnalités Secondaires

> ⚠️ **Note** : Ces fonctionnalités sont optionnelles et ne font pas partie des critères critiques d'évaluation. Elles démontrent cependant une maturité supplémentaire du projet.

#### 10.5.1 Application Flask de Démonstration

**Objectif** : Permettre à un utilisateur d'uploader une image de véhicule et recevoir une prédiction de dommage.

```python
# app/app.py - Structure de base
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Charger le modèle entraîné
model = load_model('checkpoints/model_b/best_model.pth')
model.eval()

@app.route('/')
def index():
    """Page d'accueil avec formulaire d'upload."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction.
    
    Reçoit une image, applique le prétraitement,
    effectue la prédiction et retourne le résultat.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier envoyé'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Fichier vide'}), 400
    
    # Sauvegarder et traiter l'image
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Prédiction
    image = preprocess_image(filepath)
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
    
    class_names = ['undamaged', 'damaged']
    result = {
        'prediction': class_names[predicted_class],
        'confidence': probabilities[0][predicted_class].item() * 100,
        'probabilities': {
            name: prob.item() * 100 
            for name, prob in zip(class_names, probabilities[0])
        }
    }
    
    return render_template('result.html', result=result, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**Fonctionnalités de l'interface** :
- Upload d'image par glisser-déposer ou sélection
- Affichage de l'image uploadée
- Prédiction avec niveau de confiance
- Visualisation des probabilités par classe
- Option de télécharger un rapport PDF

#### 10.5.2 Génération Automatique de Rapports PDF

**Objectif** : Générer un rapport de diagnostic professionnel après analyse d'une image.

```python
# app/utils/report_generator.py
from fpdf import FPDF
from datetime import datetime
import os

class DamageReportGenerator:
    """
    Génère des rapports PDF de diagnostic de dommages véhicules.
    
    Le rapport inclut:
    - Informations sur le véhicule (si fournies)
    - Image analysée
    - Résultat de la prédiction
    - Niveau de confiance
    - Date et heure de l'analyse
    - Recommandations
    """
    
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
    
    def generate_report(
        self,
        image_path: str,
        prediction: str,
        confidence: float,
        probabilities: dict,
        vehicle_info: dict = None,
        output_path: str = None
    ) -> str:
        """
        Génère un rapport PDF complet.
        
        Args:
            image_path: Chemin vers l'image analysée
            prediction: Classe prédite ('damaged' ou 'undamaged')
            confidence: Niveau de confiance (0-100)
            probabilities: Probabilités par classe
            vehicle_info: Infos véhicule (optionnel)
            output_path: Chemin de sortie (auto-généré si None)
        
        Returns:
            Chemin vers le fichier PDF généré
        """
        self.pdf.add_page()
        
        # En-tête
        self._add_header()
        
        # Informations véhicule (si fournies)
        if vehicle_info:
            self._add_vehicle_info(vehicle_info)
        
        # Image analysée
        self._add_image_section(image_path)
        
        # Résultats de l'analyse
        self._add_results_section(prediction, confidence, probabilities)
        
        # Recommandations
        self._add_recommendations(prediction, confidence)
        
        # Pied de page
        self._add_footer()
        
        # Sauvegarde
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"outputs/reports/damage_report_{timestamp}.pdf"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.pdf.output(output_path)
        
        return output_path
    
    def _add_header(self):
        """Ajoute l'en-tête du rapport."""
        self.pdf.set_font('Arial', 'B', 20)
        self.pdf.cell(0, 15, 'Rapport de Diagnostic Véhicule', ln=True, align='C')
        self.pdf.set_font('Arial', '', 12)
        self.pdf.cell(0, 10, f'Date: {datetime.now().strftime("%d/%m/%Y %H:%M")}', ln=True, align='C')
        self.pdf.ln(10)
    
    def _add_results_section(self, prediction, confidence, probabilities):
        """Ajoute la section des résultats."""
        self.pdf.set_font('Arial', 'B', 14)
        self.pdf.cell(0, 10, 'Résultats de l\'Analyse', ln=True)
        self.pdf.set_font('Arial', '', 12)
        
        # Verdict principal
        status_color = (255, 0, 0) if prediction == 'damaged' else (0, 128, 0)
        self.pdf.set_text_color(*status_color)
        self.pdf.set_font('Arial', 'B', 16)
        verdict = 'DOMMAGE DÉTECTÉ' if prediction == 'damaged' else 'AUCUN DOMMAGE DÉTECTÉ'
        self.pdf.cell(0, 15, verdict, ln=True, align='C')
        
        # Reset couleur
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_font('Arial', '', 12)
        
        # Confiance
        self.pdf.cell(0, 10, f'Niveau de confiance: {confidence:.1f}%', ln=True)
        
        # Probabilités détaillées
        self.pdf.ln(5)
        for class_name, prob in probabilities.items():
            self.pdf.cell(0, 8, f'  - {class_name}: {prob:.1f}%', ln=True)
    
    def _add_recommendations(self, prediction, confidence):
        """Ajoute des recommandations basées sur le résultat."""
        self.pdf.ln(10)
        self.pdf.set_font('Arial', 'B', 14)
        self.pdf.cell(0, 10, 'Recommandations', ln=True)
        self.pdf.set_font('Arial', '', 11)
        
        if prediction == 'damaged':
            if confidence > 90:
                rec = "Dommage clairement identifié. Inspection physique recommandée."
            elif confidence > 70:
                rec = "Dommage probable. Vérification manuelle conseillée."
            else:
                rec = "Résultat incertain. Analyse supplémentaire nécessaire."
        else:
            if confidence > 90:
                rec = "Véhicule en bon état apparent. Aucune action requise."
            else:
                rec = "Pas de dommage évident, mais vérification visuelle conseillée."
        
        self.pdf.multi_cell(0, 8, rec)
```

**Contenu du rapport** :
- En-tête avec logo et date
- Informations véhicule (plaque, modèle, etc.)
- Image analysée intégrée
- Verdict clair (DOMMAGE / PAS DE DOMMAGE)
- Niveau de confiance avec indicateur visuel
- Probabilités détaillées par classe
- Recommandations automatiques
- Pied de page avec disclaimer

---

## 11. Checklist de Validation

### 11.1 Avant de Coder — Setup Environnement

#### Google Drive
- [ ] Dossier `ENSA_Deep_Learning/` créé dans My Drive
- [ ] Sous-dossiers créés : `datasets/raw/`, `datasets/processed/`
- [ ] Sous-dossiers créés : `checkpoints/model_a/`, `checkpoints/model_b/`
- [ ] Sous-dossiers créés : `outputs/figures/`, `outputs/logs/`

#### VS Code + Extension Colab
- [ ] Extension **Google Colab** installée dans VS Code
- [ ] Extension **Jupyter** installée
- [ ] Connexion Google testée

#### Datasets (upload sur Drive)
- [ ] **CarDD** téléchargé (~5 GB) et uploadé dans `datasets/raw/`
- [ ] **Stanford Cars 224×224** téléchargé (~500 MB) et uploadé dans `datasets/raw/`
- [ ] Structure CarDD vérifiée : `CarDD_COCO/train2017/`, `val2017/`, `test2017/`
- [ ] Dossiers inutiles ignorés : `annotations/`, `CarDD_SOD/`

### 11.2 Préparation des Données (dans Colab)

- [ ] Google Drive monté dans Colab
- [ ] GPU disponible vérifié (`nvidia-smi`)
- [ ] Images Stanford collectées depuis tous les sous-dossiers
- [ ] Échantillonnage de 4,000 images Stanford (seed=42)
- [ ] Dataset combiné créé : 8,000 images (4,000 damaged + 4,000 undamaged)
- [ ] Split stratifié 70/15/15 appliqué
- [ ] Structure `processed/train/`, `val/`, `test/` créée sur Drive
- [ ] Distribution des classes vérifiée (50/50 dans chaque split)
- [ ] Images visualisées (qualité, résolution)

### 11.3 Architecture Model A

- [ ] VGGBlock implémenté et testé
- [ ] BaselineCNN implémenté
- [ ] Forward pass testé (pas d'erreur de dimension)
- [ ] Nombre de paramètres vérifié (~6.5M)
- [ ] Chaque choix justifié dans le code (commentaires)

### 11.4 Architecture Model B

- [ ] ResidualBlock implémenté avec skip connection
- [ ] Projection 1×1 fonctionnelle
- [ ] DeepCNN implémenté
- [ ] Forward pass testé
- [ ] Nombre de paramètres vérifié (~11M)
- [ ] Initialisation des poids implémentée
- [ ] Différences avec Model A clairement documentées

### 11.5 Pipeline d'Entraînement

- [ ] Dataset PyTorch fonctionnel
- [ ] DataLoaders configurés (num_workers=2 pour Colab)
- [ ] Fonction de perte choisie (CrossEntropyLoss)
- [ ] Optimiseur configuré (Adam)
- [ ] Scheduler configuré (ReduceLROnPlateau)
- [ ] Early stopping implémenté
- [ ] Checkpointing sur Google Drive fonctionnel
- [ ] Logging des métriques (TensorBoard)

### 11.6 Évaluation

- [ ] Métriques calculées correctement
- [ ] Matrice de confusion générée
- [ ] Courbes d'apprentissage tracées (sauvées sur Drive)
- [ ] Comparaison Model A vs B documentée
- [ ] Analyse des erreurs (FP, FN) réalisée

### 11.7 Livrables Finaux

- [ ] Code propre et documenté
- [ ] Notebooks reproductibles
- [ ] README complet
- [ ] Rapport académique rédigé
- [ ] Présentation PowerPoint préparée
- [ ] Application Flask fonctionnelle (si implémentée)
- [ ] Tous les fichiers sur GitHub

---

## 12. Glossaire Technique

| Terme | Définition |
|-------|------------|
| **Batch Normalization** | Normalisation des activations par mini-batch, accélère l'entraînement |
| **Champ réceptif** | Zone de l'image d'entrée qui influence un neurone donné |
| **Dropout** | Désactivation aléatoire de neurones pendant l'entraînement (régularisation) |
| **Early Stopping** | Arrêt de l'entraînement quand la validation ne s'améliore plus |
| **F1-Score** | Moyenne harmonique de precision et recall |
| **Feature map** | Sortie d'une couche convolutive |
| **From scratch** | Implémenté par nous, pas importé d'une librairie |
| **Global Average Pooling** | Moyenne spatiale d'une feature map (réduit à 1×1) |
| **Kernel/Filtre** | Matrice de poids apprise par convolution |
| **MaxPool** | Opération de pooling prenant le maximum local |
| **Overfitting** | Le modèle mémorise le train set au lieu de généraliser |
| **Padding** | Ajout de zéros autour de l'image pour préserver la taille |
| **Precision** | TP / (TP + FP) — fiabilité des prédictions positives |
| **Recall** | TP / (TP + FN) — capacité à détecter tous les positifs |
| **ReLU** | Rectified Linear Unit: max(0, x) |
| **ResNet** | Architecture avec skip connections (He et al., 2015) |
| **Skip connection** | Connexion qui "saute" des couches (x + F(x)) |
| **Stride** | Pas de déplacement du kernel |
| **Transfer learning** | Réutiliser un modèle pré-entraîné (INTERDIT ici) |
| **VGG** | Architecture simple avec convolutions 3×3 empilées |
| **Vanishing gradient** | Gradients qui deviennent trop petits dans les réseaux profonds |

---

## 📚 Références Académiques

1. **Simonyan & Zisserman (2014)** — "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG)
2. **He et al. (2015)** — "Deep Residual Learning for Image Recognition" (ResNet)
3. **Ioffe & Szegedy (2015)** — "Batch Normalization: Accelerating Deep Network Training"
4. **Srivastava et al. (2014)** — "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"

---



> **Ce PRD est la source de vérité.**  

> 1. **Respecte les contraintes académiques** — pas de modèles pré-définis
> 2. **Documente chaque choix** — le professeur veut des concepteurs
> 3. **Teste chaque composant** — forward pass, dimensions, gradients
> 4. **Priorise la clarté** — code lisible > code clever
> 5. **Suis la structure** — organisation professionnelle
>
