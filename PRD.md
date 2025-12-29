# PRD — Système de détection automatique des dommages sur véhicules

**Version :** 1.0

**Date :** 2025-12-29

---

## 1. Contexte et synthèse

Les entreprises de location de véhicules font face à des litiges fréquents lors de la restitution des véhicules : dommages mineurs (rayures, bosses, fissures) difficiles à détecter ou contestés par les propriétaires. L'objectif est de proposer une solution automatique basée sur la vision par ordinateur et le deep learning pour détecter et localiser visuellement les dommages avant/après la location et faire un rapport comparative dans le but de détecter les domages causés par le client locataire.

---

## 2. Objectifs produit

* Développer et comparer **deux architectures CNN** implémentées from scratch pour détecter les dommages visibles sur véhicules.
* Fournir des résultats reproductibles (code, notebooks, jeux de données décrits) et un rapport académique détaillé.
* Atteindre des performances mesurables et comparables via des métriques standards (precision, recall, F1, mAP si détection).

---

## 3. Portée (Scope)

### 3.1 In-Scope (ce que nous ferons)

* Conception et implémentation **from scratch** de deux architectures CNN : baseline simple (VGG-like) et deep CNN amélioré (skip connections).
* Prétraitement des images et pipeline d'augmentation des données.
* Entraînement, validation et test sur jeux de données publics pertinents.
* Évaluation quantitative (accuracy, precision, recall, F1, IoU, mAP si annotations bbox/masque disponibles).
* Notebooks reproductibles (prétraitement, entraînement, évaluation) et dépôt GitHub structuré.

### 3.2 Out-of-Scope 

* Système complet de comparaison avant / après automatisé.
* Génération automatique de rapports PDF pour les inspections.
* Déploiement en production (API, application mobile, edge).
* Intégration avec systèmes tiers (gestion flotte, CRM).

---


## 5. Exigences fonctionnelles

### 5.1 Données

* Supporter images RGB en formats standards (JPEG/PNG).
* Lire annotations en formats COCO / Pascal VOC / Roboflow exports (JSON, XML).
* Pipeline d'augmentation configurable (rotations, flips, crops, variations d'éclairage).

### 5.2 Modèles

* **Baseline CNN (Model A)** : VGG-like, blocs Conv→ReLU→Pool, quelques couches fully-connected, sortie classification binaire/multi-classe.
* **Deep CNN (Model B)** : plus de profondeur, blocs résiduels/skips implémentés manuellement, dropout, batch-norm optionnel.
* Les deux modèles doivent être implémentés *from scratch* (pas de recours à modèles pré-entraînés ni import d'architectures complètes).

### 5.3 Tâches prises en charge

* **Classification image** : (a) sans dommage / (b) avec dommage (ou multi-classes : scratch, dent, fissure).
* **Option souhaitée** : détection localisée via boîtes englobantes si jeux de données fournis (entraînement d’un head de détection basé sur le backbone CNN développé).

### 5.4 Entraînement et évaluation

* Séparation train/validation/test (ex. 70/15/15) et réplicabilité (seed fixé).
* Early stopping, scheduler LR, checkpoints.
* Metrics : Accuracy, Precision, Recall, F1-score; si localisation : IoU, mAP@0.5.

### 5.5 Reproductibilité et code

* Scripts/notebooks reproductibles (prétraitement, entraînement, évaluation).
* Fichiers de configuration YAML pour hyperparamètres.
* `requirements.txt` complet.

---

## 6. Exigences non-fonctionnelles


* **Performances** : viser une précision et un rappel cohérents (>75% comme objectif réaliste dépendant du dataset) ; démontrer trade-offs.
* **Organisation** : repo clair, README, data/README, listant MIT

---

## 7. Données recommandées (sélection initiale)

* **CarDD (USTC)** — images + masques/COCO : adapté pour segmentation/détection.
* **Roboflow — car-damage (Skillfactory)** — grandes quantités d’images annotées en bboxes (31 classes).
* **Roboflow — Car Damage Images (Kadad)** — petit dataset pour prototypage bbox.
* **Humans-in-the-Loop — Car Parts & Damages** — segmentation polygonale, CC0.
* **Kaggle — Car Damage Assessment** — classification image-level (utile pour baseline).

> Remarque : On privilégie d’abord un jeu mixte (bbox/segmentation) pour entraîner une solution avec localisation ; compléter par datasets plus petits si nécessaire.

---

## 8. Critères de succès & métriques

* **Technique (modèle)** : atteindre des F1-scores comparables entre validation et test ; rapport d’analyse des erreurs (FP/FN).
* **Reproductibilité** : scripts exécutables, notebooks documentés, résultats replicables par un troisième lecteur.
* **Qualité** : architecture expliquée, justification des choix, analyse expérimentale claire.

Objectifs chiffrés indicatifs :

* Baseline (Model A) : F1 ≥ 0.70
* Deep (Model B) : amélioration significative vs baseline (ΔF1 ≥ 0.05)
* mAP@0.5 (si bbox) : viser > 0.5 sur dataset testé (dépendra du dataset)

---

## 9. Plan de release / jalons (sprint-like)

* **Phase 0 — Préparation (1)**

  * Recherche datasets, structure du repo, environment, `requirements.txt`.

* **Phase 1 — Data & Prétraitement**

  * Téléchargement, standardisation, annotations, notebooks de prétraitement.

* **Phase 2 — developpement**

  * Implémenter Model A, entraînement initial, essais d’augmentations.
  * Implémenter Model B (skip connections), régularisation, hyperparam tuning.
  * Implémentation des autres fonctionnalités (Hors Dl)

* **Phase 4 — Évaluation & Rapport**

  * Comparaison, analyses d’erreurs, figures, rédaction rapport final.


---

## 10. Livrables

* Repo GitHub complet (structure validée)
* Notebooks : prétraitement, baseline, deep model
* Code source `src/` : modèles, utils, entraînement, évaluation
* Fichiers de configuration (YAML), `requirements.txt`
* Rapport PDF et présentation slides.

---

## 11. Risques et atténuations

* **Risque : données insuffisantes ou déséquilibrées** —> atténuation : data augmentation, combiner datasets, synthetic augmentation (simuler rayures).
* **Risque : overfitting** —> atténuation : dropout, regularization, early stopping, cross-validation.
* **Risque : non-conformité académique (usage de modèles pré-entraînés)** —> atténuation : implémentation from scratch, documenter tout usage externe.
* **Risque : performances faibles en condition réelle (lumière, angles)** —> atténuation : diversifier les données, tests d’augmentations photométriques

---

## 13. Annexes utiles (à inclure dans le repo)

* `data/README.md` : liste complète des datasets, liens, licences, counts.
* `docs/technical_design.md` : description couche-par-couche des deux architectures.
* `docs/evaluation_protocol.md` : protocole d’évaluation, seeds, split details.
* `docs/report.md` : squelette du rapport final.

---
