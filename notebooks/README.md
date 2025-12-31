# Dataset – CarDD (Car Damage Detection Dataset)

## Source
CarDD est un dataset public dédié à la détection de dommages sur véhicules.
Il repose sur le format COCO (Common Objects in Context).

## Contenu utilisé
Cette version correspond au split *train* du dataset.

- Nombre total d’images : 2816
- Images contenant au moins un dommage : 2816
- Nombre total de dommages annotés : 6211
- Nombre de classes de dommages : 6

## Types de dommages
Les annotations couvrent 6 types de dommages différents, fournis dans le fichier
`instances_train2017.json`.

## Nettoyage et préparation
Les vérifications suivantes ont été effectuées :
- aucune bounding box invalide détectée
- aucune bounding box trop petite détectée
- toutes les images contiennent au moins un dommage

Aucune annotation n’a été supprimée à cette étape.
Le dataset brut est conservé intact dans `data/raw/`.

## Utilisation
Le dataset est prêt à être utilisé pour des tâches de détection fine de dommages
(localisation par bounding boxes et classification multi-classes).

# prerequies pour utiliser le dataset propre
Installer les dépendances suivantes :

```bash
pip install numpy==1.26.4
pip install pillow matplotlib
pip install torch torchvision torchaudio
pip install pycocotools
```
# Étapes d’utilisation

1. Exécuter uniquement les cellules de la **partie 1** du notebook de prétraitement.
2. Depuis le terminal, accéder au dossier **src** (`cd src`).
3. Exécuter le fichier **preprocess_card.py** (`python preprocess_card.py`).
4. Le dataset redimensionné doit apparaître dans le dossier **data/processed**.
5. Revenir dans le notebook et exécuter la **partie 2 – vérification des annotations**.
6. Une image du dataset annoté s’affichera : si c’est le cas, tout est OK.
7. Tu peux ensuite commencer à travailler dans ton notebook **cnn_baseline**, en démarrant avec le code de la **partie 3-Création du datasets pytorch**. du notebook de pretraitement
