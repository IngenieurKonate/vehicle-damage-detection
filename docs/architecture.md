# Architecture des Modèles

## Model A — Baseline CNN (VGG-like)

Architecture inspirée de VGG avec convolutions 3×3 empilées.

### Caractéristiques
- 5 blocs convolutifs
- Doublement progressif des canaux (32 → 512)
- MaxPooling 2×2 après chaque bloc
- Classifier avec Dropout 0.5
- ~6.5M paramètres

### Structure
```
Input (3, 224, 224)
    ↓
VGGBlock(3→32) → (32, 112, 112)
    ↓
VGGBlock(32→64) → (64, 56, 56)
    ↓
VGGBlock(64→128) → (128, 28, 28)
    ↓
VGGBlock(128→256) → (256, 14, 14)
    ↓
VGGBlock(256→512) → (512, 7, 7)
    ↓
AdaptiveAvgPool → Flatten → FC(512→256) → Dropout → FC(256→2)
    ↓
Output (2)
```

## Model B — Deep CNN avec Skip Connections

Architecture inspirée de ResNet avec blocs résiduels.

### Caractéristiques
- Stem agressif (conv 7×7 stride 2)
- 4 stages de blocs résiduels (2 blocs par stage)
- Skip connections pour meilleur flux de gradient
- BatchNorm pour stabilité
- ~11M paramètres

### Structure
```
Input (3, 224, 224)
    ↓
Stem: Conv7×7(s=2) → BN → ReLU → MaxPool → (64, 56, 56)
    ↓
Stage1: 2× ResidualBlock(64→64) → (64, 56, 56)
    ↓
Stage2: 2× ResidualBlock(64→128, s=2) → (128, 28, 28)
    ↓
Stage3: 2× ResidualBlock(128→256, s=2) → (256, 14, 14)
    ↓
Stage4: 2× ResidualBlock(256→512, s=2) → (512, 7, 7)
    ↓
AdaptiveAvgPool → Flatten → FC(512→2)
    ↓
Output (2)
```

## Comparaison

| Aspect | Model A | Model B |
|--------|---------|---------|
| Profondeur | 10 conv | 18 conv |
| Skip Connections | Non | Oui |
| BatchNorm | Non | Oui |
| Régularisation | Dropout | BatchNorm |
| Paramètres | ~6.5M | ~11M |
