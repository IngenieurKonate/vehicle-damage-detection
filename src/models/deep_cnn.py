"""
Model B - Deep CNN avec Skip Connections (ResNet-like).

Architecture CNN profonde avec:
- Skip connections pour meilleur flux de gradient
- BatchNorm pour stabilité
- Stem agressif (conv 7x7 stride 2)
- 4 stages de blocs résiduels
- Global Average Pooling

Paramètres estimés: ~11M
"""

# TODO: Implémenter ResidualBlock, DeepCNN
