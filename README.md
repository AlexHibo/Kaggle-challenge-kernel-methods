# 🧠 Kernel Methods for Image Classification
**MVA / IASD 2025-2026 - Data Challenge Kaggle**

Ce projet contient l'implémentation d'un pipeline complet de classification d'images pour un sous-ensemble de CIFAR-10, basé exclusivement sur des **méthodes à noyaux**. L'objectif est de maximiser la précision sans utiliser de bibliothèques de Deep Learning, tout en gérant un dataset comportant des labels bruités.

**Auteurs :** Alexandre Mallez & Rayane Dakhlaoui

---

## 📈 Performance Finale
* **Accuracy (Validation) :** ~69%
* **Méthode retenue :** Fusion de descripteurs CKN & HOG + PCA + Kernel Ridge Regression (KRR) avec noyau RBF.

---

## 🚀 Pipeline Technique

Le script `run.py` exécute la séquence optimisée suivante :

### 1. Prétraitement & Augmentation
* **Normalisation :** Mise à l'échelle des pixels.
* **Augmentation :** Flips horizontaux et "Random Crops" (padding de 4px reflect) pour moyenner les prédictions et gagner en robustesse face au bruit des labels.

### 2. Extraction de Features
* **CKN (Convolutional Kernel Networks) :** Couche convolutive avec filtres appris par **K-means (implémenté from scratch)**, blanchiment ZCA et Spatial Pyramid Pooling (SPM) sur 3 niveaux (1x1, 2x2, 4x4).
* **HOG (Histogram of Oriented Gradients) :** Implémentation manuelle (9 bins d'orientation, cellules 4x4, normalisation par bloc L2-Hys).

### 3. Réduction & Régularisation
* **PCA (via SVD) :** Projection sur les **2048 composantes principales** pour éliminer le bruit et accélérer le calcul de la matrice de Gram.

### 4. Classification
* **KRR (Kernel Ridge Regression) :** Résolution analytique via décomposition de Cholesky.
* **Noyau RBF :** $$K(x, y) = \exp(-\gamma \|x-y\|^2)$$ avec $\gamma$ estimé sur la variance des données.

---

## 📂 Structure du Projet

```text
.
├── run.py                 # Script principal (génère Yte_pred.csv)
├── requirements.txt       # Dépendances (numpy, scipy, cvxopt, pandas)
├── kernel_method.pdf      # Rapport théorique
├── challenge_kernel.ipynb # Notebook d'expérimentation
└── src/                   # Code source
    ├── __init__.py
    ├── data.py            # Chargement et data augmentation
    ├── kernels.py         # RBF, Linear, Chi2, Hist. Intersection
    ├── ckn.py             # CKN (K-means, whitening, convolution)
    ├── hog.py             # Descripteurs HOG "from scratch"
    ├── pca.py             # PCA via SVD
    └── regressors.py      # KRR et SVM (via cvxopt)
