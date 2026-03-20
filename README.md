# Kaggle-challenge-kernel-methods
Our work on the kaggle challenge during the kernel course from MVA/IASD
Voici une version exhaustive et structurée de ton README.md. C'est un "gros" Markdown qui regroupe le contexte théorique (issu de ton PDF), la structure technique du code et le mode d'emploi.🧠 Kernel Methods for Image ClassificationMVA / IASD 2025-2026 - Data Challenge KaggleCe projet implémente un pipeline de classification d'images (sous-ensemble de CIFAR-10) basé exclusivement sur des méthodes à noyaux. L'objectif était de maximiser la précision sans utiliser de bibliothèques de Deep Learning (type PyTorch/TensorFlow), en gérant un dataset bruité.Auteurs : Alexandre Mallez & Rayane Dakhlaoui📈 Performance FinaleAccuracy (Validation) : ~69%Méthode gagnante : Fusion de features CKN & HOG + PCA + Kernel Ridge Regression (KRR) avec noyau RBF.🚀 Le Pipeline TechniqueLe script run.py exécute la séquence optimisée suivante :1. Prétraitement & AugmentationNormalization : Mise à l'échelle des pixels.Augmentation : Flips horizontaux et "Random Crops" (avec padding de 4px) pour moyenner les prédictions et gagner en robustesse face au bruit des labels.2. Extraction de Features (The Core)CKN (Convolutional Kernel Networks) : Implémentation d'une couche convolutive avec filtres aléatoires (ou K-means), blanchiment ZCA et Pooling Spatial (SPM) sur 3 niveaux (1x1, 2x2, 4x4).HOG (Histogram of Oriented Gradients) : Implémentation "from scratch" (gradients, 9 bins d'orientation, cellules 4x4, normalisation par bloc L2-Hys).3. Réduction & RégularisationPCA (SVD) : Projection sur les 2048 composantes principales. Cela permet de réduire le bruit, d'éviter l'overfitting et d'accélérer drastiquement le calcul du noyau.4. ClassificationKRR (Kernel Ridge Regression) : Résolution analytique via décomposition de Cholesky (plus rapide que le SVM pour des grands volumes de features).Noyau RBF : Paramètre $\gamma$ optimisé en fonction de la variance des données.📂 Structure du DépôtPlaintext.
├── run.py                 # SCRIPT PRINCIPAL : Entraîne et génère Yte_pred.csv
├── requirements.txt       # Liste des dépendances (numpy, scipy, cvxopt, etc.)
├── kernel_method.pdf      # Rapport théorique complet
├── challenge_kernel.ipynb # Notebook de recherche et visualisation
└── src/                   # Modules internes
    ├── data.py            # Loading, mise en forme (HWC) et data augmentation
    ├── kernels.py         # Fonctions de noyaux (RBF, Linear, Chi2, Hist. Intersection)
    ├── ckn.py             # Architecture CKN (whitening, convolution, pooling)
    ├── hog.py             # Descripteurs HOG faits main
    ├── pca.py             # Analyse en composantes principales via SVD
    └── regressors.py      # Classifieurs KRR et SVM (dual QP via cvxopt)
🛠️ Installation & Utilisation1. PrérequisAssure-toi d'avoir Python 3.8+ d'installé.Bash# Création de l'environnement
python -m venv ml_cuda
source ml_cuda/bin/activate  # Linux/Mac
# ou .\ml_cuda\Scripts\activate  # Windows

# Installation des dépendances
pip install -r requirements.txt
2. ExécutionPlace les fichiers Xtr.csv, Xte.csv et Ytr.csv à la racine, puis lance :Bashpython run.py
Le script va créer un fichier Yte_pred.csv prêt pour la soumission Kaggle.📊 Résultats des ExpérimentationsÉtapeMéthodePrécision (Val)Gain Clé1Raw pixels + RBF-SVM~25%Baseline2HOG + RBF-SVM~57%Invariance spatiale3HOG + Augmentation + KRR~62%Robustesse au bruit4CKN (512 filters) + PCA + KRR~63%Features non-linéaires5CKN + HOG + PCA + KRR~69%Fusion de descripteurs💡 Détails théoriquesNoyau RBF : $K(x, y) = \exp(-\gamma \|x-y\|^2)$Régularisation : Utilisation de $\lambda = 0.05$ pour compenser le bruit important dans les étiquettes de CIFAR-10.Whitening (ZCA) : Appliqué sur les patchs CKN pour décorréler les pixels voisins avant la convolution.
