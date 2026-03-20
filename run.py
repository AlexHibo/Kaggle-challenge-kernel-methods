"""
Final pipeline
  1. Load data
  2. Data augmentation (flip + crops) with feature averaging
  3. Extract CKN features (random filters + ZCA + ReLU + SPM)
  4. Extract HOG features
  5. Standardize + PCA
  6. KRR classification
  7. Generate submission

"""
import numpy as np
import pandas as pd
from src.data import load_data, to_hwc, augment
from src.kernels import rbf_kernel, feat_standardize
from src.ckn import extract_ckn_features
from src.regressors import KernelRidgeClassifier
from src.pca import pca_analysis, pca_project
from src.hog import extract_hog


Xtr, Xte, Ytr = load_data()

SEED = 42
n_comp = 2048
lamb= 0.05
alpha = 0.3
verbose = False

np.random.seed(SEED)

X_train, Y_train = augment(to_hwc(Xtr), Ytr, seed=SEED, multi=3)
X_test = Xte.copy()
N = X_train.shape[0]
X_train = X_train.transpose(0, 3, 1, 2).reshape(N, 3072)

# Extract CKN features
F_train, patch_mean, W_zca, filters = extract_ckn_features(
    X_train,
    patch_size=3,        # taille des patches (3x3)
    n_filters=512,       # nombre de filtres aléatoires
    subsampling=2,       # facteur de sous-échantillonnage spatial
    pool_mode='avg',     # 'avg' ou 'max'
    levels=(1, 2, 4),   # niveaux de la pyramide spatiale (SPM)
    use_whitening=True,  # ZCA whitening recommandé
    use_kmeans=True, 
    normalize='l2',      # power-norm + L2
    return_whitening=True,  # récupère les paramètres pour le test
    seed = SEED,
    verbose=verbose)

F_test = extract_ckn_features(
    X_test,
    patch_size=3,
    n_filters=512,
    subsampling=2,
    levels=(1, 2, 4),
    use_whitening=True,
    patch_mean=patch_mean,   # paramètres fitted sur le train
    W_zca=W_zca,             
    filters=filters,          # filtres du train
    return_whitening=False,
    seed=SEED,
    verbose=verbose)

if verbose:
    print("Extract HOG features...")
# Extract HOG features and standardize
F_train_hog = extract_hog(to_hwc(X_train))
F_test_hog = extract_hog(to_hwc(X_test))
F_train_hog_sc, F_test_hog_sc = feat_standardize(F_train_hog, F_test_hog)
F_train_sc, F_eval_sc = feat_standardize(F_train, F_test)

# Combine CKN and HOG features
F_train_group = np.hstack([alpha * F_train_sc, (1 - alpha) * F_train_hog_sc])
F_eval_group = np.hstack([alpha * F_eval_sc, (1 - alpha) * F_test_hog_sc])

if verbose:
    print("PCA projection...")
# PCA projection
pca_mean, pca_s, pca_Vt = pca_analysis(F_train_group, title="CKN features")
F_train_sc, F_eval_sc = pca_project(F_train_group, F_eval_group, pca_mean, pca_Vt, n_comp)

if verbose:
    print("Train Kernel Ridge Regression...")
# Train Kernel Ridge Classifier
model = KernelRidgeClassifier(lam=lamb, kernel_fn=rbf_kernel)
model.fit(F_train_sc, Y_train)

if verbose:
    print("Evaluate on validation set...")
# Predict on test set
Y_pred = model.predict(F_eval_sc)

# Save predictions to CSV
df = pd.DataFrame({'Prediction': Y_pred.astype(int)})
df.index += 1
df.to_csv('Yte.csv', index_label='Id')

print(f"\nFichier 'Yte.csv' généré")

