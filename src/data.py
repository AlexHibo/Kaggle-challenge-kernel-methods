"""
Data loading, preprocessing, and augmentation.
"""
import numpy as np
import pandas as pd


def load_data(data_dir="."):
    """Load Xtr, Xte, Ytr from CSV files."""
    Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072))) 
    Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072))) 
    Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()
    return Xtr, Xte, Ytr


def to_hwc(X_flat):
    """(N, 3072) layout R…R G…G B…B -> (N, 32, 32, 3) float64."""
    N = X_flat.shape[0]
    return X_flat.reshape(N, 3, 32, 32).transpose(0, 2, 3, 1)


def train_val_split(X, y, val_frac=0.2, seed=42):
    """Stratified train/val split."""
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        va_idx.extend(idx[:n_val])
        tr_idx.extend(idx[n_val:])
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]


def augment(X_hwc, y, seed=42, multi=2):
    """
    Flip horizontal
    Random crop : pad de 4px en reflect, puis crop 32x32 aléatoire
    """
    rng = np.random.default_rng(seed)
    N = len(X_hwc)
    out_X, out_y = [X_hwc], [y]

    # flip horizontal 
    flipped = X_hwc[:, :, ::-1, :].copy()
    out_X.append(flipped) 
    out_y.append(y)

    # random crop (pad 4, crop 32x32) 
    pad = 4
    padded = np.pad(X_hwc, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='reflect')
    crops = np.empty_like(X_hwc)
    for i in range(N):
        r = rng.integers(0, 2*pad)
        c = rng.integers(0, 2*pad)
        crops[i] = padded[i, r:r+32, c:c+32, :]
    out_X.append(crops)
    out_y.append(y)

    X_aug = np.concatenate(out_X, axis=0)
    y_aug = np.concatenate(out_y, axis=0)

    indices = np.random.choice(len(X_aug), N * multi, replace=False)

    return X_aug[indices], y_aug[indices]


def accuracy_value(y_true, y_pred):
    return (y_true == y_pred).mean()

