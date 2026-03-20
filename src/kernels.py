"""
Kernel functions and feature preprocessing.
"""
import numpy as np

def linear_kernel(X, Y):
    return X @ Y.T


def rbf_kernel(X, Y, gamma=None):
    """
    K(x,y) = exp(-γ ||x-y||^2)
    """
    if gamma is None:
        # gamma = 1.0 / X.shape[1]
        gamma = 1.0 / (X.shape[1] * X.var()) # meilleur generalement
    XX = np.einsum('ij,ij->i', X, X)[:, None]
    YY = np.einsum('ij,ij->i', Y, Y)[None, :]
    D2 = XX + YY - 2.0 * (X @ Y.T)
    np.maximum(D2, 0.0, out=D2)
    return np.exp(-gamma * D2)


def chi2_kernel(X, Y, gamma=1.0):
    """
    K(x,y) = exp(-γ sum_k (x_k - y_k)² / (|x_k| + |y_k| + eps))
    """
    N, M = len(X), len(Y)
    K = np.empty((N, M), dtype=np.float64)
    for i in range(N):
        diff  = X[i] - Y
        denom = np.abs(X[i]) + np.abs(Y) + 1e-10
        dist  = np.sum(diff * diff / denom, axis=1)
        np.clip(dist, 0.0, 500.0 / gamma, out=dist)  # évite  overflow
        K[i]  = np.exp(-gamma * dist)
    return K


def histogram_intersection_kernel(X, Y):
    """
    K(x,y) = sum_k min(x_k, y_k)
    """
    N, M = len(X), len(Y)
    K = np.empty((N, M), dtype=np.float64)
    for i in range(N):
        K[i] = np.sum(np.minimum(X[i], Y), axis=1)
    return K


def polynomial_kernel(X, Y, degree=3, coef0=1.0, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (gamma * (X @ Y.T) + coef0) ** degree


def combine_kernels(kernel_list, weights=None):
    if weights is None:
        weights = [1.0 / len(kernel_list)] * len(kernel_list)
    return sum(w * K for w, K in zip(weights, kernel_list))


def feat_standardize(X_tr, X_te=None):
    """Normalise colonnes (mean=0, std=1). Fit sur train uniquement."""
    mu  = X_tr.mean(axis=0)
    sig = X_tr.std(axis=0) + 1e-8
    X_tr_sc = (X_tr - mu) / sig
    if X_te is not None:
        return X_tr_sc, (X_te - mu) / sig
    return X_tr_sc

