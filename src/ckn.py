
import numpy as np

# Meilleur initialisation du kmeans pour que les filtres soient plus variés que des duplicates aléatoires
def kmeans_plus_plus_init(X, k, rng):
    idx = rng.integers(len(X))
    centers = [X[idx]]
    min_dists = np.full(len(X), np.inf, dtype=np.float32)
    
    for _ in range(k - 1):
        c = centers[-1]  # dernier centre ajouté
        
        # Distance au dernier centre seulement 
        diff = X - c
        new_dists = (diff ** 2).sum(axis=1)
        
        # Mise à jour du minimum
        min_dists = np.minimum(min_dists, new_dists)
        
        # Tirage proportionnel
        probs = min_dists / min_dists.sum()
        idx = rng.choice(len(X), p=probs)
        centers.append(X[idx])
    
    return np.stack(centers)

def kmeans_scratch(X, k, n_iter=30, seed=42, verbose=True):
    """
    K-means from scratch.
    X : (N, d) float32
    Returns: centers (k, d)
    """
    rng = np.random.default_rng(seed)
    
    centers = kmeans_plus_plus_init(X, k, rng)
    
    for it in range(n_iter):
        # Assignation : distance euclidienne au carré
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
        X_sq    = (X ** 2).sum(axis=1, keepdims=True)      # (N, 1)
        C_sq    = (centers ** 2).sum(axis=1, keepdims=True) # (k, 1)
        dots    = X @ centers.T                              # (N, k)
        dists   = X_sq + C_sq.T - 2 * dots                  # (N, k)
        labels  = dists.argmin(axis=1)                       # (N,)
        
        # Mise à jour des centres
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if mask.sum() == 0:
                # Cluster vide : réinitialiser aléatoirement
                new_centers[j] = X[rng.integers(len(X))]
            else:
                new_centers[j] = X[mask].mean(axis=0)
        
        # Convergence
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        
        if verbose:
            print(f"  iter {it+1}/{n_iter} — shift: {shift:.6f}")
        if shift < 1e-6:
            print(f"  Convergence atteinte à l'itération {it+1}")
            break
    
    return centers


# Patch extraction 

def _extract_patches_raw(img, patch_size):
    """
    img : (H, W, C)
    Returns: (n_h, n_w, patch_size*patch_size*C)
    """
    H, W, C = img.shape
    p = patch_size
    n_h = H - p + 1
    n_w = W - p + 1
    patches = np.empty((n_h, n_w, p * p * C), dtype=np.float32)
    for r in range(n_h):
        for c in range(n_w):
            patches[r, c] = img[r:r+p, c:c+p, :].ravel()
    return patches


# Random filters 
def make_random_filters(patch_size, n_channels=3, n_filters=256, seed=42):
    """
    Random Gaussian filters, L2-normalized.
    Returns: (n_filters, patch_size*patch_size*n_channels)
    """
    rng = np.random.default_rng(seed)
    d = patch_size * patch_size * n_channels
    W = rng.standard_normal((n_filters, d)).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    return W


#  K-means filters 

def make_kmeans_filters(X, patch_size=3, n_filters=256,
                        patch_mean=None, W_zca=None,
                        n_subsample=200000, seed=42, verbose=True):
    """
    Learn filters via k-means on whitened, L2-normalized patches.
    Returns: (n_filters, patch_dim)
    """

    if verbose:
        print("  K-means filters: collecting patches...")
    rng = np.random.default_rng(seed)
    N = X.shape[0]

    all_patches = []
    for i in range(N):
        img = X[i].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        patches = _extract_patches_raw(img, patch_size)
        all_patches.append(patches.reshape(-1, patches.shape[-1]))

    all_patches = np.concatenate(all_patches, axis=0)

    if len(all_patches) > n_subsample:
        idx = rng.choice(len(all_patches), n_subsample, replace=False)
        all_patches = all_patches[idx]

    # whiten
    if patch_mean is not None and W_zca is not None:
        all_patches = (all_patches - patch_mean) @ W_zca.T

    # L2-normalize
    norms = np.linalg.norm(all_patches, axis=1, keepdims=True)
    all_patches = all_patches / (norms + 1e-8)

    if verbose:
        print(f"  K-means on {len(all_patches)} patches -> {n_filters} clusters...")
    centers = kmeans_scratch(all_patches.astype(np.float32), k=n_filters,
                         n_iter=30, seed=seed, verbose=verbose)

    # L2-normalize centers
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    if verbose:
        print(f"  K-means filters ready: {centers.shape}")
    return centers.astype(np.float32)


#  ZCA whitening 

def fit_patch_whitening(X, patch_size=3, reg=1e-3, n_subsample=200000, seed=42):
    """
    Fit ZCA whitening on a subsample of patches from X.
    X : (N, 3072) flat images
    Returns: patch_mean (d,), W_zca (d, d)
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    # collect patches
    all_patches = []
    for i in range(N):
        img = X[i].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        patches = _extract_patches_raw(img, patch_size)
        all_patches.append(patches.reshape(-1, patches.shape[-1]))

    all_patches = np.concatenate(all_patches, axis=0)

    if len(all_patches) > n_subsample:
        idx = rng.choice(len(all_patches), n_subsample, replace=False)
        all_patches = all_patches[idx]

    patch_mean = all_patches.mean(axis=0)
    centered = all_patches - patch_mean

    cov = (centered.T @ centered) / len(centered)
    U, S, _ = np.linalg.svd(cov)
    W_zca = U @ np.diag(1.0 / np.sqrt(S + reg)) @ U.T
    W_zca = W_zca.astype(np.float32)

    return patch_mean, W_zca


#  Average/max pooling 

def spatial_pool(Z, subsampling, mode='avg'):
    """
    Z : (H, W, C)
    Pool with stride = subsampling.
    mode: 'avg' for average pooling, 'max' for max pooling
    Returns: (H', W', C)
    """
    H, W, C = Z.shape
    s = subsampling
    n_h = H // s
    n_w = W // s
    Z_cut = Z[:n_h * s, :n_w * s, :]
    blocks = Z_cut.reshape(n_h, s, n_w, s, C)
    if mode == 'max':
        return blocks.max(axis=(1, 3))
    return blocks.mean(axis=(1, 3))


# SPM features

def ckn_spm_features(feat_map, levels=(1, 2, 4)):
    """
    Spatial Pyramid Matching on a feature map.
    feat_map : (H, W, n_filters)
    levels : tuple of grid sizes (e.g. 1x1, 2x2, 4x4)
    Returns: concatenated feature vector
    """
    H, W, C = feat_map.shape
    parts = []
    for L in levels:
        bh = max(1, H // L)
        bw = max(1, W // L)
        for i in range(L):
            for j in range(L):
                r0, r1 = i * bh, min((i + 1) * bh, H)
                c0, c1 = j * bw, min((j + 1) * bw, W)
                block = feat_map[r0:r1, c0:c1, :]
                parts.append(block.mean(axis=(0, 1)))
    return np.concatenate(parts)


# Single image CKN layer

def ckn_layer(img, filters, patch_size=3,
              subsampling=2, pool_mode='avg',
              patch_mean=None, W_zca=None):
    """
    Apply one CKN layer to a single (H, W, C) image.

    Returns: feature_map (H', W', n_filters)
    """
    # 1 — Extract patches
    patches = _extract_patches_raw(img, patch_size)
    ph, pw = patches.shape[:2]
    flat = patches.reshape(-1, patches.shape[2])        # (ph*pw, d)

    # 2 — ZCA whitening (if fitted)
    if patch_mean is not None and W_zca is not None:
        flat = (flat - patch_mean) @ W_zca.T

    # 3 — L2-normalise each patch vector
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    flat = flat / (norms + 1e-8)

    # 4 — Arc-cosine kernel approximation: ReLU(patches @ W)
    Z = np.maximum(flat @ filters.T, 0)                 # (ph*pw, n_filters)
    Z = Z.reshape(ph, pw, -1)                            # (ph, pw, n_filters)

    # 5 — Spatial pooling
    return spatial_pool(Z, subsampling, mode=pool_mode)  # (H', W', n_filters)


# Full feature extraction using CKN

def extract_ckn_features(X, patch_size=3, n_filters=256,
                         subsampling=2, pool_mode='avg',
                         levels=(1, 2, 4),
                         use_whitening=True, whitening_reg=1e-3,
                         use_kmeans=False,
                         normalize='l2',
                         patch_mean=None, W_zca=None,
                         filters=None,
                         return_whitening=False,
                         seed=42, verbose=True):
    """
    Extract CKN features for a dataset.

    X : (N, 3072) flat images
    Returns: F (N, feat_dim) or (F, patch_mean, W_zca) if return_whitening
    """
    # Whitening
    if patch_mean is not None and W_zca is not None:
        pass  # pre-fitted (test set path)
    elif use_whitening:
        if verbose:
            print("  Fitting ZCA patch whitening on training patches...")
        patch_mean, W_zca = fit_patch_whitening(
            X, patch_size=patch_size, reg=whitening_reg, seed=seed)
    else:
        patch_mean, W_zca = None, None

    # Filters
    if filters is not None:
        pass  # pre-computed (test set path)
    elif use_kmeans:
        filters = make_kmeans_filters(X, patch_size=patch_size,
                                      n_filters=n_filters,
                                      patch_mean=patch_mean, W_zca=W_zca,
                                      seed=seed, verbose=verbose)
    else:
        filters = make_random_filters(patch_size, n_channels=3,
                                      n_filters=n_filters, seed=seed)

    N = X.shape[0]
    features = []

    for i in range(N):
        if verbose and i % 500 == 0:
            print(f"  CKN features: {i}/{N}")

        img = X[i].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.float32)
        img = img / 255.0 if img.max() > 1.5 else img

        feat_map = ckn_layer(img, filters,
                             patch_size=patch_size,
                             subsampling=subsampling,
                             pool_mode=pool_mode,
                             patch_mean=patch_mean,
                             W_zca=W_zca)
        feat = ckn_spm_features(feat_map, levels=levels)
        features.append(feat)

    F = np.stack(features)

    if normalize == 'l2':
        F = np.sign(F) * np.sqrt(np.abs(F))
        F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8

    F = F.astype(np.float32)

    if return_whitening:
        return F, patch_mean, W_zca, filters
    return F
