

from numpy.linalg import svd
from scipy.stats import entropy
import numpy as np

def pca_analysis(F, title="PCA", verbose=False):
    """Print variance thresholds + effective rank."""
    mean = F.mean(axis=0)
    _, s, Vt = svd(F - mean, full_matrices=False)
    var_ratio = (s**2) / (s**2).sum()
    cum_var = np.cumsum(var_ratio)

    if verbose:
        print(f"\n--- {title} diagnostic ---")
        for thresh in [0.90, 0.95, 0.99]:
            d = np.searchsorted(cum_var, thresh) + 1
            print(f"  {thresh*100:.0f}% variance -> {d} components (out of {len(s)})")

    p = (s**2) / (s**2).sum()
    eff_rank = np.exp(entropy(p))
    if verbose:
        print(f"  Effective rank: {eff_rank:.1f} / {len(s)}")
    return mean, s, Vt


def pca_project(F_tr, F_te, mean, Vt, n_components):
    """Project onto top PCA components."""
    Vt_d = Vt[:n_components]
    return (F_tr - mean) @ Vt_d.T, (F_te - mean) @ Vt_d.T