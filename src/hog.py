
import numpy as np


def hog_single(img, orientations=9, ppc=(4, 4), cpb=(2, 2)):
    """
    img       : (H, W, C) float, valeurs quelconques
    orientations : nb de bins angulaires
    ppc       : (cy, cx) pixels par cell
    cpb       : (by, bx) cells par block

    Retourne un vecteur 1D.
    Pour 32x32, ppc=(4,4), cpb=(2,2) :
      grid 8x8 cells → 7x7 blocks → 7*7*2*2*9 = 1764 features
    """
    H, W, C = img.shape
    cy, cx = ppc
    by, bx = cpb

    #  1. Gradients, max-magnitude sur les canaux ─
    Gx = np.zeros((H, W))
    Gy = np.zeros((H, W))
    mag_best = np.zeros((H, W))

    for c in range(C):
        ch = img[:, :, c].astype(np.float64)
        # différences centrales avec bords simples
        gx = np.empty_like(ch)
        gy = np.empty_like(ch)

        gx[:, 1:-1] = ch[:, 2:] - ch[:, :-2]
        gx[:, 0]    = 0.0  
        gx[:, -1]   = 0.0   
        gy[1:-1, :] = ch[2:, :] - ch[:-2, :]
        gy[0, :]    = 0.0
        gy[-1, :]   = 0.0

        mag = np.hypot(gx, gy)
        mask = mag > mag_best
        Gx[mask] = gx[mask]
        Gy[mask] = gy[mask]
        mag_best[mask] = mag[mask]

    # 2. Orientation non-signée [0, pi) 
    angle = np.arctan2(Gy, Gx) % np.pi   # [0, pi)

    # 3. Cell histograms — soft-binning bilinéaire 
    n_cy = H // cy
    n_cx = W // cx
    bin_w = np.pi / orientations

    # Pour chaque pixel : bin flottant, bin bas et haut + poids
    bin_f  = angle / bin_w - 0.5          # bin "continu"
    bin_lo = np.floor(bin_f).astype(int) % orientations
    bin_hi = (bin_lo + 1) % orientations
    w_hi   = bin_f - np.floor(bin_f)
    w_lo   = 1.0 - w_hi

    cells = np.zeros((n_cy, n_cx, orientations))
    for i in range(n_cy):
        for j in range(n_cx):
            sl = np.s_[i*cy:(i+1)*cy, j*cx:(j+1)*cx]
            m  = mag_best[sl].ravel()
            bl = bin_lo[sl].ravel()
            bh = bin_hi[sl].ravel()
            wl = w_lo[sl].ravel()
            wh = w_hi[sl].ravel()
            np.add.at(cells[i, j], bl, m * wl)
            np.add.at(cells[i, j], bh, m * wh)

    #  Block normalization L2-Hys 
    n_by = n_cy - by + 1
    n_bx = n_cx - bx + 1
    feats = []
    for i in range(n_by):
        for j in range(n_bx):
            block = cells[i:i+by, j:j+bx, :].ravel().copy()
            norm  = np.sqrt(block @ block + 1e-5)
            block /= norm
            np.clip(block, 0, 0.2, out=block)
            norm2 = np.sqrt(block @ block + 1e-5)
            feats.append(block / norm2)

    return np.concatenate(feats)


def extract_hog(X_hwc, orientations=9, ppc=(4, 4), cpb=(2, 2)):
    """Applique hog_single sur toute une matrice (N, 32, 32, 3)."""
    features = []
    for img in X_hwc:
        f = hog_single(img, orientations, ppc, cpb)
        features.append(f)
    return np.array(features)