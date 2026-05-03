# Kernel Methods for Image Classification
**MVA / IASD 2025-2026 - Kaggle Data Challenge**

This repository contains the implementation of a complete image classification pipeline for a subset of the CIFAR-10 dataset (5000 training / 2000 test images), based exclusively on **kernel methods**. The project was developed without deep learning frameworks, focusing on robust feature extraction and regularization to handle significant label noise (see Report)

**Authors:** Alexandre Mallez & Rayane Dakhlaoui

---

## Performance Summary
* **Final Validation Accuracy:** ~69%
* **Winning Method:** Feature Fusion (CKN & HOG) + PCA + Kernel Ridge Regression (KRR).
* **Ranking:** 4/35 teams with first submission, **1/35** teams with this final pipeline

---

## Technical Pipeline

The `run.py` script executes the following optimized sequence:

### 1. Data Preprocessing & Augmentation
* **Normalization:** Standard pixel scaling and centering.
* **Augmentation:** Horizontal flips and "Random Crops" (4px reflect padding) are used to create a multi-crop voting system. This significantly improves robustness against the training label noise.

### 2. Feature Extraction
* **CKN (Convolutional Kernel Networks):** A convolutional layer using 512 filters learned via **K-means++ (implemented from scratch)**, followed by ZCA whitening and Spatial Pyramid Pooling (SPM) over 3 levels (1x1, 2x2, 4x4).
* **HOG (Histogram of Oriented Gradients):** A manual implementation featuring 9 orientation bins, 4x4 pixel cells, and L2-Hys block normalization for spatial invariance.

### 3. Dimensionality Reduction (PCA)
* **Method:** SVD-based PCA.
* **Component Selection:** Projection onto the top **2048 principal components**. This step acts as a powerful regularizer by discarding noisy directions and drastically reduces the computational cost of the Gram matrix.

### 4. Classification
* **KRR (Kernel Ridge Regression):** Solved analytically via Cholesky decomposition ($K + \lambda I$).
* **RBF Kernel:** $K(x, y) = \exp(-\gamma \|x-y\|^2)$ where $\gamma$ is adaptively estimated based on data variance.
* **Hyperparameters:** $\lambda = 0.05$ (regularization) and $\alpha = 0.3$ (weighting factor between CKN and HOG features).

---

## Project Structure

```text
.
├── run.py                 # MAIN SCRIPT: Trains the model and generates Yte.csv
├── requirements.txt       # Dependencies
├── Report.pdf             # Detailed theoretical report and analysis
├── challenge_kernel.ipynb # Experimental notebook and visualization
└── src/                   # Source code modules
    ├── __init__.py
    ├── data.py            # Loading, HWC transformation, and augmentation
    ├── kernels.py         # RBF, Linear, Chi2, and Histogram Intersection kernels
    ├── ckn.py             # CKN logic (K-means, ZCA whitening, SPM)
    ├── hog.py             # Manual HOG descriptor implementation
    ├── pca.py             # PCA analysis and projection via SVD
    └── regressors.py      # KRR and SVM solvers (via cvxopt)
```

---

## Installation and Generation

Be sure to install the data on this link :
https://www.kaggle.com/competitions/data-challenge-kernel-methods-2025-2026/

```bash
pip install -r requirements.txt
python run.py
```



## Experimental Results

| Step | Method | Val. acc. | Key gain |
| :--- | :--- | :--- | :--- |
| 1 | Raw pixels + RBF-SVM | $\approx 25\%$ | Baseline |
| 2 | HOG + RBF-SVM | $\approx 57\%$ | Spatial invariance |
| 2b | HOG + augmentation + RBF-SVM | $\approx 59\%$ | Augmentation |
| 3 | HOG + augmentation + RBF-KRR | $\approx 62\%$ | Faster regressor |
| 4 | CKN (256 filters) + PCA + RBF-KRR | $\approx 63\%$ | Richer features |
| **5** | **CKN + HOG + PCA + RBF-KRR** | **$\approx 69\%$** | **Fusion** |

## Details 

For more details about our works, you can check the notebook and our report


