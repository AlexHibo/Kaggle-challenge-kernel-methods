"""
SVM (OvA) classifier using cvxopt (dual QP).
"""
import numpy as np
import cvxopt
import cvxopt.solvers
from src.kernels import rbf_kernel
from scipy.linalg import solve as scipy_solve

cvxopt.solvers.options["show_progress"] = False

def one_hot(y, n_classes=10):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


class KernelRidgeClassifier:
    def __init__(self, lam=1.0, kernel_fn=None, **kernel_kwargs):
        self.lam           = lam
        self.kernel_fn     = kernel_fn if kernel_fn is not None else rbf_kernel
        self.kernel_kwargs = kernel_kwargs

    def fit(self, X_tr, y_tr):
        self.X_tr_ = X_tr
        K = self.kernel_fn(X_tr, X_tr, **self.kernel_kwargs)
        K += self.lam * np.eye(len(X_tr))
        Y = one_hot(y_tr)
        self.alpha_ = scipy_solve(K, Y, assume_a='pos')
        return self

    def decision_function(self, X_te):
        K_te = self.kernel_fn(X_te, self.X_tr_, **self.kernel_kwargs)
        return K_te @ self.alpha_

    def predict(self, X_te):
        return np.argmax(self.decision_function(X_te), axis=1)

    def score(self, X_te, y_te):
        return np.mean(self.predict(X_te) == y_te)


def solve_binary_svm(K, y_bin, C=1.0):
    """
    Solve the dual SVM QP for a binary problem.

    K     : (N, N) kernel matrix (symmetric PSD)
    y_bin : (N,)   labels in {-1, +1}
    C     : regularization

    Returns: alpha (N,), b (float)
    """
    N = len(y_bin)
    y = y_bin.astype(np.float64)

    Q = np.outer(y, y) * K

    P = cvxopt.matrix(Q)
    q = cvxopt.matrix(-np.ones(N))
    G = cvxopt.matrix(np.vstack([-np.eye(N), np.eye(N)]))
    h = cvxopt.matrix(np.hstack([np.zeros(N), C * np.ones(N)]))
    A = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(np.zeros(1))

    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    if sol["status"] != "optimal":
        print(f"  Warning: cvxopt status: {sol['status']}")

    alpha = np.array(sol["x"]).ravel()

    decision = (alpha * y) @ K
    sv_mask = (alpha > 1e-5) & (alpha < C - 1e-5)
    if sv_mask.sum() == 0:
        sv_mask = alpha > 1e-5
    b_val = float(np.mean(y[sv_mask] - decision[sv_mask]))

    return alpha, b_val


class SVMClassifier:
    """
    OvA multiclass SVM. 
    """

    def __init__(self, C=1.0):
        self.C = C
        self.alphas_ = {}
        self.biases_ = {}
        self.classes_ = None

    def fit(self, K_tr, y_tr):
        """
        K_tr : (N, N) precomputed kernel matrix
        y_tr : (N,)   labels 0..9
        """
        self.classes_ = np.unique(y_tr)
        self.y_tr_ = y_tr
        for c in self.classes_:
            print(f"  SVM class {c} vs rest ...", end="\r")
            y_bin = np.where(y_tr == c, 1.0, -1.0)
            alpha, b = solve_binary_svm(K_tr, y_bin, C=self.C)
            self.alphas_[c] = alpha
            self.biases_[c] = b
        print(f"\nSVM fit done ({len(self.classes_)} classes).")
        return self

    def predict(self, K_te):
        """
        K_te : (N_te, N_tr) kernel matrix test x train
        Returns: (N_te,) predicted labels
        """
        scores = np.column_stack(
            [
                K_te @ (self.alphas_[c] * np.where(self.y_tr_ == c, 1.0, -1.0))
                + self.biases_[c]
                for c in self.classes_
            ]
        )
        return self.classes_[np.argmax(scores, axis=1)]
