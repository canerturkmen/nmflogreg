import numpy as np

from .model import Model
from .mixins import NMFMixin

class NMF(Model, NMFMixin):
    """
    This class implements the classical Lee-Seung Nonnegative Matrix Factorization with
    Multiplicative update algorithms.
    """

    def __init__(self, r=10, method="kl", **kwargs):
        """
        Initialize the NMF model.

        :param r: the "low-rank" for the factor matrices.
        :type r: int
        :param method: one of "kl" (for generalized KL divergence) or "euc" (Euclidean dist),
            the objective function to be minimized.
        :type method: str
        """
        super(NMF, self).__init__(**kwargs)
        self.R = r
        if method == "kl":
            self.update_W = self._update_W_kl
            self.update_H = self._update_H_kl
            self.divergence_func = self._kl_divergence
        elif method == "euc":
            self.update_W = self._update_W_euc
            self.update_H = self._update_H_euc
            self.divergence_func = self._euc_distance

    def fit(self, X):
        """
        Fit the NMF model

        :param X: The design matrix to be factorized with dimensions (n_features, n_obs)
        :type X: numpy.array
        """
        I, J = X.shape
        R = self.R
        eps = self.eps
        err = []
        self.converged = False

        W = np.random.rand(I, R)
        H = np.random.rand(R, J)

        err.append(self.divergence_func(X, W, H))

        for i in range(self.maxiter):
            W = self.update_W(X, W, H)
            H = self.update_H(X, W, H)

            err.append(self.divergence_func(X, W, H))

            if abs((err[-1] - err[-2])/(err[-2] + eps)) < self.tol:
                self.converged = True
                break

        self.params = (W, H)
        return err

    def get_params(self):
        return self.params
