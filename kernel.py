"""
NOT IMPLEMENTED

Module for implementation of the Kernel NMF and the Kernel-NMF coupled model
"""

import numpy as np
from .model import Model

class KernelNMF(Model):

    def __init__(self, r=10, method="euc", **kwargs):
        raise NotImplementedError("Kernel NMF variant is not yet fully implemented")
        # super(KernelNMF, self).__init__(**kwargs)
        # self.r = r
        # if method == "euc":
        #     self.divergence_func = self._divergence_euc
        # else:
        #     raise NotImplementedError

    def _update_W_euc(self, A, W):
        return W * .5 * (1 + (A.dot(W) / (W.dot(W.T).dot(W) + 1e-6)))

    def _divergence_euc(self, A, W):
        return np.linalg.norm(A - W.dot(W.T), "fro")

    def fit(self,A):
        """
        :param A: the similarity matrix to be decomposed
        """
        N, _ = A.shape
        R = self.r
        errors = []

        W = np.random.rand(N, R)

        for i in range(self.maxiter):
            W = self._update_W_euc(A, W)
            errors.append(self.divergence_func(A, W))

        self.params = W
        return errors

    def get_params(self):
        return self.params

class CoupledKernelLogR(Model):

    def __init__(self, r=10, method="euc", **kwargs):
        raise NotImplementedError("Kernel NMF variant is not yet fully implemented")
        # super(KernelNMF, self).__init__(**kwargs)
        # self.r = r
        # if method == "euc":
        #     self.divergence_func = self._divergence_euc
        # else:
        #     raise NotImplementedError

    def fit(self, A, y):
        """
        :param A: the similarity (kernel) matrix to be decomposed. shape (n_obs, n_obs)
        :param y: the binary targets for the classification problem
        """
        pass
