"""
Mixins for logistic regression and NMF models, shared across different models
"""

import numpy as np
from scipy.special import expit as _sigmoid, kl_div
from scipy.stats import bernoulli

class LogisticMixin(object):

    def _negative_log_likelihood(self, w, y, X, mask=None):
        """
        Returns logistic regression negative log likelihood
        :param w: the parameters at their current estimates of shape (n_features,)
        :param y: the response vector of shape (n_obs,)
        :param X: the design matrix of shape (n_features, n_obs)
        :param mask: the binary mask vector of shape (n_obs,). 1 if observed, 0 o/w

        :returns: negative log likelihood value
        :rtype: float
        """
        sigm = _sigmoid(w.dot(X))
        if mask is not None:
            return -np.sum(np.log(bernoulli.pmf(y, sigm) * mask + 1e-5))
        else:
            return -np.sum(np.log(bernoulli.pmf(y, sigm) + 1e-5))

    def _update_param(self, w, y, X, mask=None, eta=0.01):
        """
        :param w: the parameters at their current estimates of shape (n_features,)
        :param y: the response vector of shape (n_obs,)
        :param X: the design matrix of shape (n_features, n_obs)
        :param mask: the binary mask vector of shape (n_obs,). 1 if observed, 0 o/w
        :param eta: the batch gradient descent step size

        :returns: updated parameter vector of shape (n_features,)
        """
        # if mask is not None:
        #     X = X * mask
        #     y = y * mask
        if mask is not None:
            return w + eta * X.dot(mask*  (y - _sigmoid(w.dot(X)))) #?
        else:
            return w + eta * X.dot((y - _sigmoid(w.dot(X)))) #?

    def _update_param_mult(self, w, y, X, mask=None):
        """
        Logistic regression, implemented with the multiplicative update rule. Note that the
        multiplicative update works quite poorly and only handles the case where a
        nonnegative coefficient vector is required.

        :param w: the parameters at their current estimates of shape (n_features,)
        :param y: the response vector of shape (n_obs,)
        :param X: the design matrix of shape (n_features, n_obs)
        :param mask: the binary mask vector of shape (n_obs,). 1 if observed, 0 o/w

        :returns: updated parameter vector of shape (n_features,)
        """
        if mask is not None:
            X = X * mask
            y = y * mask
        return w * X.dot(y) / (X.dot(_sigmoid(w.dot(X))) + 1e-10)

    def _score(self, w, X):
        return _sigmoid(w.dot(X))

class NMFMixin(object):

    def _kl_divergence(self, X, W, H):
        """
        Calculate the generalized Kullback-Leibler divergence (also called Information Divergence or
        I-Divergence) between two matrices.
        """
        B = W.dot(H)
        return np.sum(kl_div(X,B))

    def _euc_distance(self, X, W, H):
        """
        Calculate the Euclidean distance between two matrices.
        """
        return np.linalg.norm(X - W.dot(H), "fro")

    def _update_W_kl(self, X, W, H):
        """
        update first parameterizing matrix as per KL divergence multiplicative
        update step
        """
        eps = self.eps
        return W * (X / (W.dot(H) + eps)).dot(H.T) / (np.sum(H,1) + eps)

    def _update_H_kl(self, X, W, H):
        """
        Update the second factor matrix as per KL divergence multiplicative update
        """
        eps = self.eps
        return H * (W.T.dot(X / (W.dot(H) + eps)).T / (np.sum(W,0) + eps)).T

    def _update_W_euc(self, X, W, H, phi=1.):
        """
        :param phi: the dispersion parameter
        :type phi: float
        """
        eps = self.eps
        return W * (1 / phi+eps) * X.dot(H.T) / (W.dot(H).dot(H.T) + eps)

    def _update_H_euc(self, X, W, H, phi=1.):
        """
        :param phi: the dispersion parameter
        :type phi: float
        """
        eps = self.eps
        return H * (1 / phi + eps) * W.T.dot(X) / (W.T.dot(W).dot(H) + eps)
