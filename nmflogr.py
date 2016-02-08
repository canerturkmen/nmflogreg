import numpy as np

from .model import Model
from .mixins import NMFMixin, LogisticMixin

from scipy.special import expit as _sigmoid

class CoupledNMFLogR(Model, NMFMixin, LogisticMixin):

    def __init__(self, r=10, method="kl", phi=0., eta=.1, **kwargs):
        """
        Initialize the coupled NMF and logistic regression model.

        :param r: the number of ranks for NMF to approximate to
        :type r: int
        :param method: the divergence function of NMF. one of "kl" or "euc"
        :type method: str
        :param phi: the dispersion parameter balancing NMF and logistic cost functions. Both for
            Generalized Poisson and Euclidean cases
        :type phi: float
        :param eta: the step size for logistic regression
        :type eta: float
        """

        super(CoupledNMFLogR, self).__init__(**kwargs)

        self.R = r
        self.method = method
        self.phi = phi
        self.eta = eta # step size

        if method == "kl":
            self.update_W = self._update_W_kl_disp
            self.update_H = self._update_H_coupled_kl_disp
            self.divergence_func = self._kl_divergence
        elif method == "euc":
            #raise NotImplementedError("Euclidean distance currently not implemented")
            self.update_W = self._update_W_euc
            self.update_H = self._update_H_coupled_euc
            self.divergence_func = self._euc_distance

    def fit(self, X, y, mask=None):
        """
        Fit the coupled NMF and logistic regression model. Returns a tuple of the divergence metric
        and negative log likelihood of the logistic regression, recorded once per iteration.

        :param X: the data (design) matrix
        :param y: the binary outcome values
        :param mask: the 'mask', of the same shape as y to take into account when
            updating parameters. 1 if the value is observed, 0 otherwise

        :type X: numpy.array
        :type y: numpy.array
        :type mask: numpy.array
        """

        I, J = X.shape
        R = self.R
        self.converged = False
        eps = self.eps

        err = []
        nll  = []

        W = np.random.rand(I, R)
        H = np.random.rand(R, J)

        beta = np.random.rand(R)
        phi = self.phi # precision parameter

        err.append(self.divergence_func(X, W, H))
        nll.append(self._negative_log_likelihood(beta,y,H,mask))
        for i in range(self.maxiter):

            #include the dispersion parameter in the second matrix calculation
            #only for the Gaussian case

            H = self.update_H(X, W, H, y, beta, phi, mask)
            # normalize H
            # H = H / (np.linalg.norm(H, axis=0, ord=1) + 1e-3)

            W = self.update_W(X, W, H, phi)

            # update the logistic regression parameters
            beta = self._update_param(beta, y, H, mask, eta=self.eta)

            err.append(self.divergence_func(X, W, H))
            nll.append(self._negative_log_likelihood(beta,y,H,mask))

            if abs((err[-1] - err[-2])/(err[-2] + eps)) < self.tol and abs((nll[-1] - nll[-2])/(nll[-2] + eps)) < self.tol:
                self.converged = True
                break

        self.params = (W, H, beta)
        return (err, nll)

    def _update_H_coupled_kl_disp(self, X, W, H, y, beta, phi=0., mask=None):
        eps = 1e-12
        WH = W.dot(H)

        if mask is None:
            mask = np.ones(len(y))

        return H * (W.T.dot(X * (WH + phi) / (WH * (WH + phi*X) + eps)) + (H * y * mask)) / ((np.sum(W,0) + eps) + (_sigmoid(beta.dot(H)) * H * mask).T ).T

    def _update_W_kl_disp(self, X, W, H, phi=0.):
        """
        update first parameterizing matrix as per KL divergence multiplicative
        update step
        """
        eps = self.eps
        WH = W.dot(H)
        return W * (X * (WH + phi) / (WH * (WH + phi*X) + eps)).dot(H.T) / (np.sum(H,1) + eps)

    def _update_H_coupled_kl(self, X, W, H, y, beta, mask=None):
        eps = 1e-12
        if mask is None:
            return H * (W.T.dot(X / (W.dot(H) + eps)) + (H * y)) / ((np.sum(W,0) + eps) + (_sigmoid(beta.dot(H)) * H).T ).T
        else:
            return H * (W.T.dot(X / (W.dot(H) + eps)) + (H * y * mask)) / ((np.sum(W,0) + eps) + (_sigmoid(beta.dot(H)) * H * mask).T ).T

    def _update_H_coupled_euc(self, X, W, H, y, beta, phi=1., mask=None):
        eps = 1e-12
        if mask is None:
            return H * ((1 / phi+eps) * W.T.dot(X) + (H * y)) / ( (1 / phi+eps) * (W.T.dot(W).dot(H) + eps).T + (_sigmoid(beta.dot(H)) * H).T ).T
        else:
            return H * ((1 / phi+eps) * W.T.dot(X) + (H * y)) /                                 \
             ( (1 / phi+eps) * (W.T.dot(W).dot(H) + eps).T + (_sigmoid(beta.dot(H)) * H).T ).T

    def get_params(self):
        return self.params

    def transform(X):
        raise NotImplementedError
