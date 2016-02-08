"""
Logistic regression
"""

from .model import Model
from .mixins import LogisticMixin
import numpy as np

class LogisticRegression(Model, LogisticMixin):

    def __init__(self, update="bgd", eta=0.1, **kwargs):
        """
        Initialize the "vanilla" Logistic Regression model with batch gradient descent learner.

        :param update: one of 'multi' or 'bgd'. multi chooses multiplicative updates, bgd
            is classical batch gradient descent. Multiplicative update learns only nonnegative
            weight vectors thus solving a "nonnegative logistic regression"
        :param eta: the step size for logistic regression
        """
        super(LogisticRegression, self).__init__(**kwargs)

        self.eta = eta

        if update=='multi':
            self.func_update = self._update_param_mult
        else:
            self.func_update = self._update_param

    def fit(self, X, y, mask=None):
        """
        Fit the logistic regression model.

        :param X: design matrix of shape (n_features, n_obs)
        :param y: the response vector of shape (n_obs,)
        """
        I, J = X.shape
        err = []

        self.converged = False

        w = np.random.rand(I)
        err.append(self._negative_log_likelihood(w, y, X, mask))

        for i in range(self.maxiter):
            # update parameter
            w = self.func_update(w, y, X, mask, eta=self.eta)

            # calculate errors and check convergence
            err.append(self._negative_log_likelihood(w, y, X, mask))

            if abs((err[-1] - err[-2])/(err[-2] + self.eps)) < self.tol:
                self.converged = True
                break

        self.params = w
        return err

    def get_params(self):
        return self.params

    def transform(self, X):
        return self._score(self.params, X)
