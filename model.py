"""
Abstract class for statistical models
"""

from abc import ABCMeta, abstractmethod

class Model(object):
    """
    This abstract class Model contains an interface to be implemented for all
    probabilistic model classes contained in this library.
    """
    __metaclass__ = ABCMeta
    eps = 1e-12 # the epsilon value (very small number)

    def __init__(self, **kwargs):
        """
        Initialize a model. Possible keywords are "maxiter", the maximum number of
        iterations. "tol", or tolerance for relative improvement in the gradient
        descent updates.
        """
        self.maxiter = 2000
        self.tol = 1e-1
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    @abstractmethod
    def fit(*args):
        raise NotImplementedError

    @abstractmethod
    def get_params(*args):
        raise NotImplementedError

    @staticmethod
    def check_correct(a):
        """
        Check that an array of numbers is indeed nonincreasing. Used on algorithm
        error vectors to check correctness.
        :param a: the Python list of floats, all corresponding to an error value
        :type a: list
        """
        correct = True
        EPS = 1e-12
        for ix in range(1, len(a)):
            if a[ix] /(a[ix-1] + EPS) >= 1.2:
                correct = False
                break

        return correct
