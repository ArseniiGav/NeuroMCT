import numpy as np
from torch import tensor, int64, float64
from scipy.stats import poisson

class LogLikelihood:
    def __init__(self, data, model):
        self._data = np.array(data)
        self._model = model

    def __call__(self, pars):
        lmbd = self._model(pars)
        return np.sum(self._data * np.log(lmbd) - lmbd)

class NegativeLogLikelihood:
    def __init__(self, data, model):
        self._data = np.array(data)
        self._model = model

    def __call__(self, pars):
        lmbd = self._model(pars)
        return -2 * np.sum(self._data * np.log(lmbd) - lmbd)

class UnbinnedNegativeLogLikelihood:
    def __init__(self, data, model):
        self._n_tot = len(data)
        self._data = tensor(np.array(data), dtype=float64)
        self._model = model

    def __call__(self, pars):
        nl_pars, norm = pars[:3], pars[-1]
        log_prob = self._model(
                self._data,
                nl_pars,
                ).sum()
        log_poisson = poisson.logpmf(self._n_tot, norm*self._n_tot)
        return -2 * (log_prob + log_poisson)

class UnbinnedLogLikelihood:
    def __init__(self, data, model):
        self._n_tot = len(data)
        self._data = tensor(np.array(data), dtype=float64)
        self._model = model

    def __call__(self, pars):
        nl_pars, norm = pars[:3], pars[-1]
        log_prob = self._model(
                self._data,
                nl_pars,
                ).sum()
        log_poisson = poisson.logpmf(self._n_tot, norm*self._n_tot)
        return log_prob + log_poisson

class LogLikelihoodRatio:
    def __init__(self, data, model):
        self._data = np.array(data)
        self._model = model

    def __call__(self, pars):
        lmbd = self._model(pars)
        mask = (self._data == 0)
        if np.any(mask):
            ln = np.zeros(len(self._data))
            ln[~mask] = np.log(lmbd[~mask] / self._data[~mask])
        else:
            ln = np.log(lmbd / self._data)
        return -2 * np.sum(self._data * ln + self._data - lmbd)
