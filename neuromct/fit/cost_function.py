import numpy as np

class LogLikelihood:
    def __init__(self, data, model):
        self._data = data
        self._model = model

    def __call__(self, pars):
        lmbd = self._model(pars)
        return np.sum(self._data * np.log(lmbd) - lmbd)

class LogPoissonRatio:
    def __init__(self, data, model):
        self._data = data
        self._model = model

    def __call__(self, pars):
        lmbd = self._model(pars)
        mask = lmbd <= 1e-16
        if mask.any():
            lmbd[mask] = 1e-16
        mask = (self._data == 0)
        if mask.any():
            ln = np.zeros(len(self._data))
            ln[~mask] = np.log(lmbd[~mask] / self._data[~mask])
        else:
            ln = np.log(lmbd / self._data)
        return -2 * np.sum(self._data * ln + self._data - lmbd)
