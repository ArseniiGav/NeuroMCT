import numpy as np

class LogLikelihood:
    def __init__(self, data, model):
        self._offset = 1
        self._data = np.array(data) + self._offset
        self._model = model

    def __call__(self, pars):
        lmbd = self._model(pars) + self._offset
        return np.sum(self._data * np.log(lmbd) - lmbd)

class LogLikelihoodRatio:
    def __init__(self, data, model):
        self._offset = 1
        self._data = np.array(data) + self._offset
        self._model = model

    def __call__(self, pars):
        lmbd = self._model(pars) + self._offset
        mask = (self._data == 0)
        if np.any(mask):
            ln = np.zeros(len(self._data))
            ln[~mask] = np.log(lmbd[~mask] / self._data[~mask])
        else:
            ln = np.log(lmbd / self._data)
        return -2 * np.sum(self._data * ln + self._data - lmbd)
