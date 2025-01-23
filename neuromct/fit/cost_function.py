import numpy as np

class LogLikelihood:
    def __init__(self, data, model):
        self._data = data
        self._model = model

    def __call__(self, pars):
        lmbd = self._model(pars)
        return np.sum(self._data * np.log(lmbd) - lmbd)
