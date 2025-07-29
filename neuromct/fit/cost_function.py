"""
Cost functions for parameter inference and model fitting.

This module provides likelihood-based cost functions for both binned and unbinned
analyses, supporting different statistical inference methods including Bayesian nested sampling and MCMC,
and Frequentist optimization with the MINOS algorithm.
"""

import numpy as np
from torch import tensor, int64, float64
from scipy.stats import poisson

class LogLikelihood:
    """
    Log-likelihood function for binned maximum likelihood estimation.
    
    This class implements the log-likelihood function for Poisson-distributed binned data.
    
    Parameters
    ----------
    data : array_like
        Observed event counts in each bin
    model : callable
        Model function that returns expected counts given parameters
        
    Methods
    -------
    __call__(pars)
        Evaluate the log-likelihood for given parameters
    """
    def __init__(self, data, model):
        self._data = np.array(data)
        self._model = model

    def __call__(self, pars):
        """
        Evaluate log-likelihood for given parameters.
        
        Parameters
        ----------
        pars : array_like
            Model parameters
            
        Returns
        -------
        float
            Log-likelihood value
        """
        lmbd = self._model(pars)
        return np.sum(self._data * np.log(lmbd) - lmbd)

class NegativeLogLikelihood:
    """
    Negative log-likelihood function for minimization-based fitting.
    
    This class implements the negative log-likelihood scaled by -2, which
    follows a chi-squared distribution under certain conditions. Used for
    optimization algorithms that perform minimization.
    
    Parameters
    ----------
    data : array_like
        Observed event counts in each bin
    model : callable
        Model function that returns expected counts given parameters
        
    Methods
    -------
    __call__(pars)
        Evaluate the negative log-likelihood for given parameters
    """
    def __init__(self, data, model):
        self._data = np.array(data)
        self._model = model

    def __call__(self, pars):
        """
        Evaluate negative log-likelihood for given parameters.
        
        Parameters
        ----------
        pars : array_like
            Model parameters
            
        Returns
        -------
        float
            Negative log-likelihood value (scaled by -2)
        """
        lmbd = self._model(pars)
        return -2 * np.sum(self._data * np.log(lmbd) - lmbd)

class UnbinnedNegativeLogLikelihood:
    """
    Unbinned negative log-likelihood function for continuous data.
    
    This class implements the unbinned negative log-likelihood for continuous
    data analysis, typically used with normalizing flows models that can
    evaluate exact probability densities at individual data points.
    
    Parameters
    ----------
    data : array_like
        Observed continuous data points (e.g., N_pe values)
    model : callable
        Model function that returns probability density given parameters
        
    Attributes
    ----------
    _n_tot : int
        Total number of data points
    _data : torch.Tensor
        Data converted to PyTorch tensor format
    _model : callable
        Model function for density evaluation
        
    Methods
    -------
    __call__(pars)
        Evaluate the unbinned negative log-likelihood for given parameters
    """
    def __init__(self, data, model):
        self._n_tot = len(data)
        self._data = tensor(np.array(data), dtype=float64)
        self._model = model

    def __call__(self, pars):
        """
        Evaluate unbinned negative log-likelihood for given parameters.
        
        Parameters
        ----------
        pars : array_like
            Model parameters, where pars[:3] are physics parameters
            and pars[-1] is a normalization factor
            
        Returns
        -------
        float
            Unbinned negative log-likelihood value
        """
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
