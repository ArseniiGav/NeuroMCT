"""
Metropolis-Hastings MCMC sampler for Bayesian parameter inference.
"""

from tqdm import tqdm
import numpy as np
from uproot import recreate as uproot_recreate

class SamplerMH:
    """
    Metropolis-Hastings MCMC sampler for Bayesian inference.
        
    Parameters
    ----------
    cost_fn : callable
        Log-likelihood function that takes parameters and returns log-likelihood
    prior_fn : callable  
        Log-prior function that takes parameters and returns log-prior
    initial_pos : array_like
        Starting position for the MCMC chain
    cov : array_like
        Covariance matrix for the proposal distribution
    par_names : list
        Names of the parameters being sampled
    rng : numpy.random.Generator, optional
        Random number generator (default: None)
    """
    def __init__(self, cost_fn, prior_fn, initial_pos, cov, par_names, rng=None):
        self._cost_fn = cost_fn
        self._prior_fn = prior_fn
        self._initial_pos = initial_pos
        self._npars = len(initial_pos)
        self._cov = cov
        self._par_names = par_names
        self._rng = rng if rng is not None else np.random.default_rng(None)
        self._last_position = None

    def sample(self, n_samples):
        """
        Generate MCMC samples using Metropolis-Hastings algorithm.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        dict
            Dictionary containing samples, log-probabilities, log-priors, and acceptance rate
        """
        self.samples = np.zeros((n_samples, self._npars))
        self.samples[0] = self._initial_pos if self._last_position is None else self._last_position

        self.log_probs = np.zeros(n_samples)
        self.log_probs[0] = self._cost_fn(self._initial_pos)

        self.log_priors = np.zeros(n_samples)
        self.log_priors[0] = self._prior_fn(self._initial_pos)

        accepted = 0

        chain_steps = self._rng.multivariate_normal(
                np.zeros(self._npars), self._cov, n_samples-1
                )

        for i in tqdm(range(1, n_samples), miniters=n_samples//10,
                      desc='\n', maxinterval=60*60*24,):
            current_x = self.samples[i-1]
            proposed_x = current_x + chain_steps[i-1]

            current_log_prob = self.log_probs[i-1]
            proposed_log_prob = self._cost_fn(proposed_x)

            current_log_prior = self.log_priors[i-1]
            proposed_log_prior = self._prior_fn(proposed_x)

            log_acceptance_ratio = proposed_log_prob + proposed_log_prior - current_log_prob - current_log_prior
            # log_acceptance_ratio = proposed_log_prob - current_log_prob

            if np.log(self._rng.uniform(0, 1)) < log_acceptance_ratio:
                self.samples[i] = proposed_x
                self.log_probs[i] = proposed_log_prob
                self.log_priors[i] = proposed_log_prior
                accepted += 1
            else:
                self.samples[i] = current_x
                self.log_probs[i] = current_log_prob
                self.log_priors[i] = current_log_prior

        self._last_position = self.samples[-1]
        self.acceptance_rate = accepted / n_samples
        print(self.acceptance_rate)

        return self.samples, self.log_probs, self.log_priors, self.acceptance_rate

    def save(self, output_folder, name, metadata={}):
        full_output = f"{output_folder}/{name}.root"
        res_dict = {
                par_name : array for par_name, array in zip(self._par_names, self.samples.T)
                }
        res_dict['log_probs'] = self.log_probs

        with uproot_recreate(full_output) as file:
            file['results'] = res_dict
            file['acceptance_rate'] = f'{self.acceptance_rate:.2f}'
            # if metadata:
            #     metadata = {k: v for k, v in metadata.items() if v is not None}
            #     file['metadata'] = metadata

    def estimate_covariance(self, n_batch, batch_len, final_batch_len=10000):
        for _ in range(n_batch):
            self.sample(batch_len)
            self._cov = (2.38**2) * np.cov(self.samples.T) / self._npars
            self._initial_pos = self._last_position
        self.sample(final_batch_len)
        self._cov = (2.38**2) * np.cov(self.samples.T) / self._npars
        self._initial_pos = self._last_position
        return self._cov
