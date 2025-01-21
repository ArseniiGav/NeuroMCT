from tqdm import tqdm
import numpy as np

# from ..configs import data_configs
# from ..models.ml import setup
# from ..models.ml.tede import TEDE
# from ..plot import matplotlib_setup
# from ..utils import tede_argparse

class SamplerMH:
    def __init__(self, cost_fn, initial_pos, cov, rng=None):
        self._cost_fn = cost_fn
        self._initial_pos = initial_pos
        self._npars = len(initial_pos)
        self._cov = cov
        self._rng = rng if rng is not None else np.random.default_rng(None)

    def sample(self, n_samples):
        self.samples = np.zeros((n_samples, self._npars))
        self.samples[0] = self._initial_pos

        self.log_probs = np.zeros(n_samples)
        self.log_probs[0] = self._cost_fn(*self._initial_pos)
        accepted = 0

        chain_steps = self._rng.multivariate_normal(
                np.zeros(self._npars), self._cov, n_samples-1
                )

        for i in tqdm(range(1, n_samples)):
            current_x = self.samples[i-1]
            proposed_x = current_x + chain_steps[i-1]

            current_log_prob = self.log_probs[i-1]
            proposed_log_prob = self._cost_fn(*proposed_x)

            log_acceptance_ratio = proposed_log_prob - current_log_prob

            if np.log(self._rng.uniform(0, 1)) < log_acceptance_ratio:
                self.samples[i] = proposed_x
                self.log_probs[i] = proposed_log_prob
                accepted += 1
            else:
                self.samples[i] = current_x
                self.log_probs[i] = current_log_prob

        self._last_position = self.samples[-1]
        self.acceptance_rate = accepted / n_samples
        return self.samples, self.log_probs, self.acceptance_rate

    def save(self):
        pass

    def estimate_covariance(self, n_samples):
        pass
