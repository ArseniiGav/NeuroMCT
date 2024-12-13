import torch


class PoissonNoise:
    def __call__(self, x):
        # Apply Poisson noise
        x_noised = torch.poisson(x)
        return x_noised


class NormalizeToUnity:
    def __call__(self, x):
        # Normalize x to unity (e.g. normalize spectra to get a PDF)
        x_normalized = x / torch.sum(x)
        return x_normalized


class LogScale:
    def __init__(self, ε=1e-8):
        self.ε = ε

    def __call__(self, x):
        # Apply the log rescaling transformation
        x_log_scale = torch.log(x + self.ε)
        return x_log_scale
