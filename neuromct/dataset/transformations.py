import torch


class PoissonNoise:
    def __call__(self, spectra):
        # Apply Poisson noise
        spectra_noised = torch.poisson(spectra)
        return spectra_noised


class NormalizeToUnity:
    def __call__(self, spectra):
        # Normalize spectra to get a pdf
        pdf = spectra / torch.sum(spectra)
        return pdf
