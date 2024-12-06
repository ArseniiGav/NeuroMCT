import torch


class PoissonResampling:
    def __init__(self, N_resamplings):
        self.N_resamplings = N_resamplings

    def __call__(self, sample):
        spectra, params, source_types = sample
        
        # Poisson resampling of the spectra
        spectra_resampled = torch.poisson(spectra.repeat(self.N_resamplings, 1))  # Shape: (N_resamplings, number_of_bins)
        
        # Repeat params and source_types to match N_resamplings
        params_resampled = params.repeat(self.N_resamplings, 1)  # Shape: (N_resamplings, number_of_params)
        source_types_resampled = source_types.repeat(self.N_resamplings)  # Shape: (N_resamplings,)
        
        return spectra_resampled, params_resampled, source_types_resampled


class NormalizeToUnity:
    def __call__(self, sample):
        spectra, params, source_types = sample
        
        # Normalize spectra to get a pdf
        pdf = spectra / torch.sum(spectra, dim=1, keepdim=True)
        
        return pdf, params, source_types
