import torch


class PoissonNoise:
    def __call__(self, f):
        # Apply Poisson noise
        f_noised = torch.poisson(f)
        return f_noised


class BuildPDF:
    def __init__(self, Δx):
        self.Δx = Δx

    def __call__(self, f):
        # Build a PDF: normalize to the sum and to the bin size Δx
        sum_f = torch.sum(f)
        pdf = f / (sum_f * self.Δx)
        return pdf


class LogScale:
    def __call__(self, f):
        # Apply the log rescaling transformation
        f_log_scale = torch.log(f)
        return f_log_scale
