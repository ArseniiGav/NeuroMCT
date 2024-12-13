import torch
import torch.nn as nn
import torch.nn.functional as F
from neuromct.configs import data_configs
from ot.lp import wasserstein_1d


class CosineDistanceLoss(nn.Module):
    def __init__(self):
        super(CosineDistanceLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        cos_sim = F.cosine_similarity(y_pred, y_true, dim=-1)
        return 1 - cos_sim.mean()


class GeneralizedPoissonNLLLoss(nn.Module):
    def __init__(self, log=False):
        super(GeneralizedPoissonNLLLoss, self).__init__()
        self.log = log
    
    def forward(self, y_pred, y_true):
        if self.log:
            output = torch.exp(y_pred) - 1 - (torch.exp(y_true) - 1) * (torch.log(torch.exp(y_pred) - 1 + 1e-08))
            output = output.mean()
        else:
            output = nn.PoissonNLLLoss(log_input=False)
        return output


class WassersteinLoss(nn.Module):
    def __init__(self, hist_based=False):
        super(WassersteinLoss, self).__init__()
        self.hist_based = hist_based
        if self.hist_based:
            kNPE_bins_edges = torch.tensor(
                data_configs["kNPE_bins_edges"], dtype=torch.float64)
            self.kNPE_bins_centers = (kNPE_bins_edges[1:] + kNPE_bins_edges[:-1]) / 2

    def forward(self, spectra_predict, spectra_true):
        # Verify data consistency
        if spectra_predict.shape[0] != spectra_true.shape[0]:
            raise ValueError("Mismatch in the batch sizes between predicted and true spectra.")
        elif spectra_predict.shape[1] != spectra_true.shape[1]:
            raise ValueError("Mismatch in the binning sizes between predicted and true spectra.")

        original_deterministic_setting = torch.are_deterministic_algorithms_enabled()
        if self.hist_based:
            batch_size = spectra_true.shape[0]
            self.kNPE_bins_centers_repeated = self.kNPE_bins_centers.repeat((batch_size, 1)).T

            #torch.cumsum used in wasserstein_1d from POT does not have a deterministic implementation            
            torch.use_deterministic_algorithms(False)
            loss = wasserstein_1d(
                self.kNPE_bins_centers_repeated.to(spectra_predict.device),
                self.kNPE_bins_centers_repeated.to(spectra_predict.device),
                spectra_predict.T,
                spectra_true.T
            )
            torch.use_deterministic_algorithms(original_deterministic_setting)
            return torch.mean(loss)
        else:
            #torch.cumsum used in wasserstein_1d from POT does not have a deterministic implementation
            torch.use_deterministic_algorithms(False)
            loss = wasserstein_1d(
                spectra_predict.T,
                spectra_true.T
            )
            torch.use_deterministic_algorithms(original_deterministic_setting)
            return torch.mean(loss)
