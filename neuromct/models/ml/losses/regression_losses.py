import torch
import torch.nn as nn
import torch.nn.functional as F


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
