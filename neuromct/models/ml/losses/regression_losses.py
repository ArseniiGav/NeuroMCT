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

class GeneralizedKLDivLoss(nn.Module):
    def __init__(self, log_input, log_target, reduction, eps=1e-9):
        """
        Initialize the KL-divergence loss.
        :param reduction: Specifies the reduction to apply to the output.
                          Options: 'batchmean', 'sum', 'mean', or 'none'.
                log: Specifies the scale of predicted input.
        """
        super(GeneralizedKLDivLoss, self).__init__()
        self.eps = eps
        self.log_input = log_input
        self.kl_div = nn.KLDivLoss(reduction=reduction, log_target=log_target)

    def forward(self, predicted, target):
        """
        Compute the KL-divergence loss.
        :param predicted: Predicted probabilities (Q).
        :param target: Target probabilities (P).
        :return: KL-divergence loss.
        """
        if self.log_input:
            kl_div_loss = self.kl_div(predicted, target)
        else:
            kl_div_loss = self.kl_div(torch.log(predicted + self.eps), target)
        return kl_div_loss
