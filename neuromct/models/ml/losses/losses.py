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
    def __init__(self, log_input, log_target, reduction):
        """
        Generalized Poisson Negative Log Likelihood Loss.
        Args:
            log_input (bool): If True, y_pred is expected to be in log-space.
            log_target (bool): If True, y_true is expected to be in log-space 
                               (not supported by nn.PoissonNLLLoss, so will be converted).
            reduction (str): Specifies the reduction to apply to the output.
                             Options: 'sum', 'mean', or 'none'.
        """
        super(GeneralizedPoissonNLLLoss, self).__init__()
        self.log_input = log_input
        self.log_target = log_target
        self.poisson_nll = nn.PoissonNLLLoss(
            log_input=True, reduction=reduction)

    def forward(self, y_pred, y_true):
        if not self.log_input:
            y_pred = torch.log(y_pred)
        if self.log_target:
            y_true = torch.exp(y_true)
        return self.poisson_nll(y_pred, y_true)


class GeneralizedKLDivLoss(nn.Module):
    def __init__(self, log_input, log_target, reduction):
        """
        Initialize the KL-divergence loss.
        :param reduction: Specifies the reduction to apply to the output.
                          Options: 'batchmean', 'sum', 'mean', or 'none'.
                log: Specifies the scale of predicted input.
        """
        super(GeneralizedKLDivLoss, self).__init__()
        self.log_input = log_input
        self.kl_div = nn.KLDivLoss(
            reduction=reduction, log_target=log_target)

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
            kl_div_loss = self.kl_div(torch.log(predicted), target)
        return kl_div_loss
