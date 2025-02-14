import torch.nn as nn
import torch.nn.functional as F

from entmax import entmax_bisect, entmax15


class Entmax(nn.Module):
    def __init__(self, alpha=1.25, dim=-1):
        super(Entmax, self).__init__()
        self.alpha = alpha
        self.dim = dim

    def forward(self, input):
        if self.alpha == 1.0:
            return F.softmax(input, dim=self.dim)
        elif self.alpha == 1.5:
            return entmax15(input, dim=self.dim)
        else:
            return entmax_bisect(
                input, alpha=self.alpha, 
                dim=self.dim, ensure_sum_one=True,
                n_iter=100)


class Softmax(nn.Module):
    """
    Applies the softmax function with temperature scaling.
    
    Args:
        temperature (float): The temperature to scale the logits. Defaults to 1.0.
        dim (int): The dimension along which softmax will be computed. Defaults to -1.
    """
    def __init__(self, temperature=1.0, dim=-1):
        super(Softmax, self).__init__()
        self.temperature = temperature
        self.dim = dim

    def forward(self, input):
        if self.temperature == 0:
            raise ValueError("Temperature must be non-zero.")
        return F.softmax(input / self.temperature, dim=self.dim)
