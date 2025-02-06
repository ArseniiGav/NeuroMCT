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
