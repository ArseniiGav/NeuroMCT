import torch.nn as nn
import torch.nn.functional as F


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
