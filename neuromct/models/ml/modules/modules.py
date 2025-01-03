import torch.nn as nn
import torch.nn.functional as F


class TSoftmax(nn.Module):
    def __init__(self, temperature: float):
        super(TSoftmax, self).__init__()
        self.temperature = temperature

    def forward(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)
