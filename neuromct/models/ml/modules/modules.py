import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableTemperatureSoftmax(nn.Module):
    def __init__(self, initial_temperature=1.0, min_temperature=0.1, max_temperature=2.0):
        """
        Softmax with learnable temperature.

        Args:
            initial_temperature (float): Initial temperature value.
            min_temperature (float): Minimum temperature value for clamping.
            max_temperature (float): Maximum temperature value for clamping.
        """
        super(LearnableTemperatureSoftmax, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature, dtype=torch.float32))
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

    def forward(self, logits):
        """
        Forward pass for softmax with learnable temperature.

        Args:
            logits (Tensor): Raw model outputs (unnormalized scores), shape (batch_size, num_classes).
        
        Returns:
            Tensor: Probability distribution after applying softmax with learnable temperature.
        """
        # Clamp temperature to prevent it from becoming too small or too large
        temperature = torch.clamp(self.temperature, self.min_temperature, self.max_temperature)
        return F.softmax(logits / temperature, dim=-1)
