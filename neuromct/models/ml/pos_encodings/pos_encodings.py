import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 max_len: int):
        super(PositionalEncoding, self).__init__()
        self.emb_size = emb_size

        den = torch.exp(- torch.arange(0, emb_size, 2) * torch.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, tgt):
        return self.dropout(tgt * torch.sqrt(self.emb_size) + self.pos_embedding[:tgt.size(0), :])


class PositionalEncodingByDistance(nn.Module):
    def __init__(self,
                 output_dim: int,
                 d_model: int,
                 energy_bin_centers: float):
        super(PositionalEncodingByDistance, self).__init__()
        self.output_dim = output_dim
        self.d_model = d_model
        
        distance_matrix = torch.abs(energy_bin_centers.unsqueeze(1) - energy_bin_centers.unsqueeze(0))
        self.register_buffer("distance_matrix", distance_matrix)  # Save as a constant
        
        self.distance_to_encoding = nn.Linear(1, d_model)
        
    def forward(self):
        distance_encoding = self.distance_to_encoding(self.distance_matrix.unsqueeze(-1))  # Shape: (output_dim, output_dim, d_model)
        return distance_encoding
