import torch
import torch.nn as nn

from .modules import Entmax, Softmax


class TEDE(nn.Module):
    def __init__(self,
             n_sources: int,
             output_dim: int,
             activation: str,
             d_model: int,
             nhead: int,
             num_encoder_layers: int,
             dim_feedforward: int,
             dropout: float,
             temperature: float,
             bin_size: float
        ):
        super(TEDE, self).__init__()
        self.bin_size = bin_size

        self.param_emb_embedding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.source_type_embedding = nn.Embedding(n_sources, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=activation
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 4, output_dim // 2), # * 4 stands for the number of parameters and source type
            nn.LayerNorm(output_dim // 2),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(output_dim // 2, output_dim),
            Softmax(temperature=temperature, dim=1)
        )

    def forward(self, params, source_types):
        params = params.unsqueeze(2) # [B, param_dim] -> [B, param_dim, 1]
        param_emb = self.param_emb_embedding(params) # [B, param_dim, 1] -> [B, param_dim, d_model]
        source_type_emb = self.source_type_embedding(source_types) #  [B, 1] -> [B, 1, d_model]
        embs_cat = torch.cat((param_emb, source_type_emb), dim=1) # [B, param_dim+1, d_model]
        x = self.transformer_encoder(embs_cat) # [B, param_dim+1, d_model]
        spectra_pdf = self.regression_head(x) / self.bin_size # [B, output_dim]
        return spectra_pdf
