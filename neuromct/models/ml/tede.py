import torch
import torch.nn as nn


class TEDE(nn.Module):
    def __init__(self,
             param_dim: int,
             n_sources: int,
             output_dim: int,
             d_model: int,
             nhead: int,
             num_encoder_layers: int,
             dim_feedforward: int,
             dropout: float,
        ):
        super(TEDE, self).__init__()

        self.param_emb_embedding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        self.source_type_embedding = nn.Embedding(
            n_sources, d_model
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_encoder_layers
        )
        
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 4, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.Softmax(dim=1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.orthogonal_(module.weight)

    def forward(self, params, source_types):
        params = params.unsqueeze(2) # [B, param_dim] -> [B, param_dim, 1]
        param_emb = self.param_emb_embedding(params) # [B, param_dim, 1] -> [B, param_dim, d_model]
        source_type_emb = self.source_type_embedding(source_types) #  [B, 1] -> [B, 1, d_model]
        x = torch.cat((param_emb, source_type_emb), dim=1) # [B, param_dim+1, d_model]
        x = self.transformer_encoder(x) # [B, param_dim+1, d_model]
        spectra_pdf = self.regression_head(x) # [B, output_dim]
        return spectra_pdf
