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

        self.param_emb_layer = nn.Linear(param_dim, d_model // 2)
        self.source_type_emb_layer = nn.Embedding(n_sources, d_model // 2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
            nn.Softmax()
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
        param_emb = self.param_emb_layer(params)
        source_type_emb = self.source_type_emb_layer(source_types)
        x = torch.cat([param_emb, source_type_emb], dim=1)
        x = self.transformer_encoder(x)
        spectra = self.regression_head(x)
        return spectra
