import torch
import torch.nn as nn


class TransformerRegressor(nn.Module):
    def __init__(self,
             output_dim: int,
             param_dim: int,
             n_sources: int,
             d_model: int,
             nhead: int,
             num_encoder_layers: int,
             dim_feedforward: int,
             dropout: float,
        ):
        super(TransformerRegressor, self).__init__()

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
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x):
        param_emb = self.param_emb_layer(x[:, :self.param_dim])
        source_type_emb = self.source_type_emb_layer(x[:, self.param_dim])
        x = torch.cat([param_emb, source_type_emb], dim=1)
        x = self.transformer_encoder(x)
        x = self.regression_head(x)
        return x
