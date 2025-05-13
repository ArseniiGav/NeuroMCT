"""
Transformer Encoder Density Estimator (TEDE) model implementation.

This module implements a transformer-based model for energy spectrum estimation.
The model uses a transformer encoder architecture to process LS parameters
and source types to predict energy spectrum probability densities.
"""

import torch
import torch.nn as nn

from .modules import Softmax


class TEDE(nn.Module):
    """
    Transformer Encoder Density Estimator (TEDE) model.

    This model implements a transformer encoder-based architecture for energy spectrum estimation.
    It processes LS parameters and source types through a transformer encoder
    to predict the probability density function of the energy spectrum.

    The model architecture consists of:
    1. Input embeddings for parameters and source types
    2. A transformer encoder for feature processing
    3. Regression head for final bin-to-bin density estimation

    Parameters
    ----------
    n_sources : int
        Number of different calibration source types.
    output_dim : int
        Number of bins in the energy spectrum.
    activation : str
        Activation function for the transformer layers. 
        Options: 'relu', 'gelu'.
    d_model : int
        Dimension of the transformer model's internal representation.
    nhead : int
        Number of attention heads in the transformer encoder.
    num_encoder_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int
        Dimension of the feedforward network in transformer layers.
    dropout : float
        Dropout rate for regularization in the transformer layers.
    temperature : float
        Temperature parameter for Softmax scaling.
    bin_size : torch.Tensor
        Bin size of the energy spectrum.
    """
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
             bin_size: torch.Tensor
        ):
        super(TEDE, self).__init__()
        self.bin_size = bin_size
        self.d_model = d_model

        # self.param_emb_embedding = nn.Sequential(
        #     nn.Linear(1, d_model // 4),
        #     #nn.LayerNorm(d_model // 4),
        #     nn.ReLU() if activation == 'relu' else nn.GELU(),
        #     nn.Linear(d_model // 4, d_model // 2),
        #     #nn.LayerNorm(d_model // 2),
        #     nn.ReLU() if activation == 'relu' else nn.GELU(),
        #     nn.Linear(d_model // 2, d_model),
        # )
        self.param_emb_embedding = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(d_model, d_model * 3),
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
            nn.Linear(d_model * 4, output_dim), # * 4 stands for the number of parameters and source type
            nn.LayerNorm(output_dim),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(output_dim, output_dim),
            Softmax(temperature=temperature, dim=1)
        )

    def forward(self, params, source_types):
        """
        Forward pass through the TEDE model.

        Parameters
        ----------
        params : torch.Tensor
            LS parameters of shape [batch_size, param_dim].
        source_types : torch.Tensor
            Source type indices of shape [batch_size].

        Returns
        -------
        torch.Tensor
            Predicted PDF encoded in output_dim number of bins: [batch_size, output_dim].

        Notes
        -----
        The forward pass consists of the following steps:
        1. Embed source types and parameters
        2. Concatenate embeddings for transformer input
        3. Process through transformer encoder
        4. Project to output dimension
        5. Apply temperature-scaled Softmax for normalization
        """
        param_emb = self.param_emb_embedding(params)  # [B, param_dim] -> [B, param_dim*d_model]
        param_emb = param_emb.reshape(-1, params.shape[1], self.d_model)  # [B, param_dim*d_model] -> [B, param_dim, d_model]
        source_type_emb = self.source_type_embedding(source_types) #  [B, 1] -> [B, 1, d_model]
        embs_cat = torch.cat((param_emb, source_type_emb), dim=1) # [B, param_dim+1, d_model]
        x = self.transformer_encoder(embs_cat) # [B, param_dim+1, d_model]
        spectra_pdf = self.regression_head(x) / self.bin_size # [B, output_dim]
        return spectra_pdf
