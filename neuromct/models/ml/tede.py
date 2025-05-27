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
    params_dim : int
        Number of LS parameters: kB, fC, Y.
    output_dim : int
        Number of bins in the energy spectrum.
    activation : str
        Activation function for the transformer layers. 
        Options: 'relu', 'gelu'.
    n_tokens_per_param : int
        Number of tokens per LS parameter.
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
             params_dim: int,
             output_dim: int,
             activation: str,
             n_tokens_per_param: int,
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
        self.params_dim = params_dim
        self.n_tokens_per_param = n_tokens_per_param
        self.d_model = d_model

        emb_output_dim = d_model * n_tokens_per_param * params_dim
        self.param_emb_embedding = nn.Sequential(
            nn.Linear(params_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(d_model, emb_output_dim),
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
            nn.Linear(emb_output_dim + d_model, output_dim),
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
        4. Map the transformer output to the output (number of bins) dimension
        5. Apply temperature-scaled Softmax for normalization and divide by bin size to get PDF
        """
         # [B, params_dim] -> [B, params_dim*n_tokens_per_param*d_model]
        param_emb = self.param_emb_embedding(params)

        # [B, params_dim*n_tokens_per_param*d_model] -> [B, params_dim*n_tokens_per_param, d_model]
        param_emb = param_emb.reshape(-1, self.params_dim * self.n_tokens_per_param, self.d_model) 

        # [B, 1] -> [B, 1, d_model]
        source_type_emb = self.source_type_embedding(source_types)

        # [B, params_dim*n_tokens_per_param+1, d_model]
        embs_cat = torch.cat((param_emb, source_type_emb), dim=1)

        # [B, params_dim*n_tokens_per_param+1, d_model]
        x = self.transformer_encoder(embs_cat)

        # [B, params_dim*n_tokens_per_param+1, d_model] -> [B, output_dim]
        spectra_pdf = self.regression_head(x) / self.bin_size
        return spectra_pdf
