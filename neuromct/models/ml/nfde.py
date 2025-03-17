"""
Normalizing Flows Density Estimator (NFDE) model implementation.

This module implements a conditional normalizing flow model for energy spectrum estimation.
The model uses a sequence of planar or radial flows to transform a complex spectrum to a known base distribution,
conditioned on LS parameters and source types.
"""

import math 

import torch
import torch.nn as nn


class Flow(nn.Module):
    """
    Normalizing flow module implementing planar and radial flows.

    This class constructs a conditional normalizing flow that can be configured as either planar or radial.
    It leverages a FCNN to parametrize the flow's parameters with the input conditions. 
    As input conditions it takes: the LS parameters and the calibration source type.

    Parameters
    ----------
    n_conditions : int
        The number of conditions (LS parameters).
    n_sources : int
        The number of different calibration sources.
    n_units : int
        The number of hidden units in the intermediate layers of the FCNN.
    activation : str
        The activation function to use in the FCNN. 
        Options: 'relu', 'gelu', 'tanh', 'silu'.
    flow_type : str
        The type of flow to use. 
        Options: 'planar' or 'radial'.
    dropout : float
        Dropout rate for regularization.
    """
    def __init__(self,
            n_conditions: int,
            n_sources: int,
            n_units: int,
            activation: str,
            flow_type: str, 
            dropout: float,
        ):
        super().__init__()
        self.flow_type = flow_type

        if activation == 'relu':
            activation_func = nn.ReLU()
        elif activation == 'gelu':  
            activation_func = nn.GELU()
        elif activation == 'tanh':
            activation_func = nn.Tanh()
        elif activation == 'silu':
            activation_func = nn.SiLU()
        else:
            raise ValueError(f'''Unknown activation function: {activation}. 
                             Choose from ['relu', 'gelu', 'tanh', 'silu']''')
        
        n_params = n_conditions - 1
        self.param_net = nn.Sequential(
            nn.Linear(n_params, n_units),    
            activation_func,
            nn.Dropout(dropout),
            nn.Linear(n_units, n_units),
            activation_func,
            nn.Dropout(dropout),
            nn.Linear(n_units, n_units),
        )
        self.source_type_embedding = nn.Embedding(n_sources, n_units)

        n_units_combined = n_units * 2
        self.conditions_to_params_net = nn.Sequential(
            nn.Linear(n_units_combined, n_units_combined),
            activation_func,
            nn.Dropout(dropout),
            nn.Linear(n_units_combined, n_units_combined // 2),
            activation_func,
            nn.Dropout(dropout),
            nn.Linear(n_units_combined // 2, 3),
        )

    def _get_u_hat(self, 
                   u: torch.Tensor, 
                   w: torch.Tensor
        ) -> torch.Tensor:
        """
        Adjusts u to ensure invertibility of the planar flow transformation.
        Computes u_hat = u + ((m - u * w) * w) / ||w||^2,
        where m = -1 + log(1 + exp(u * w)).

        Parameters
        ----------
        u (torch.Tensor):
            The current value of u.
        w (torch.Tensor):
            The current value of w.

        Returns
        -------
        u_hat (torch.Tensor):
            The adjusted version of u.
        """
        wu = u * w
        m_wu = -1 + torch.log(1 + torch.exp(wu))
        u_hat = u + (m_wu - wu) * w / torch.norm(w, p=2) ** 2
        return u_hat

    def forward(self, 
                x: torch.Tensor, 
                params: torch.Tensor, 
                source_types: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the normalizing flow. Normalizing direction.

        As input takes the values x, the corresponding LS parameters, 
        and the type of the calibration source.

        Returns the transformed vector z and the logarithm of the absolute value 
        of the determinant of the Jacobian of the transformation.

        Parameters
        ----------
        x: (torch.Tensor): 
            The 1d vector with the x values. 
            To be moved further in the normalizing direction.
            Shape: [n_x_values].
        params (torch.Tensor): 
            The 1d vector containing LS parameters. 
            Shape: [param_dim].
        source_types (torch.Tensor): 
            The 1d vector containing the corresponding source type. 
            Shape: [1].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]: 
        z (torch.Tensor): 
            The 1d vector transformed in the normalizing direction by the flow. 
            Shape: [n_x_values].
        log_det_jacobian (torch.Tensor): 
            Logarithm of the absolute value of the determinant of the Jacobian of the transformation. 
            Shape: [n_x_values].
        """
        params_emb = self.param_net(params) # [param_dim] -> [n_units]
        source_types_emb = self.source_type_embedding(source_types) # [1] -> [1, n_units]
        source_types_emb = source_types_emb.squeeze(0) # [1, n_units] -> [n_units]
        input_emb_cat = torch.cat([params_emb, source_types_emb], dim=0) # [n_units * 2]
        flow_params = self.conditions_to_params_net(input_emb_cat) # [n_units * 2] -> [3]
        
        if self.flow_type == 'planar':
            w = torch.tanh(flow_params[0])
            u = torch.tanh(flow_params[1])
            b = flow_params[2]
    
            if u * w < -1: 
                u = self._get_u_hat(u, w)

            m = x * w + b
            h = torch.tanh(m)
            z = x + u * h
            
            abs_det_jacobian = (1 + u * (1 - h**2) * w).abs()
            log_det_jacobian = torch.log(1e-10 + abs_det_jacobian)

        elif self.flow_type == 'radial':
            α = torch.log(torch.exp(flow_params[0]) + 1)
            β = torch.exp(flow_params[1]) - 1
            γ = flow_params[2]

            r = x - γ
            z = x + α * β * r / (α + torch.abs(r))

            abs_det_jacobian = (1 + (α**2 * β) / (α + torch.abs(r))**2).abs()
            log_det_jacobian = torch.log(1e-10 + abs_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, 
            z: torch.Tensor, 
            params: torch.Tensor, 
            source_types: torch.Tensor, 
            max_iters: int = 50
        ) -> torch.Tensor:
        """
        Inverse pass of the normalizing flow. Generative direction.
        Computes the inverse transformation of the input z via iterative fixed-point updates.

        As input takes the transformed values z, the corresponding LS parameters, 
        and the type of the calibration source.

        Returns the inverse transformaition of the vector z: a vector x.
        
        Parameters
        ----------
        z: (torch.Tensor): 
            The 1d vector with the z values. 
            To be moved further in the generative direction.
            Shape: [n_x_values].
        params (torch.Tensor): 
            The 1d vector containing LS parameters. 
            Shape: [param_dim].
        source_types (torch.Tensor): 
            The 1d vector containing the corresponding source type. 
            Shape: [1].
        max_iters (int), optional:
            Maximum number of iterations for the fixed-point update (default is 50).

        Returns
        -------
        x (torch.Tensor): 
            The estimated input tensor x that approximates the inverse transformation of z.
            Shape: [n_x_values].
        """
        x0 = torch.rand(z.shape, device=z.device, requires_grad=True)

        params_emb = self.param_net(params) # [param_dim] -> [n_units]
        source_types_emb = self.source_type_embedding(source_types) # [1] -> [1, n_units]
        source_types_emb = source_types_emb.squeeze(0) # [1, n_units] -> [n_units]
        input_emb_cat = torch.cat([params_emb, source_types_emb], dim=0) # [n_units * 2]
        flow_params = self.conditions_to_params_net(input_emb_cat) # [n_units * 2] -> [3]

        if self.flow_type == 'planar':
            w = torch.tanh(flow_params[0])
            u = torch.tanh(flow_params[1])
            b = flow_params[2]

            if u * w < -1: 
                u = self._get_u_hat(u, w)

            for _ in range(max_iters): 
                z0, _ = self.forward(x0, params, source_types)

                m = x0 * w + b
                f_prime = u * (1 - torch.tanh(m)**2) * w
                f_prime_inverse = 1 / (1 + f_prime)
                x0 = x0 + (z - z0) * f_prime_inverse

        elif self.flow_type == 'radial':
            α = torch.log(torch.exp(flow_params[0]) + 1)
            β = torch.exp(flow_params[1]) - 1
            γ = flow_params[2]

            for _ in range(max_iters):
                z0, _ = self.forward(x0, params, source_types)

                r = x0 - γ          
                f_prime = 1 + (α**2 * β) / (α + torch.abs(r))**2
                f_prime_inverse = 1 / f_prime
                x0 = x0 + (z - z0) * f_prime_inverse
        return x0


class NFDE(nn.Module):
    """
    Normalizing Flows Density Estimator (NFDE) model.

    This model implements a sequence of normalizing flows for energy spectrum estimation.
    It uses multiple flows (planar or radial) to transform a complex energy spectrum into a known base distribution,
    conditioned on LS parameters and source types.

    Each flow is conditioned on the inputs through a parameter network and source type embeddings.
    The forward pass is used during training to compute log-probabilities (hence to estimate the density),
    while the inverse pass is used for generating energies.

    Parameters
    ----------
    n_flows : int
        Number of flows in the model.
    n_conditions : int
        Number of input parameters.
    n_sources : int
        Number of different calibration source types.
    n_units : int
        Number of hidden units in each flow's parameter neural network.
    activation : str
        Activation function for the flow's parameter neural networks. 
        Options: 'relu', 'gelu', 'tanh', 'silu'.
    flow_type : str
        Type of normalizing flow to use. 
        Options: 'planar' or 'radial'.
    dropout : float
        Dropout rate for regularization in the flow's parameter neural networks.
    """

    def __init__(self, 
            n_flows: int,
            n_conditions: int,
            n_sources: int,
            n_units: int,
            activation: str,
            flow_type: str,
            dropout: float
        ):
        super().__init__()
        self.flows = self._flows_block(
            n_flows, n_conditions, n_sources, n_units, 
            activation, flow_type, dropout)
        self.pi = torch.tensor(math.pi, dtype=torch.float64)

    def _flows_block(self, 
            n_flows: int, 
            n_conditions: int, 
            n_sources: int, 
            n_units: int, 
            activation: str, 
            flow_type: str,
            dropout: float
        ) -> nn.ModuleList:
        """
        Create a sequence of normalizing flow layers.

        Parameters
        ----------
        n_flows : int
            Number of flow layers.
        n_conditions : int
            Number of the input parameters.
        n_sources : int
            Number of the calibration source types.
        n_units : int
            Number of hidden units in each flow's neural network.
        activation : str
            Activation function for the flow's parameter neural networks.
        flow_type : str
            Type of the normalizing flow.
        dropout : float
            Dropout rate.

        Returns
        -------
        nn.ModuleList
            List of the normalizing flows.
        """
        return nn.ModuleList([
            Flow(n_conditions, n_sources, n_units, activation, flow_type, dropout)
            for _ in range(n_flows)
        ])

    def _log_prob_comp(
            self, 
            x: torch.Tensor, 
            params: torch.Tensor, 
            source_types: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helper method to compute log-probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with energy values of shape [n_en_values].
        params : torch.Tensor
            Input tensor of the LS parameters of shape [param_dim].
        source_types : torch.Tensor
            Input tensor of the source type of shape [1].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - Log-probabilities of the base distribution
            - Sum of log-dets of Jacobians of the transformations
        """
        z, log_det_jacobian = self.forward(x, params, source_types)
        base_log_prob = -0.5 * (z ** 2 + torch.log(2 * self.pi))
        return base_log_prob, log_det_jacobian

    def forward(
            self, 
            x: torch.Tensor, 
            params: torch.Tensor, 
            source_types: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the NFDE model (normalizing direction).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of with energy values of shape [n_en_values].
        params : torch.Tensor
            Input tensor of the LS parameters of shape [param_dim].
        source_types : torch.Tensor
            Input tensor of the source type of shape [1].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - Transformed values z
            - Sum of log-dets of Jacobians of the transformations
        """
        log_det_sum = torch.zeros_like(x)
        for flow in self.flows:
            x, log_det = flow(x, params, source_types)
            log_det_sum += log_det
        z = x
        return z, log_det_sum

    def inverse(self, z: torch.Tensor, params: torch.Tensor, source_types: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass through the NFDE model (generative direction).

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of with transformed values z of shape [n_z_values].
        params : torch.Tensor
            Input tensor of the LS parameters of shape [param_dim].
        source_types : torch.Tensor
            Input tensor of the source type of shape [1].

        Returns
        -------
        torch.Tensor
            Generated samples x under the input conditions.
        """
        for flow in self.flows[::-1]:
            z = flow.inverse(z, params, source_types)
        x = z
        return x

    def log_prob(
            self, 
            x: torch.Tensor, 
            params: torch.Tensor, 
            source_types: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute log-probabilities of input x under the input conditions.
        Density estimation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of with energy values of shape [n_en_values].
        params : torch.Tensor
            Input tensor of the LS parameters of shape [param_dim].
        source_types : torch.Tensor
            Input tensor of the source type of shape [1].

        Returns
        -------
        torch.Tensor
            Log-probabilities of input x under the input conditions.
        """
        base_log_prob, log_det_jacobian = self._log_prob_comp(x, params, source_types)
        return base_log_prob + log_det_jacobian

    def generate_energies(
            self, 
            n_en_values: int, 
            params: torch.Tensor, 
            source_types: torch.Tensor
        ) -> torch.Tensor:
        """
        Generate energy samples from the model.

        Parameters
        ----------
        n_en_values : int
            Number of energy values to generate.
        params : torch.Tensor
            Light source parameters of shape [n_conditions].
        source_types : torch.Tensor
            Source type indices of shape [1].

        Returns
        -------
        torch.Tensor
            Generated energy samples of shape [n_en_values].
        """
        z = torch.randn(n_en_values, device=params.device)
        return self.inverse(z, params, source_types)
