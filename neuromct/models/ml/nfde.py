import math 

import torch
import torch.nn as nn


class Flow(nn.Module):
    """
    Normalizing flow module implementing planar and radial flows.

    This class constructs a conditional normalizing flow that can be configured as either planar or radial one.
    It leverages a FCNN to parametrize the flow's parameters with the input conditions. 
    As input conditions it takes: the LS parameters and the calibration source type.

    Parameters
    ----------
    n_conditions (int):
        The number of conditions.
    n_sources (int): 
        The number of different calibration sources.
    n_units (int):
        The number of hidden units in the intermediate layers of the FCNN.
    activation (str):
        The activation function to use in the FCNN. Options include: 'relu', 'gelu', 'tanh', 'silu'.
    flow_type (str): 
        The type of flow to use. Options are 'planar' and 'radial'.
    """
    def __init__(self,
            n_conditions: int,
            n_sources: int,
            n_units: int,
            activation: str,
            flow_type: str, 
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
            nn.Linear(n_units, n_units),
            activation_func,
            nn.Linear(n_units, n_units),
        )
        self.source_type_embedding = nn.Embedding(n_sources, n_units)

        n_units_combined = n_units * 2
        self.conditions_to_params_net = nn.Sequential(
            nn.Linear(n_units_combined, n_units_combined),
            activation_func,
            nn.Linear(n_units_combined, n_units_combined // 2),
            activation_func,
            nn.Linear(n_units_combined // 2, 3),
        )

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
            self.w = torch.tanh(flow_params[0])
            self.u = torch.tanh(flow_params[1])
            self.b = flow_params[2]
    
            m = x * self.w + self.b
            h = torch.tanh(m)
            z = x + self.u * h
            
            mask = self.u * self.w < -1
            if (self.u * self.w)[mask].shape[0] != 0:
                self.u = self._get_u_hat(mask)
            abs_det_jacobian = (1 + self.u * (1 - h**2) * self.w).abs()
            log_det_jacobian = torch.log(1e-10 + abs_det_jacobian)
        elif self.flow_type == 'radial':
            self.α = torch.log(torch.exp(flow_params[0]) + 1)
            self.β = torch.exp(flow_params[1]) - 1
            self.γ = flow_params[2]

            r = x - self.γ
            z = x + self.α * self.β * r / (self.α + torch.abs(r))

            det_jacobian = 1 + (self.α**2 * self.β) / (self.α + torch.abs(r))**2
            log_det_jacobian = torch.log(1e-10 + det_jacobian)
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
            self.w = torch.tanh(flow_params[0])
            self.u = torch.tanh(flow_params[1])
            self.b = flow_params[2]

            for _ in range(max_iters): 
                z0, _ = self.forward(x0, params, source_types)

                m = x0 * self.w + self.b
                p = self.u * (1 - torch.tanh(m)**2) * self.w
                f_prime_inverse = 1 - p / (1 + p)
                x0 = x0 + (z - z0) * f_prime_inverse

        elif self.flow_type == 'radial':
            self.α = torch.log(torch.exp(flow_params[0]) + 1)
            self.β = torch.exp(flow_params[1]) - 1
            self.γ = flow_params[2]

            for _ in range(max_iters):
                z0, _ = self.forward(x0, params, source_types)

                r = x0 - self.γ          
                f_prime = x0 + (self.α * self.β * r) / (self.α + torch.abs(r))
                f_prime_inverse = 1 / f_prime
                x0 = x0 + (z - z0) * f_prime_inverse
        return x0

    def _get_u_hat(self, mask):
        """
        Adjusts the vector u to ensure invertibility of the planar flow transformation.
        When the product u * w falls below a threshold (i.e., less than -1),
        this helper method computes an adjusted version, u_hat.

        Parameters
        ----------
        mask (torch.Tensor):
            A boolean tensor indicating the indices where u * w is less than -1.

        Returns
        -------
        u_hat (torch.Tensor):
            The adjusted version of u (u_hat).
        """
        wu = (self.u * self.w)[mask]
        m_wu = -1 + torch.log(1 + torch.exp(wu))
        u_hat = self.u.clone()
        u_hat[mask] = (
            self.u[mask] + (m_wu - wu) * self.w[mask] / torch.norm(self.w[mask], p=2) ** 2
        )
        return u_hat


class NFDE(nn.Module):
    def __init__(self, 
            n_flows: int,
            n_conditions: int,
            n_sources: int,
            n_units: int,
            activation: str,
            flow_type: str,
            base_type: str,
        ):
        super(NFDE, self).__init__()

        self.flows = self._flows_block(
            n_flows, n_conditions, n_sources, 
            n_units, activation, flow_type
        )
        self.base_type = base_type
        self.pi = torch.tensor(math.pi, dtype=torch.float32)
        if self.base_type == 'uniform':
            self.lb = 0.0
            self.rb = 20.0

    def _flows_block(self, 
            n_flows: int, 
            n_conditions: int, 
            n_sources: int, 
            n_units: int, 
            activation: str, 
            flow_type: str
        ):
        flows_block_list = []
        for _ in range(n_flows):
            flows_block_list.append(
                Flow(n_conditions, n_sources, 
                     n_units, activation, flow_type)
            )
        flows_block_module_list = nn.ModuleList(flows_block_list)
        return flows_block_module_list

    def _log_prob_comp(self, x, params, source_types):
        z, log_det_jacobian = self.forward(x, params, source_types)
        if self.base_type == 'uniform':
            within_bounds = (z >= self.lb) & (z <= self.rb)
            base_log_prob = torch.where(
                within_bounds,
                torch.full_like(z, -torch.log(self.rb - self.lb)),
                torch.full_like(z, self.lb)
            )
        elif self.base_type == 'normal':
            base_log_prob = -0.5 * (z ** 2 + torch.log(2 * self.pi))
        elif self.base_type == 'lognormal':
            positive_mask = z > 0
            log_z = torch.log(z.clamp(min=1e-10))

            normalization_term = -torch.log(z * (2 * self.pi)**0.5)
            exponent_term = -(log_z ** 2) / 2 
    
            base_log_prob = normalization_term + exponent_term
            base_log_prob = torch.where(positive_mask, base_log_prob, torch.tensor(0.0))
        return base_log_prob, log_det_jacobian

    def forward(self, x, params, source_types):
        log_det_sum = torch.zeros_like(x)
        for flow in self.flows:
            x, log_det = flow(x, params, source_types)
            log_det_sum += log_det
        z = x
        return z, log_det_sum
        
    def inverse(self, z, params, source_types):
        for flow in self.flows[::-1]:
            z = flow.inverse(z, params, source_types)
        x = z
        return x

    def log_prob(self, x, params, source_types):
        base_log_prob, log_det_jacobian = self._log_prob_comp(x, params, source_types)
        return base_log_prob + log_det_jacobian

    def generate_energies(self, n_en_values, params, source_types):
        self.eval()
        with torch.no_grad():
            if self.base_type == 'uniform':
                z = torch.rand(n_en_values).to(params.device) * (self.rb - self.lb) + self.lb
            elif self.base_type == 'normal':
                z = torch.randn(n_en_values).to(params.device)
            elif self.base_type == 'lognormal':
                z = torch.exp(torch.randn(n_en_values).to(params.device))
            x = self.inverse(z, params, source_types)
        self.train()
        return x
