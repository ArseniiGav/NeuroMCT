import math 

import torch
import torch.nn as nn


class Flow(nn.Module):
    def __init__(self,
            n_conditions: int,
            n_sources: int,
            activation: str,
            flow_type: str
        ):
        super().__init__()
        self.activation = activation
        self.flow_type = flow_type

        n_params = n_conditions - 1
        self.param_net = nn.Sequential(
            nn.Linear(n_params, 10),
            nn.LayerNorm(10),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(10, 10),
            nn.LayerNorm(10),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(10, 10),
        )
        self.source_type_embedding = nn.Embedding(n_sources, 10)

        self.conditions_to_params_net = nn.Sequential(
                nn.Linear(20, 20),
                nn.LayerNorm(20),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Linear(20, 10),
                nn.LayerNorm(10),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Linear(10, 3),
            )

    def forward(self, x, params, source_types):
        x = x.squeeze(1)
        params_emb = self.param_net(params) # [B, param_dim] -> [B, 10]
        source_types = source_types.squeeze(1) # [B, 1] -> [B]
        source_types_emb = self.source_type_embedding(source_types) # [B] -> [B, 10]
        input_emb_cat = torch.cat([params_emb, source_types_emb], dim=1) # [B, 20]
        flow_params = self.conditions_to_params_net(input_emb_cat) # [B, 20] -> [B, 3]
        if self.flow_type == 'planar':
            self.w = torch.tanh(flow_params[:, 0])
            self.u = torch.tanh(flow_params[:, 1])
            self.b = flow_params[:, 2]
    
            m = x * self.w + self.b
            h = torch.tanh(m)
            z = x + self.u * h
            
            mask = self.u * self.w < -1
            if (self.u * self.w)[mask].shape[0] != 0:
                self.u = self._get_u_hat(mask)
            abs_det_jacobian = (1 + self.u * (1 - h**2) * self.w).abs()
            log_det_jacobian = torch.log(1e-10 + abs_det_jacobian)
        elif self.flow_type == 'radial':
            self.α = torch.log(torch.exp(flow_params[:, 0]) + 1)
            self.β = torch.exp(flow_params[:, 1]) - 1
            self.γ = flow_params[:, 2]

            r = x - self.γ
            z = x + self.α * self.β * r / (self.α + torch.abs(r))

            det_jacobian = 1 + (self.α**2 * self.β) / (self.α + torch.abs(r))**2
            log_det_jacobian = torch.log(1e-10 + det_jacobian)
        return z.unsqueeze(1), log_det_jacobian.unsqueeze(1)

    def inverse(self, z, params, source_types, max_iters=50):
        z = z.squeeze(1)
        x0 = torch.rand(z.shape, device=z.device, requires_grad=True)

        params_emb = self.param_net(params) # [B, param_dim] -> [B, 10]
        source_types = source_types.squeeze(1) # [B, 1] -> [B]
        source_types_emb = self.source_type_embedding(source_types) # [B] -> [B, 10]
        input_emb_cat = torch.cat([params_emb, source_types_emb], dim=1) # [B, 20]
        flow_params = self.conditions_to_params_net(input_emb_cat) # [B, 20] -> [B, 3]
        if self.flow_type == 'planar':
            self.w = torch.tanh(flow_params[:, 0])
            self.u = torch.tanh(flow_params[:, 1])
            self.b = flow_params[:, 2]

            for _ in range(max_iters): 
                z0, _ = self.forward(x0.unsqueeze(1), params, source_types)
                z0 = z0.squeeze(1)

                m = x0 * self.w + self.b
                p = self.u * (1 - torch.tanh(x0 * self.w + self.b)**2) * self.w
                f_prime_inverse = 1 - p / (1 + p)
                x0 = x0 + (z - z0) * f_prime_inverse

        elif self.flow_type == 'radial':
            self.α = torch.log(torch.exp(flow_params[:, 0]) + 1)
            self.β = torch.exp(flow_params[:, 1]) - 1
            self.γ = flow_params[:, 2]

            for _ in range(max_iters):
                z0, _ = self.forward(x0.unsqueeze(1), params, source_types)
                z0 = z0.squeeze(1)

                r = x0 - self.γ          
                f_prime = x0 + (self.α * self.β * r) / (self.α + torch.abs(r))
                f_prime_inverse = 1 / f_prime
                x0 = x0 + (z - z0) * f_prime_inverse
        return x0.unsqueeze(1)

    def _get_u_hat(self, mask):
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
            activation: str,
            n_conditions: int,
            base_type: str,
            flow_type: str,
        ):
        super(NFDE, self).__init__()

        self.flows = self._flows_block(
            n_flows, n_conditions, activation, flow_type)
        self.base_type = base_type
        self.pi = torch.tensor(math.pi)

    def _flows_block(self, n_flows, n_conditions, activation, flow_type):
        flows_block_list = []
        for _ in range(n_flows):
            flows_block_list.append(Flow(n_conditions, activation, flow_type))
        flows_block_module_list = nn.ModuleList(flows_block_list)
        return flows_block_module_list

    def _log_prob_comp(self, x, params, source_types):
        z, log_det_jacobian = self.forward(x, params, source_types)
        if self.base_type == 'uniform':
            within_bounds = (z >= 0.0) & (z <= 20.0)
            base_log_prob = torch.where(
                within_bounds,
                torch.full_like(z, -torch.log(torch.tensor(20.0, device=z.device))),
                torch.full_like(z, 0.0)
            )
            base_log_prob = base_log_prob.sum(dim=1)
        elif self.base_type == 'normal':
            base_log_prob = -0.5 * torch.sum(z ** 2 + torch.log(2 * self.pi), dim=1)
        elif self.base_type == 'lognormal':
            positive_mask = z > 0
            log_z = torch.log(z.clamp(min=1e-10))

            normalization_term = -torch.log(z * (2 * self.pi)**0.5)
            exponent_term = -(log_z ** 2) / 2 
    
            base_log_prob = normalization_term + exponent_term
            base_log_prob = torch.where(positive_mask, base_log_prob, torch.tensor(0.0))
            base_log_prob = base_log_prob.sum(dim=1)
        return base_log_prob, log_det_jacobian

    def forward(self, x, params, source_types):
        log_det_sum = torch.zeros_like(x[:, 0])
        for flow in self.flows:
            x, log_det = flow(x, params, source_types)
            log_det_sum += log_det.squeeze(1)
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

    def generate_energies(self, n_energies, params, source_types):
        with torch.no_grad():
            if self.base_type == 'uniform':
                z = torch.rand(n_energies, 1).to(params.device) * 20
            elif self.base_type == 'normal':
                z = torch.randn(n_energies, 1).to(params.device)
            elif self.base_type == 'lognormal':
                z = torch.exp(torch.randn(n_energies, 1).to(params.device))
            x = self.inverse(z, params, source_types)
            return x
