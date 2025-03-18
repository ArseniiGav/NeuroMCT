import torch

from .tede import TEDE
from .nfde import NFDE
from ...configs import data_configs
from ...utils import (
    tede_argparse,
    nfde_argparse
)

def setup(model_type, device, base_path_to_models=None):
    base_path_to_models = base_path_to_models if base_path_to_models is not None \
            else data_configs['base_path_to_models']
    
    if model_type == 'tede':
        args = tede_argparse()
        tede_model = TEDE(
            n_sources=args.n_sources,
            output_dim=args.output_dim,
            d_model=args.d_model,
            activation=args.activation_function,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            temperature=args.temperature,
            bin_size=data_configs['bin_size']
        ).double().to(device)

        tede_model.load_state_dict(
            torch.load(
                f"{base_path_to_models}/models/tede_model.pth", 
                map_location=device,
                weights_only=True
            )
        )
        tede_model.eval()
        return tede_model
    elif model_type == 'nfde':
        args = nfde_argparse()
        nfde_model = NFDE(
            n_flows=args.n_flows,
            n_conditions=data_configs['n_conditions'],
            n_sources=args.n_sources,
            n_units=args.n_units,
            activation=args.activation_function,
            flow_type=args.flow_type,
            dropout=args.dropout
        ).double().to(device)

        nfde_model.load_state_dict(
            torch.load(
                f"{base_path_to_models}/models/nfde_model.pth", 
                map_location=device,
                weights_only=True
            )
        )
        nfde_model.eval()
        return nfde_model
    else:
        raise ValueError("Model type should be 'tede' or 'nfde'")
