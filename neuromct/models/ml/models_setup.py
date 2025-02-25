import torch

from .tede import TEDE
from .nfde import NFDE
from ...configs import data_configs
from ...utils import tede_argparse

def setup(model_type, device, path_to_models=None):
    path_to_models = path_to_models if path_to_models is not None \
            else data_configs['path_to_models']

    if model_type == 'tede':
        args = tede_argparse()
        tede = TEDE(
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
        ).float().to(device)

        tede.load_state_dict(
            torch.load(
                f"{path_to_models}/tede_model.pth", 
                map_location=device,
                weights_only=True
            )
        )
        tede.eval()
        return tede
    elif model_type == 'nfde':
        nfde = NFDE(
            n_flows=args.n_flows,
            activation=args.activation_function,
            n_conditions=args.n_conditions,
            base_type=args.base_type,
            flow_type=args.flow_type
        ).float().to(device)

        nfde.load_state_dict(
            torch.load(
                f"{path_to_models}/tede_model.pth", 
                map_location=device,
                weights_only=True
            )
        )
        nfde.eval()
        return nfde
    else:
        raise ValueError("Model type should be 'tede' or 'nfde'")
