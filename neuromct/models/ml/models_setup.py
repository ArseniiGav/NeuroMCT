import torch
from .tede import TEDE
from ...configs import data_configs
from ...utils import tede_argparse
# from NF import FlowsModel

def setup(model_type, device, path_to_models=None):
    if model_type == 'tede':
        args = tede_argparse()
        model = TEDE(
            n_sources=args.n_sources,
            output_dim=args.output_dim,
            d_model=args.d_model,
            activation=args.activation,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            entmax_alpha=args.entmax_alpha
        ).double().to(device)

        path_to_models = path_to_models if path_to_models is not None \
                else data_configs['path_to_models']
        model.load_state_dict(torch.load(f"{path_to_models}/tede_model.pth", map_location=device))
        model.eval()
        return model

    elif model_type == 'nf':
        nf_model = FlowsModel(base_type=base_type,
                flow_type=flow_type,
                N_conditions=N_conditions,
                num_layers=num_layers,
                lr=lr,
                loss_function=loss_function,
                sources=sources
        ).to(device)
        nf_model.load_state_dict(torch.load(path, map_location=device))
        nf_model.eval()

        return nf_model
    else:
        raise ValueError("Model type should be 'tede' or 'nf'")
