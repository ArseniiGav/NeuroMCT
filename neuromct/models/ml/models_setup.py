import torch
import pickle
import numpy as np
from .tede import TEDE
from ...configs import data_configs
from ...utils import tede_argparse
# from NF import FlowsModel

# setup the hp and rest
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup(model_type):
    if model_type == 'tede':
        args = tede_argparse()
        model = TEDE(
            n_sources=data_configs['n_sources'],
            output_dim=args.output_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            temperature=args.temperature
        ).double().to(device)
        
        model.load_state_dict(torch.load(f"{data_configs['path_to_models']}/tede_model.pth", map_location=device))
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
