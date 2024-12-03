import torch
import pickle
import numpy as np
from GANH import cWGAN_GP
from Regressor import TransformerRegressor
from NF import FlowsModel

# setup the hp and rest
device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
xllim, xrlim = 400, 16400
bins = np.arange(xllim, xrlim+1, 20)
lr_r = 1e-4

lr = 1e-4
num_layers = 100
base_type = 'normal'
flow_type = 'planar'
loss_function = 'kl-div'
sources = ['Cs137', 'K40', 'Co60', 'AmBe', 'AmC']
N_conditions = 3


def setup(model_type, path):
    if model_type == 'gan':
        GAN_model = cWGAN_GP(latent_dim = 20, output_dim=800).to(device)
        GAN_model.load_state_dict(torch.load(path, map_location=device))
        GAN_model.eval()
        
        return GAN_model
    elif model_type == 'transformer':
        TRegressor_model = TransformerRegressor(bins=bins, lr=lr_r,).to(device)
        TRegressor_model.load_state_dict(torch.load(path, map_location=device))
        TRegressor_model.eval()
        
        return TRegressor_model
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
