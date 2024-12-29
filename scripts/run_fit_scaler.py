import numpy as np
from neuromct.configs import data_configs
from neuromct.utils import construct_params_grid
from neuromct.dataset import fit_minimax_scaler

path_to_models = data_configs['path_to_models']
grid_size = data_configs['training_grid_size']
kB = np.arange(*data_configs['kB_training_grid_lims'], dtype=np.float64)
fC = np.arange(*data_configs['fC_training_grid_lims'], dtype=np.float64)
LY = np.arange(*data_configs['LY_training_grid_lims'], dtype=np.float64)

params_grid = construct_params_grid(kB, fC, LY, grid_size)
fit_minimax_scaler(path_to_models, params_grid)
