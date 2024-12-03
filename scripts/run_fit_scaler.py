import numpy as np
from neuromct.configs import configs
from neuromct.utils import construct_params_grid
from neuromct.dataset import fit_minimax_scaler


path_to_models = configs['path_to_models']
grid_size = configs['training_grid_size']
kB = np.arange(*configs['kB_training_grid_lims'], dtype=np.float64)
fC = np.arange(*configs['fC_training_grid_lims'], dtype=np.float64)
LY = np.arange(*configs['LY_training_grid_lims'], dtype=np.float64)

params_grid = construct_params_grid(kB, fC, LY, grid_size)
fit_minimax_scaler(path_to_models, params_grid)
