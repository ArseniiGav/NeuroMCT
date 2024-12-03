import numpy as np


def construct_model_input(kB, fC, LY, grid_size, n_sources, scaler):
    params_grid = construct_params_grid(kB, fC, LY, grid_size)
    params_grid_scaled = scaler.transform(params_grid)
    model_input = add_sources_labels_to_params_grid(params_grid_scaled, n_sources)
    return model_input

def construct_params_grid(kB, fC, LY, grid_size):
    kB_list = []
    fC_list = []
    LY_list = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                kB_list.append(kB[i])
                fC_list.append(fC[j])
                LY_list.append(LY[k])
    
    kB_list = np.array(kB_list, dtype=np.float64).reshape(-1, 1)
    fC_list = np.array(fC_list, dtype=np.float64).reshape(-1, 1)
    LY_list = np.array(LY_list, dtype=np.float64).reshape(-1, 1)

    params_grid = np.concatenate([kB_list, fC_list, LY_list], axis=1, dtype=np.float64)
    return params_grid

def add_sources_labels_to_params_grid(params_grid, n_sources):
    n_params_conbinations = params_grid.shape[0]
    number_of_spectra = n_sources * n_params_conbinations
    one_hot_basis = np.eye(n_sources, dtype=np.float64)
    sources_labels_ohe = np.repeat(one_hot_basis, n_params_conbinations, axis=0).astype(int)

    params_grid_tiled = np.tile(params_grid, (n_sources, 1))
    model_input = np.hstack((params_grid_tiled, sources_labels_ohe), dtype=np.float64)
    return model_input
