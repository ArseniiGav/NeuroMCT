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

def construct_source_types_vector(n_datasets, n_sources):
    vector_basis = np.arange(n_sources, dtype=np.int64)
    source_types_vector = np.repeat(vector_basis, n_datasets).reshape(-1, 1)
    return source_types_vector
