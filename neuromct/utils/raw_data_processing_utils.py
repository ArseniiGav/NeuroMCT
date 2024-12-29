import numpy as np

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
