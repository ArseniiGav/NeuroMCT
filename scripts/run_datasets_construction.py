import torch
import argparse
import numpy as np
from neuromct.dataset import load_minimax_scaler, load_raw_data
from neuromct.utils import construct_params_grid, construct_source_types_vector
from neuromct.configs import data_configs

parser = argparse.ArgumentParser(description='Process the raw data and build model inputs')
parser.add_argument("--dataset_type", type=str, default="", help='Dataset type: training/val1/val2')
args = parser.parse_args()
dataset_type = args.dataset_type

path_to_processed_data = data_configs['path_to_processed_data']
path_to_raw_data = data_configs['path_to_raw_data']
path_to_models = data_configs['path_to_models']
scaler = load_minimax_scaler(path_to_models)

n_sources = data_configs['n_sources']
sources = data_configs['sources']

if dataset_type == 'val2':
    dataset_dir_name = "validation_data2"
    n_datasets = data_configs['val2_n_datasets']
    kB_values = np.array(data_configs['kB_val2_values'], dtype=np.float32).reshape(-1, 1)
    fC_values = np.array(data_configs['fC_val2_values'], dtype=np.float32).reshape(-1, 1)
    LY_values = np.array(data_configs['LY_val2_values'], dtype=np.float32).reshape(-1, 1)
    params_values = np.concatenate((kB_values, fC_values, LY_values), axis=1, dtype=np.float32)
    params_values_scaled = scaler.transform(params_values)
    kNPE_bins_edges = data_configs['kNPE_bins_edges']
    
    for j in range(len(kB_values)):      
        NPEs_counts_arrays = []
        for source in sources:    
            NPEs_counts_array = load_raw_data(path_to_raw_data, source, dataset_dir_name+f"_{j+1}", n_datasets, kNPE_bins_edges)
            NPEs_counts_arrays.append(NPEs_counts_array)
        NPEs_counts_arrays = np.vstack(NPEs_counts_arrays, dtype=np.float32)
        NPEs_counts_tensor = torch.tensor(NPEs_counts_arrays, dtype=torch.float32)
        
        params_grid_scaled = np.repeat(params_values_scaled[j].reshape(1, -1), n_datasets, axis=0)
        params_grid_scaled_tiled = np.tile(params_grid_scaled, (n_sources, 1))
        params_grid_scaled_tiled_tensor = torch.tensor(params_grid_scaled_tiled, dtype=torch.float32)
        source_types = construct_source_types_vector(n_datasets, n_sources)
        source_types_tensor = torch.tensor(source_types, dtype=torch.int32)

        torch.save(NPEs_counts_tensor, f"{path_to_processed_data}/{dataset_type}/{dataset_type}_{j+1}_spectra.pt")
        torch.save(params_grid_scaled_tiled_tensor, f"{path_to_processed_data}/{dataset_type}/{dataset_type}_{j+1}_params.pt")
        torch.save(source_types_tensor, f"{path_to_processed_data}/{dataset_type}/{dataset_type}_{j+1}_source_types.pt")
elif dataset_type == "training" or dataset_type == "val1":
    if dataset_type == "training":
        dataset_dir_name = ""
    elif dataset_type == "val1":
        dataset_dir_name = "validation_data"

    grid_size = data_configs[f'{dataset_type}_grid_size']
    n_points = grid_size**3
    kB = np.arange(*data_configs[f'kB_{dataset_type}_grid_lims'], dtype=np.float32)
    fC = np.arange(*data_configs[f'fC_{dataset_type}_grid_lims'], dtype=np.float32)
    LY = np.arange(*data_configs[f'LY_{dataset_type}_grid_lims'], dtype=np.float32)
    kNPE_bins_edges = data_configs['kNPE_bins_edges']

    params_grid = construct_params_grid(kB, fC, LY, grid_size)
    params_grid_scaled = scaler.transform(params_grid)
    params_grid_scaled_tiled = np.tile(params_grid_scaled, (n_sources, 1))
    params_grid_scaled_tiled_tensor = torch.tensor(params_grid_scaled_tiled, dtype=torch.float32)
    source_types = construct_source_types_vector(n_points, n_sources)
    source_types_tensor = torch.tensor(source_types, dtype=torch.int32)

    NPEs_counts_arrays = []
    for source in sources:    
        NPEs_counts_array = load_raw_data(path_to_raw_data, source, dataset_dir_name, n_points, kNPE_bins_edges)
        NPEs_counts_arrays.append(NPEs_counts_array)
    NPEs_counts_arrays = np.vstack(NPEs_counts_arrays, dtype=np.float32)
    NPEs_counts_tensor = torch.tensor(NPEs_counts_arrays, dtype=torch.float32)
    
    torch.save(NPEs_counts_tensor, f"{path_to_processed_data}/{dataset_type}/{dataset_type}_spectra.pt")
    torch.save(params_grid_scaled_tiled_tensor, f"{path_to_processed_data}/{dataset_type}/{dataset_type}_params.pt")
    torch.save(source_types_tensor, f"{path_to_processed_data}/{dataset_type}/{dataset_type}_source_types.pt")
else:
    raise Exception('Choose between training, val1 and val2!')
