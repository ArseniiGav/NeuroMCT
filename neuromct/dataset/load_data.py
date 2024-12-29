import torch
import uproot
import numpy as np
from tqdm import tqdm

from ..configs import data_configs

def load_raw_data(path_to_raw_data, source, dataset_type, n_points, bins):
    NPEs_counts_list = []
    for i in tqdm(range(n_points)):
        reco_file = uproot.open(f"{path_to_raw_data}/{source}/{dataset_type}/reco/reco-{i}.root")
        NPEs = 1.07 * np.array(reco_file['TRec']['m_NPE'].array(), dtype=np.float64) / 1000.
        NPEs_counts, _ = np.histogram(NPEs, bins=bins)
        NPEs_counts = NPEs_counts.reshape(-1, 1)        
        NPEs_counts_list.append(NPEs_counts)

    NPEs_counts_array = np.concatenate(NPEs_counts_list, axis=1, dtype=np.float64)
    return NPEs_counts_array.T

def load_processed_data(dataset_type, path_to_processed_data, val2_rates=False):
    if dataset_type in ["training", "val1", "val2_1", "val2_2", "val2_3"]:
        spectra_path = f"{path_to_processed_data}/{dataset_type.split('_')[0]}/{dataset_type}_spectra.pt"
        params_path = f"{path_to_processed_data}/{dataset_type.split('_')[0]}/{dataset_type}_params.pt"
        source_types_path = f"{path_to_processed_data}/{dataset_type.split('_')[0]}/{dataset_type}_source_types.pt"
    else:
        raise ValueError(
            """Invalid dataset_type! 
               Choose between 'training', 'val1', 'val2_1', 'val2_2', and 'val2_3'.""")

    spectra = torch.load(spectra_path, weights_only=True)
    params = torch.load(params_path, weights_only=True)
    source_types = torch.load(source_types_path, weights_only=True)
    data = (spectra, params, source_types)
    if val2_rates:
        if dataset_type in ["val2_1", "val2_2", "val2_3"]:
            data_rates = get_val2_data_rates(data)
        else:
            raise ValueError(
                """Invalid dataset_type! 
                   Rates computation is applicable only for 'val2_1', 'val2_2', and 'val2_3'.""")
        return data_rates
    else:
        return data

def get_val2_data_rates(val2_data):
    n_sources = data_configs['n_sources']
    val2_n_datasets = data_configs['val2_n_datasets']
    n_bins = data_configs['n_bins']
    params_dim = data_configs['params_dim']
    spectra, params, source_types = val2_data

    # shape: (n_sources*n_datasets, n_bins) --> (n_sources, n_datasets, n_bins)
    spectra_reshaped = spectra.view(n_sources, val2_n_datasets, n_bins)

    # shape: (n_sources*n_datasets, params_dim) --> (n_sources, n_datasets, params_dim)
    params_reshaped = params.view(n_sources, val2_n_datasets, params_dim)

    # shape: (n_sources*n_datasets, 1) --> (n_sources, n_datasets, 1)
    source_types_reshaped = source_types.view(n_sources, val2_n_datasets, 1)

    spectra_rates = spectra_reshaped.mean(1) # shape: (n_sources, n_bins)
    params_values = params_reshaped[:, 0, :] # shape: (n_sources, params_dim)
    source_types_values = source_types_reshaped[:, 0, :] # shape: (n_sources, 1)

    return (spectra_rates, params_values, source_types_values)
