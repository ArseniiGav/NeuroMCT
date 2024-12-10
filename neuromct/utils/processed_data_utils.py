from neuromct.configs import data_configs


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
    source_types_reshaped = source_types.view(n_sources, val2_n_datasets, n_bins)

    spectra_rates = spectra_reshaped.mean(1) # shape: (n_sources, n_bins)
    params_values = params_reshaped[:, 0, :] # shape: (n_sources, params_dim)
    source_types_values = source_types_reshaped[:, 0, :] # shape: (n_sources, 1)

    return (spectra_rates, params_values, source_types_values)
