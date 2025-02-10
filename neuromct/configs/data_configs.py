import numpy as np
import torch

sources = ['Cs137', 'K40', 'Co60', 'AmBe', 'AmC']
sources_names_to_vis = [
    r'$\rm {}^{137} Cs$',
    r'$\rm {}^{40} K$',
    r'$\rm {}^{60} Co$',
    r'$\rm {}^{241}Am \mathrm{-} Be$',
    r'$\rm {}^{241}Am \mathrm{-} {}^{13}C$'
]
sources_colors_to_vis = ['darkgreen', 'royalblue', 
                         'firebrick', 'indigo', 'peru']
bins = np.arange(0.4, 16.41, 0.02, dtype=np.float64)
n_bins = bins.shape[0] - 1

data_configs = {
    # paths' configs
    "path_to_models": "/storage/jmct_paper/saved_models", 
    "path_to_processed_data": "/storage/jmct_paper/processed_data", 
    "path_to_raw_data": "/mnt/arsenii/NeuroMCT/kB_fC_LY_10k_events",
    "path_to_optuna_results": "/storage/jmct_paper/results/tede_hyperopt",
    
    # sources' configs
    "n_sources": 5,
    "sources": sources,
    "sources_names_to_vis": sources_names_to_vis,   
    "sources_colors_to_vis": sources_colors_to_vis,

    # params for plots configs
    "plot_every_n_steps": 10,
    "n_params_values_to_vis": 4,

    "params_values_to_vis_training": torch.tensor(
        [0.0500, 0.3500, 0.6500, 0.9500], dtype=torch.float64),
    "base_value_to_vis_training": torch.tensor(
        0.5000, dtype=torch.float64),

    "params_values_to_vis_val1": torch.tensor(
        [0.0750, 0.3750, 0.6750, 0.9750], dtype=torch.float64),
    "base_value_to_vis_val1": torch.tensor(
        0.4750, dtype=torch.float64),

    # data processing configs
    "training_grid_size": 21,
    "kB_training_grid_lims": (6, 24.1, 0.9), # g/cm2/GeV
    "fC_training_grid_lims": (0, 1.01, 0.05),
    "LY_training_grid_lims": (8000, 12001, 200), # 1 / MeV
    
    "val1_grid_size": 10,
    "kB_val1_grid_lims": (7.35, 24, 1.8), # g/cm2/GeV
    "fC_val1_grid_lims": (0.075, 1, 0.1),
    "LY_val1_grid_lims": (8300, 12000, 400), # 1 / MeV

    "val2_n_datasets": 1000,
    "kB_val2_values": (7.35, 14.55, 23.55),
    "fC_val2_values": (0.075, 0.475, 0.975),
    "LY_val2_values": (8300, 9900, 11900),

    "kNPE_bins_edges": bins, # kNPE
    "n_bins": n_bins,
    "params_dim": 3 # kB, fC, LY
}
