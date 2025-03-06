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
bin_size = 0.02
bins = np.arange(0.4, 16.41, bin_size, dtype=np.float32)
n_bins = bins.shape[0] - 1

data_configs = {
    # paths' configs
    "base_path_to_models": "./", 
    "path_to_processed_data": "/storage/jmct_paper/processed_data", 
    "path_to_tede_hopt_results": "/storage/jmct_paper/results/tede_hyperopt",
    "path_to_tede_training_results": "/storage/jmct_paper/results/tede_training",
    "path_to_nfde_hopt_results": "/storage/jmct_paper/results/nfde_hyperopt",
    "path_to_nfde_training_results": "/storage/jmct_paper/results/nfde_training",

    # sources' configs
    "n_sources": 5,
    "sources": sources,
    "sources_names_to_vis": sources_names_to_vis,   
    "sources_colors_to_vis": sources_colors_to_vis,

    # params for plots configs
    "plot_every_n_steps": 50,
    "n_params_values_to_vis": 4,

    "params_values_to_vis_training": torch.tensor(
        [0.0500, 0.3500, 0.6500, 0.9500], dtype=torch.float32),
    "base_value_to_vis_training": torch.tensor(
        0.5000, dtype=torch.float32),

    "params_values_to_vis_val1": torch.tensor(
        [0.0750, 0.3750, 0.6750, 0.9750], dtype=torch.float32),
    "base_value_to_vis_val1": torch.tensor(
        0.4750, dtype=torch.float32),

    "val2_n_datasets": 1000,
    "kB_val2_values": (7.35, 14.55, 23.55),
    "fC_val2_values": (0.075, 0.475, 0.975),
    "LY_val2_values": (8300, 9900, 11900),

    "bin_size": bin_size, # kNPE
    "kNPE_bins_edges": bins, # kNPE
    "n_bins": n_bins,
    "params_dim": 3, # kB, fC, LY
    "n_conditions": 4 # kB, fC, LY, source_type
}
