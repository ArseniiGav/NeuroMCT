import numpy as np


configs = {
    "path_to_models": "/storage/jmct_paper/saved_models", 
    "path_to_processed_data": "/storage/jmct_paper/processed_data", 
    "path_to_raw_data": "/mnt/arsenii/NeuroMCT/kB_fC_LY_10k_events",

    "n_sources": 5,
    "sources": ['Cs137', 'K40', 'Co60', 'AmBe', 'AmC'],

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

    "kNPE_bins_edges": np.arange(0.0, 16.01, 0.02, dtype=np.float64) # kNPE
}
