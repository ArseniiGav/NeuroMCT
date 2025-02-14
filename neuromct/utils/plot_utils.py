from ..plot import ModelResultsVisualizer

def res_visualizator_setup(configs):
    plot_every_n_steps = configs['plot_every_n_steps']
    path_to_processed_data = configs['path_to_processed_data']
    path_to_models = configs['path_to_models']

    n_sources = configs['n_sources']
    sources_names_to_vis = configs['sources_names_to_vis']
    sources_colors_to_vis = configs['sources_colors_to_vis']

    kNPE_bins_edges = configs['kNPE_bins_edges']
    params_dim = configs['params_dim']

    n_params_values_to_vis = configs['n_params_values_to_vis']
    params_values_to_vis_training = configs['params_values_to_vis_training']
    base_value_to_vis_training = configs['base_value_to_vis_training']
    params_values_to_vis_val1 = configs['params_values_to_vis_val1']
    base_value_to_vis_val1 = configs['base_value_to_vis_val1']

    kB_val2_values = configs['kB_val2_values']
    fC_val2_values = configs['fC_val2_values']
    LY_val2_values = configs['LY_val2_values']

    model_results_visualizer = ModelResultsVisualizer(
        plot_every_n_steps=plot_every_n_steps,
        path_to_scaler=path_to_models,
        path_to_processed_data=path_to_processed_data,
        n_sources=n_sources,
        sources_names_to_vis=sources_names_to_vis,
        sources_colors_to_vis=sources_colors_to_vis,
        bins_edges=kNPE_bins_edges,
        params_dim=params_dim,
        n_params_values_to_vis=n_params_values_to_vis,
        params_values_to_vis_training=params_values_to_vis_training,
        base_value_to_vis_training=base_value_to_vis_training,
        params_values_to_vis_val1=params_values_to_vis_val1,
        base_value_to_vis_val1=base_value_to_vis_val1,
        kB_val2_values=kB_val2_values,
        fC_val2_values=fC_val2_values,
        LY_val2_values=LY_val2_values
    )

    return model_results_visualizer
