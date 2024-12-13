import numpy
import torch
import matplotlib.pyplot as plt
from .matplotlib_setup import matplotlib_setup
from neuromct.dataset import load_minimax_scaler, get_val2_data_rates
from neuromct.configs import data_configs
matplotlib_setup(tick_labelsize=14, axes_labelsize=14, legend_fontsize=9)


class ModelResultsVisualizator:
    def __init__(self):

        path_to_models = data_configs['path_to_models']
        self.scaler = load_minimax_scaler(path_to_models)

        self.params_values_to_vis = data_configs['params_values_to_vis']
        self.n_params_values_to_vis = len(self.params_values_to_vis)
        self.base_value_to_vis = data_configs['base_value_to_vis']

        self.path_to_plots = data_configs['path_to_plots']
        self.path_to_processed_data = data_configs['path_to_processed_data']
        self.n_sources = data_configs['n_sources']
        self.sources_names_to_vis = data_configs['sources_names_to_vis']
        self.sources_colors_to_vis = data_configs['sources_colors_to_vis']
        self.kNPE_bins_edges = data_configs['kNPE_bins_edges']
        self.n_bins = data_configs['n_bins']
        self.params_dim = data_configs['params_dim']

        val1_data = self._load_val_data_to_vis(dataset_type="val1")
        self.val1_data_to_vis, self.val1_params_to_vis_transformed = self._get_data_to_vis(
            "val1", val1_data)
        self.val1_spectra_to_vis = self.val1_data_to_vis[0]

        self.kB_val2_values = data_configs['kB_val2_values']
        self.fC_val2_values = data_configs['fC_val2_values']
        self.LY_val2_values = data_configs['LY_val2_values']

        self.val2_values = numpy.array(
            [self.kB_val2_values, self.fC_val2_values, self.LY_val2_values],
            dtype=numpy.float64).T
        self.val2_params_to_vis = numpy.repeat(
            self.val2_values, self.n_sources, axis=0)
        self.val2_params_to_vis = torch.tensor(
            self.scaler.transform(self.val2_params_to_vis),
            dtype=torch.float64)
        self.val2_source_types_to_vis = torch.arange(
            self.n_sources, dtype=torch.int64
        ).unsqueeze(1).repeat(self.params_dim, 1)

        val2_1_data = self._load_val_data_to_vis(dataset_type="val2_1")
        val2_2_data = self._load_val_data_to_vis(dataset_type="val2_2")
        val2_3_data = self._load_val_data_to_vis(dataset_type="val2_3")
        val2_data = (val2_1_data, val2_2_data, val2_3_data)
        self.val2_data_rates_to_vis = self._get_data_to_vis("val2", val2_data)

    def _load_val_data_to_vis(self, dataset_type):
        spectra_path = f"{self.path_to_processed_data}/{dataset_type.split('_')[0]}/{dataset_type}_spectra.pt"
        params_path = f"{self.path_to_processed_data}/{dataset_type.split('_')[0]}/{dataset_type}_params.pt"
        source_types_path = f"{self.path_to_processed_data}/{dataset_type.split('_')[0]}/{dataset_type}_source_types.pt"

        spectra = torch.load(spectra_path, weights_only=True)
        spectra = spectra / spectra.sum(1)[:, None]
        params = torch.load(params_path, weights_only=True)
        source_types = torch.load(source_types_path, weights_only=True)
        data = (spectra, params, source_types)
        return data
    
    def _get_data_to_vis(self, dataset_type, data):
        if dataset_type == "val1":
            spectra, params, source_types = data

            spectra_samples_to_vis = []
            params_samples_to_vis = []
            source_types_samples_to_vis = []
            params_samples_to_vis_transformed = []
            for j in range(self.params_dim):
                if self.n_params_values_to_vis > 1:
                    param_vary_condition = torch.logical_or(
                        torch.isclose(params[:, j], self.params_values_to_vis[0]),
                        torch.isclose(params[:, j], self.params_values_to_vis[1])
                    )
                    for i in range(2, len(self.params_values_to_vis)):
                        param_vary_condition = torch.logical_or(
                            param_vary_condition,
                            torch.isclose(params[:, j], self.params_values_to_vis[i])
                        )
                    for k in range(self.params_dim):
                        if k != j:
                            param_vary_condition = torch.logical_and(
                                param_vary_condition,
                                torch.isclose(params[:, k], self.base_value_to_vis)
                            )
                else:
                    param_vary_condition = torch.isclose(params[:, j], self.params_values_to_vis[0])
                    for k in range(self.params_dim):
                        if k != j:
                            param_vary_condition = torch.logical_and(
                                param_vary_condition,
                                torch.isclose(params[:, k], self.base_value_to_vis)
                            )
                param_indexes_to_vis = torch.where(param_vary_condition)[0]

                spectra_sample_to_vis = spectra[param_indexes_to_vis]
                params_sample_to_vis = params[param_indexes_to_vis]        
                source_types_sample_to_vis = source_types[param_indexes_to_vis]

                spectra_samples_to_vis.append(spectra_sample_to_vis)
                params_samples_to_vis.append(params_sample_to_vis)
                source_types_samples_to_vis.append(source_types_sample_to_vis)

                params_sample_to_vis_transformed = self.scaler.inverse_transform(params_sample_to_vis)
                params_samples_to_vis_transformed.append(params_sample_to_vis_transformed)
            data_to_vis = (spectra_samples_to_vis, params_samples_to_vis, source_types_samples_to_vis)
            return data_to_vis, params_samples_to_vis_transformed

        elif dataset_type == "val2":
            val2_data_rates = []
            for val2_x_data in data:
                val2_x_data_rates = get_val2_data_rates(val2_x_data)
                val2_data_rates.append(val2_x_data_rates[0])
            return val2_data_rates
    
    def _get_subplot_title(self, params_transformed):
        kB, fC, Y = params_transformed
        title = r"$k_{B}$"+f": {kB:.2f} [g/cm2/GeV], "
        title = title + r"$f_{C}$"+f": {fC:.3f}, "
        title = title + r"$Y$"+f": {Y:.0f} [1/MeV]"
        return title
    
    def _get_suptitle(self, current_epoch, global_step, val_metric, val_data_type):
        suptitle = f"Validation dataset №{val_data_type}. Epoch: {current_epoch}, "
        suptitle = suptitle + f"Iteration: {global_step // self.n_sources}, "
        suptitle = suptitle + r"$W^{V_%s}_{1}$ " %val_data_type
        suptitle = suptitle + f"= {val_metric:.5f}"
        return suptitle

    def plot_val1_spectra(
            self,
            spectra_pdf_to_vis: list,
            current_epoch: int,
            global_step: int,
            val1_loss: float,
        ) -> None: 

        fig, ax = plt.subplots(self.params_dim, self.n_params_values_to_vis,
                               figsize=(self.params_dim*6, self.n_params_values_to_vis*3))
        ax = ax.flatten()
        for m in range(self.params_dim):
            for i in range(self.n_params_values_to_vis * self.n_sources):
                j = i % self.n_params_values_to_vis + m * self.n_params_values_to_vis
                k = i // self.n_params_values_to_vis

                current_params_transformed = self.val1_params_to_vis_transformed[m][i]
                subplot_title = self._get_subplot_title(current_params_transformed)

                ########### plot truth ###########
                ax[j].stairs(
                    self.val1_spectra_to_vis[m][i], self.kNPE_bins_edges,
                    label=self.sources_names_to_vis[k],
                    color=self.sources_colors_to_vis[k],
                    alpha=0.6
                )

                ########### plot predicted ###########
                ax[j].stairs(
                    spectra_pdf_to_vis[m][i], self.kNPE_bins_edges,
                    color=self.sources_colors_to_vis[k],
                    linestyle='--',
                    alpha=1.0
                )
                
                if k == 4:
                    ax[j].plot([0], [0], color='black', linewidth=1.5, label="JUNOSW")
                    ax[j].plot([0], [0], color='black', linestyle='--', linewidth=1.5, label="TEDE")
                    
                    handles, labels = ax[j].get_legend_handles_labels()                
                    legend1 = ax[j].legend(handles[:5], labels[:5], frameon=1, ncol=1, fontsize=10, loc="upper right",)
                    legend2 = ax[j].legend(handles[5:], labels[5:], frameon=1, ncol=1, fontsize=10, loc=(0.32, 0.82))
                    ax[j].add_artist(legend1)
                    ax[j].add_artist(legend2)

                ax[j].set_title(subplot_title, fontsize=10)
                ax[j].set_ylim(1e-4, 0.25)
                ax[j].set_yscale("log")
                ax[j].set_xlim(0.0, 16.0)
                
                if j >= self.n_params_values_to_vis * (self.params_dim - 1):
                    ax[j].set_xlabel("Number of photo-electrons: " + r"$N_{p.e.} \ / \ 10^3$")
                if j % self.n_params_values_to_vis == 0:
                    ax[j].set_ylabel("Prob. density: " + r"$f(N_{p.e.} | k_{B}, f_{C}, Y)$")
        
        suptitle = self._get_suptitle(current_epoch, global_step, val1_loss, val_data_type=1)
        fig.suptitle(suptitle, x=0.3, y=0.99, fontsize=20)
        fig.tight_layout()
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_v1.png')
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_v1.pdf')
        plt.close(fig)

    def plot_val2_spectra(
            self,
            spectra_pdf_to_vis: numpy.ndarray,
            current_epoch: int,
            global_step: int,
            val2_loss: float,
        ) -> None:
              
        fig, axes = plt.subplots(2, self.params_dim, 
                                 figsize=(self.params_dim * 6, self.params_dim * 2.5),
                                 gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1})
        axes = axes.reshape(2, self.params_dim)
        for j in range(self.params_dim):
            ax_main = axes[0, j]
            ax_diff = axes[1, j]

            for k in range(self.n_sources):
                i = j * self.n_sources + k

                truth = self.val2_data_rates_to_vis[j][k]
                predicted = spectra_pdf_to_vis[i]

                ########### plot truth ###########
                ax_main.stairs(
                    truth,
                    self.kNPE_bins_edges,
                    label=self.sources_names_to_vis[k],
                    color=self.sources_colors_to_vis[k],
                    linewidth=1.25,
                    alpha=0.6
                )

                ########### plot predicted ###########
                ax_main.stairs(
                    predicted,
                    self.kNPE_bins_edges,
                    color=self.sources_colors_to_vis[k],
                    linestyle='--',
                    linewidth=1.25,
                    alpha=1.0
                ) 

                ########### plot relative difference ###########
                ax_diff.stairs(
                    (predicted - truth) / truth,
                    self.kNPE_bins_edges,
                    label=self.sources_names_to_vis[k], 
                    color=self.sources_colors_to_vis[k],
                    linewidth=1.25,
                    alpha=1.0
                )

                if k == 4:
                    ax_main.plot([0], [0], color='black', linewidth=2, label="JUNOSW")
                    ax_main.plot([0], [0], color='black', linestyle='--', linewidth=2, label="TEDE")
                    
                    handles, labels = ax_main.get_legend_handles_labels()                
                    legend1 = ax_main.legend(handles[:5], labels[:5], frameon=1, ncol=1, fontsize=12, loc="upper right",)
                    legend2 = ax_main.legend(handles[5:], labels[5:], frameon=1, ncol=1, fontsize=12, loc=(0.27, 0.83))
                    ax_main.add_artist(legend1)
                    ax_main.add_artist(legend2)

            ax_diff.set_xlim(0.0, 16.0)
            ax_diff.set_xlabel("Number of photo-electrons: " + r"$N_{p.e.} \ / \ 10^3$", fontsize=16)
            ax_diff.set_ylim(-1.25, 3.25)
            ax_diff.set_yticks([-1, 0, 1, 2, 3])
            # ax_diff.set_yscale("symlog")
            if j == 0:
                ax_diff.set_ylabel(
                    r"$\frac{f_{\rm{TEDE}} - f_{\rm{JUNOSW}}}{f_{\rm{JUNOSW}}}$",
                    fontsize=17
                )

            ax_main.set_ylim(1e-5, 0.4)
            ax_main.set_xlim(0.0, 16.0)
            ax_main.tick_params(labelbottom=False)
            ax_main.set_yscale("log")
            if j == 0:
                ax_main.set_ylabel(
                    "Prob. density: " + r"$f(N_{p.e.} | k_{B}, f_{C}, Y)$",
                    fontsize=16
                )

            subplot_title = self._get_subplot_title(
                (self.kB_val2_values[j], self.fC_val2_values[j], self.LY_val2_values[j]))
            ax_main.set_title(subplot_title, fontsize=13)

        for j in range(self.params_dim):
            axes[0, j].sharex(axes[1, j])

        suptitle = self._get_suptitle(current_epoch, global_step, val2_loss, val_data_type=2)
        fig.suptitle(suptitle, x=0.3, y=0.99, fontsize=20)
        fig.tight_layout()
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_v2.png')
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_v2.pdf')
        plt.close(fig)

    def plot_training_process(
            self,
            val1_loss_to_plot: list,
            val2_loss_to_plot: list,
            val_loss_to_plot: list,
            train_loss_to_plot: list,
            train_loss: float,
            val1_loss: float,
            val2_loss: float,
            val_loss: float,
            global_step: int,
            current_epoch: int
        ) -> None:

        title = r'$\rm W^{T}_{1} = $' + f"{train_loss:.5f}, "
        title += r'$\rm W^{V_1}_{1} = $' + f"{val1_loss:.5f}, "
        title += r'$\rm W^{V_2}_{1} = $' + f"{val2_loss:.5f}, "
        title += r'$\rm W^{V}_{1} = \rm \frac{W^{V_1}_{1} + 4 \cdot \rm W^{V_2}_{}}{5}$' + f' = {val_loss:.5f}'

        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        ax.plot(
            numpy.arange(len(val1_loss_to_plot)-1),
            val1_loss_to_plot[1:], 
            label="Validation dataset №1: " + r'$\rm D^{C}_{V_1}$', 
            color='royalblue',
            alpha=0.9,  
            linewidth=1.25
        )
        ax.plot(
            numpy.arange(len(val2_loss_to_plot)-1),
            val2_loss_to_plot[1:], 
            label="Validation dataset №2: " + r'$\rm D^{C}_{V_2}$', 
            color='darkred', 
            alpha=0.9,
            linewidth=1.25
        )
        ax.plot(
            numpy.arange(len(val_loss_to_plot)-1),
            val_loss_to_plot[1:], 
            label="Validation datasets combined: " + r'$\rm D^{C}_{V}$', 
            color='darkgreen', 
            alpha=0.9,
            linewidth=1.25
        )
        ax.plot(
            numpy.arange(len(train_loss_to_plot)-1),
            train_loss_to_plot[1:], 
            label="Training data: " + r'$\rm D^{C}_{T}$', 
            color='black',
            alpha=0.9, 
            linewidth=1.25
        )

        ax.set_ylabel("Loss", fontsize=16)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.1)
        ax.set_xlabel('Iteration', fontsize=16)
        ax.legend(loc="upper right", fontsize=16)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        fig.suptitle(title, x=0.25, y=0.99, fontsize=20)
        fig.tight_layout()
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_training_process.png') 
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_training_process.pdf') 
        plt.show()
