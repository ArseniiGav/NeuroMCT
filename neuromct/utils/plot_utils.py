import numpy
import torch
import matplotlib.pyplot as plt
from .matplotlib_setup import matplotlib_setup
from .processed_data_utils import get_val2_data_rates
from neuromct.dataset import load_minimax_scaler
from neuromct.configs import data_configs
matplotlib_setup(tick_labelsize=14, axes_labelsize=14, legend_fontsize=9)


class ModelResultsVisualizator:
    def __init__(
            self,
            values_to_vis: list,
        ):

        path_to_models = data_configs['path_to_models']
        self.scaler = load_minimax_scaler(path_to_models)

        self.path_to_plots = data_configs['path_to_plots']
        self.path_to_processed_data = data_configs['path_to_processed_data']
        self.n_sources = data_configs['n_sources']
        self.sources_names_to_vis = data_configs['sources_names_to_vis']
        self.sources_colors_to_vis = data_configs['sources_colors_to_vis']
        self.kNPE_bins_edges = data_configs['kNPE_bins_edges']
        self.n_bins = data_configs['n_bins']
        self.params_dim = data_configs['params_dim']

        self.values_to_vis = values_to_vis
        self.n_values_to_vis = len(values_to_vis)

        val1_data = self._load_val_data_to_vis(dataset_type="val1")
        self.val1_data_to_vis = self._get_data_to_vis("val1", val1_data, values_to_vis)
        self.val1_spectra_to_vis = self.val1_data_to_vis[0]
        self.val1_params_to_vis_transformed = self.scaler.inverse_transform(self.val1_data_to_vis[1])

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
        params = torch.load(params_path, weights_only=True)
        source_types = torch.load(source_types_path, weights_only=True)
        data = (spectra, params, source_types)
        return data
    
    def _get_data_to_vis(self, dataset_type, data, values_to_vis):
        if dataset_type == "val1":
            spectra, params, source_types = data

            spectra_samples_to_vis = []
            params_samples_to_vis = []
            source_types_samples_to_vis = []
            for j in range(self.params_dim):
                if len(values_to_vis) > 1:
                    param_vary_condition = torch.logical_or(
                        (params[:, j] == values_to_vis[0]), (params[:, 0] == values_to_vis[1]))
                    for i in range(2, len(values_to_vis)):
                        param_vary_condition = torch.logical_or(
                            param_vary_condition, (params[:, j] == values_to_vis[i]))
                    for k in range(self.params_dim):
                        if k != j:
                            param_vary_condition = torch.logical_and(
                                param_vary_condition, (params[:, k] == 0.4750))
                else:
                    param_vary_condition = params[:, j] == values_to_vis[0]
                    for k in range(self.params_dim):
                        if k != j:
                            param_vary_condition = torch.logical_and(
                                param_vary_condition, (params[:, k] == 0.4750))
                param_indexes_to_vis = torch.where(param_vary_condition)[0]

                spectra_sample_to_vis = spectra[param_indexes_to_vis]
                params_sample_to_vis = params[param_indexes_to_vis]        
                source_types_sample_to_vis = source_types[param_indexes_to_vis]

                spectra_samples_to_vis.append(spectra_sample_to_vis)
                params_samples_to_vis.append(params_sample_to_vis)
                source_types_samples_to_vis.append(source_types_sample_to_vis)
            data_to_vis = (spectra_samples_to_vis, params_samples_to_vis, source_types_samples_to_vis)
            return data_to_vis
        elif dataset_type == "val2":
            val2_data_rates = []
            for val2_x_data in data:
                val2_x_data_rates = get_val2_data_rates(val2_x_data)
                val2_data_rates.append(val2_x_data_rates)
            return val2_data_rates
    
    def _get_subplot_title(params_transformed):
        kB, fC, Y = params_transformed
        title = r"$k_{B}$"+f": {kB:.2f} [g/cm2/GeV], "
        title = title + r"$f_{C}$"+f": {fC:.3f}, "
        title = title + r"$Y$"+f": {Y:.0f} [1/MeV]"
        return title
    
    def _get_suptitle(self, current_epoch, global_step, val_metric, val_data_type):
        suptitle = f"Validation dataset №{val_data_type}. Epoch: {current_epoch}, "
        suptitle = suptitle + f"Iteration: {global_step // self.n_sources}, "
        suptitle = suptitle + r"$W^{\rm{V_%s}}_{\rm{1}}$ " %val_data_type
        suptitle = suptitle + f"= {val_metric:.5f}"
        return suptitle

    def plot_val1_spectra(
            self,
            spectra_pdf_to_vis: numpy.ndarray,
            current_epoch: int,
            global_step: int,
            val1_metric: float,
            save: bool
        ) -> None: 

        fig, ax = plt.subplots(self.params_dim, self.n_values_to_vis,
                               figsize=(self.params_dim*6, self.n_values_to_vis*3))
        ax = ax.flatten()
        for m in range(self.params_dim):
            for i in range(self.n_values_to_vis):
                j = i % self.n_values_to_vis + m * self.n_values_to_vis
                k = i // self.n_values_to_vis

                current_params_transformed = self.val1_params_to_vis_transformed[m][i]
                subplot_title = self._get_subplot_title(current_params_transformed[m][i])

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
                    legend1 = ax[j].legend(handles[:5], labels[:5], frameon=1, ncol=1, fontsize=11, loc="upper right",)
                    legend2 = ax[j].legend(handles[5:], labels[5:], frameon=1, ncol=1, fontsize=11, loc=(0.47, 0.81))
                    ax[j].add_artist(legend1)
                    ax[j].add_artist(legend2)

                ax[j].set_title(subplot_title, fontsize=10)
                ax[j].set_ylim(1e-4, 0.25)
                ax[j].set_yscale("log")
                ax[j].set_xlim(0.0, 16.0)
                
                if j >= self.n_values_to_vis * (self.params_dim - 1):
                    ax[j].set_xlabel("Number of photo-electrons: " + r"$N_{p.e.} \ / \ 10^3$")
                if j % self.n_values_to_vis == 0:
                    ax[j].set_ylabel("Prob. density: " + r"$f(N_{p.e.} \ | \ k_{B}, f_{C}, L_{y})$")
        
        suptitle = self._get_subplot_title(current_epoch, global_step, val1_metric, val_data_type=1)
        fig.suptitle(suptitle, x=0.3, y=0.99, fontsize=20)
        fig.tight_layout()
        if save:
            fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_{global_step // self.n_sources}_v1.png')
            fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_{global_step // self.n_sources}_v1.pdf')
        plt.close(fig)

    def plot_val2_spectra(
            self,
            spectra_pdf_to_vis: numpy.ndarray,
            current_epoch: int,
            global_step: int,
            val2_metric: float,
            save: bool
        ) -> None:
              
        fig, ax = plt.subplots(1, self.params_dim, 
                               figsize=(self.params_dim*6, self.params_dim*2))
        ax = ax.flatten()
        for i in range(self.params_dim * self.n_sources):
            j = i // self.n_sources
            k = i % self.n_sources

            current_params_transformed = self.val1_params_to_vis_transformed[m][i]
            subplot_title = self._get_subplot_title(current_params_transformed[m][i])

            ########### plot truth ###########
            ax[j].stairs(
                self.val2_spectra_to_vis[j][k], self.kNPE_bins_edges,
                label=self.sources_names_to_vis[k],
                color=self.sources_colors_to_vis[k],
                linewidth=1.25,
                alpha=0.6
            )

            ########### plot predicted ###########
            ax[j].stairs(
                spectra_pdf_to_vis[j][k], self.kNPE_bins_edges,
                color=self.sources_colors_to_vis[k],
                linestyle='--',
                linewidth=1.25,
                alpha=1.0
            )
            
            if k == 4:
                ax[j].plot([0], [0], color='black', linewidth=2, label="JUNOSW")
                ax[j].plot([0], [0], color='black', linestyle='--', linewidth=2, label="TEDE")
                
                handles, labels = ax[j].get_legend_handles_labels()                
                legend1 = ax[j].legend(handles[:5], labels[:5], frameon=1, ncol=1, fontsize=12, loc="upper right",)
                legend2 = ax[j].legend(handles[5:], labels[5:], frameon=1, ncol=1, fontsize=12, loc=(0.52, 0.81))
                ax[j].add_artist(legend1)
                ax[j].add_artist(legend2)
            
            ax[j].set_title(subplot_title, fontsize=13)
            ax[j].set_ylim(1e-4, 0.25)
            ax[j].set_xlim(0.0, 16.0)
            ax[j].set_yscale("log")
            ax[j].set_xlabel("Number of photo-electrons: " + r"$N_{p.e.} \ / \ 10^3$")
            if j == 0:
                ax[j].set_ylabel("Prob. density: " + r"$f(N_{p.e.} \ | \ k_{B}, f_{C}, Y)$")

        suptitle = self._get_subplot_title(current_epoch, global_step, val2_metric, val_data_type=2)
        fig.suptitle(suptitle, x=0.3, y=0.99, fontsize=20)
        fig.tight_layout()
        if save:
            fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_{global_step // self.n_sources}_v2.png')
            fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_{global_step // self.n_sources}_v2.pdf')
        plt.close(fig)

    # def plot_val_metrics(self, save=False):
    #     self.val1_loss_to_plot.append(self.val1_epoch_loss)
    #     self.val2_loss_to_plot.append(self.val2_epoch_loss)
    #     self.val_loss_to_plot.append(self.val_epoch_loss)

    #     title = r'$\rm D^{C}_{V_1} = $' + f"{self.val1_epoch_loss:.5f}, "
    #     title += r'$\rm D^{C}_{V_2} = $' + f"{self.val2_epoch_loss:.5f}; "
    #     title += r'$\rm D^{C}_{V} = \rm \frac{D^{C}_{V_1} + 4 \cdot \rm D^{C}_{V_2}}{5}$' + f' = {self.val_epoch_loss:.5f}'

    #     fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    #     ax.plot(range(len(self.val1_loss_to_plot)), self.val1_loss_to_plot, label="Validation dataset №1: " + r'$\rm D^{C}_{V_1}$', color='royalblue', linewidth=1.25)
    #     ax.plot(range(len(self.val2_loss_to_plot)), self.val2_loss_to_plot, label="Validation dataset №2: " + r'$\rm D^{C}_{V_2}$', color='darkred', linewidth=1.25)
    #     ax.plot(range(len(self.val_loss_to_plot)), self.val_loss_to_plot, label="Validation datasets combined: " + r'$\rm D^{C}_{V}$', color='darkgreen', linewidth=1.25)

    #     ax.set_ylabel("Cosine distance", fontsize=16)
    #     ax.set_yscale("log")
    #     ax.set_ylim(1e-4, 1.1)
    #     ax.set_xlabel('Epochs', fontsize=16)
    #     ax.legend(loc="upper right", fontsize=16)
    #     ax.tick_params(axis='x', labelsize=14)
    #     ax.tick_params(axis='y', labelsize=14)
    #     # ax.set_axes(fontsize=14)

    #     fig.suptitle(title, x=0.25, y=0.99, fontsize=20)
    #     fig.tight_layout()
    #     if save:
    #         fig.savefig(f'plots/Transformer_training_process/epoch_{self.current_epoch}_val_metrics.png') 
    #     plt.show()

    # def plot_train_loss(self, save=False):

    #     fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    #     ax.plot(range(len(self.train_loss_to_plot)), self.train_loss_to_plot, label="Training data: " + r'$\rm D^{C}_{T}$', color='black', linewidth=1.25)

    #     ax.set_ylabel("Cosine distance", fontsize=16)
    #     ax.set_yscale("log")
    #     ax.set_ylim(1e-4, 1.1)
    #     ax.set_xlabel('Training steps', fontsize=16)
    #     ax.legend(loc="upper right", fontsize=16)
    #     ax.tick_params(axis='x', labelsize=14)
    #     ax.tick_params(axis='y', labelsize=14)
    #     # ax.set_axes(fontsize=14)

    #     fig.suptitle("Training dataset", x=0.1, y=0.99, fontsize=20)
    #     fig.tight_layout()
    #     if save:
    #         fig.savefig(f'plots/Transformer_training_process/epoch_{self.current_epoch}_train_loss.png') 
    #     plt.show()
