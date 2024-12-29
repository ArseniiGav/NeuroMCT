import numpy as np
import torch

from ..dataset import load_minimax_scaler, load_processed_data

import matplotlib.pyplot as plt
from .matplotlib_setup import matplotlib_setup
matplotlib_setup(tick_labelsize=14, axes_labelsize=14, legend_fontsize=9)


class ModelResultsVisualizator:
    def __init__(
            self,
            path_to_plots: str,
            path_to_scaler: str, 
            path_to_processed_data: str,
            n_sources: int,
            sources_names_to_vis: list, 
            sources_colors_to_vis: list,
            bins_edges: np.ndarray,
            params_dim: int, 
            n_params_values_to_vis: int, 
            params_values_to_vis_training: torch.Tensor, 
            base_value_to_vis_training: torch.Tensor, 
            params_values_to_vis_val1: torch.Tensor, 
            base_value_to_vis_val1: torch.Tensor,
            kB_val2_values: list,
            fC_val2_values: list,
            LY_val2_values: list
        ):
        self.path_to_plots = path_to_plots
        self.path_to_processed_data = path_to_processed_data
        self.n_sources = n_sources
        self.sources_names_to_vis = sources_names_to_vis
        self.sources_colors_to_vis = sources_colors_to_vis
        self.kNPE_bins_edges = bins_edges
        self.n_bins = len(bins_edges) - 1
        self.params_dim = params_dim

        self.scaler = load_minimax_scaler(path_to_scaler)

        self.n_params_values_to_vis = n_params_values_to_vis
        training_data = load_processed_data("training", path_to_processed_data)
        self.training_data_to_vis, self.training_params_to_vis_transformed = self._get_data_to_vis(
            training_data, params_values_to_vis_training, base_value_to_vis_training)

        val1_data = load_processed_data("val1", path_to_processed_data)
        self.val1_data_to_vis, self.val1_params_to_vis_transformed = self._get_data_to_vis(
            val1_data, params_values_to_vis_val1, base_value_to_vis_val1)

        self.kB_val2_values = kB_val2_values
        self.fC_val2_values = fC_val2_values
        self.LY_val2_values = LY_val2_values

        self.val2_values = np.array([self.kB_val2_values, self.fC_val2_values, self.LY_val2_values], 
                                       dtype=np.float64).T
        self.val2_params_to_vis = np.repeat(self.val2_values, self.n_sources, axis=0)
        self.val2_params_to_vis = torch.tensor(self.scaler.transform(self.val2_params_to_vis), 
                                               dtype=torch.float64)
        self.val2_source_types_to_vis = torch.arange(
            self.n_sources, dtype=torch.int64
        ).unsqueeze(1).repeat(self.params_dim, 1)

        val2_data = []
        for i in range(3):
            val2_i_data = load_processed_data(
                f"val2_{i+1}", path_to_processed_data, val2_rates=True)
            val2_i_data_spectra = val2_i_data[0]
            val2_i_data_spectra = val2_i_data_spectra / val2_i_data_spectra.sum(1)[:, None]
            val2_data.append(val2_i_data_spectra)
        self.val2_rates_to_vis = val2_data
    
    def _get_data_to_vis(
            self, 
            data: tuple, 
            params_values_to_vis: torch.Tensor,
            base_value_to_vis: torch.Tensor
        ):
        spectra_samples_to_vis = []
        params_samples_to_vis = []
        source_types_samples_to_vis = []
        params_samples_to_vis_transformed = []

        spectra, params, source_types = data
        spectra = spectra / spectra.sum(1)[:, None]
        for j in range(self.params_dim):
            if self.n_params_values_to_vis > 1:
                param_vary_condition = torch.logical_or(
                    torch.isclose(params[:, j], params_values_to_vis[0]),
                    torch.isclose(params[:, j], params_values_to_vis[1])
                )
                for i in range(2, self.n_params_values_to_vis):
                    param_vary_condition = torch.logical_or(
                        param_vary_condition,
                        torch.isclose(params[:, j], params_values_to_vis[i])
                    )
                for k in range(self.params_dim):
                    if k != j:
                        param_vary_condition = torch.logical_and(
                            param_vary_condition,
                            torch.isclose(params[:, k], base_value_to_vis)
                        )
            else:
                param_vary_condition = torch.isclose(params[:, j], params_values_to_vis[0])
                for k in range(self.params_dim):
                    if k != j:
                        param_vary_condition = torch.logical_and(
                            param_vary_condition,
                            torch.isclose(params[:, k], base_value_to_vis)
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
    
    def _get_subplot_title(self, params_transformed):
        kB, fC, Y = params_transformed
        title = r"$k_{B}$"+f": {kB:.2f} [g/cm2/GeV], "
        title += r"$f_{C}$"+f": {fC:.3f}, "
        title += r"$Y$"+f": {Y:.0f} [1/MeV]"
        return title
    
    def _get_suptitle_val(self, current_epoch, global_step, val_metric, val_data_type):
        suptitle = f"Validation dataset №{val_data_type}. Epoch: {current_epoch}, "
        suptitle += f"Iteration: {global_step}, "
        suptitle += r"$\rm D^{V_%s}_{C}$ " %val_data_type
        suptitle += f"= {val_metric:.5f}"
        return suptitle

    def _get_suptitle_training(self, current_epoch, global_step, train_loss):
        suptitle = f"Training dataset. Epoch: {current_epoch}, "
        suptitle += f"Iteration: {global_step}, "
        suptitle += r"$\rm D^{T}_{KL}$ "
        suptitle += f"= {train_loss:.5f}"
        return suptitle

    def plot_spectra(
            self,
            spectra_predicted_to_vis: list,
            spectra_true_to_vis: list,
            params_to_vis_transformed: list, 
            current_epoch: int,
            global_step: int,
            metric_value: float,
            dataset_type: str
        ) -> None:
        fig, ax = plt.subplots(self.params_dim, self.n_params_values_to_vis,
                               figsize=(self.params_dim*6, self.n_params_values_to_vis*3))
        ax = ax.flatten()
        for m in range(self.params_dim):
            for i in range(self.n_params_values_to_vis * self.n_sources):
                j = i % self.n_params_values_to_vis + m * self.n_params_values_to_vis
                k = i // self.n_params_values_to_vis

                current_params_transformed = params_to_vis_transformed[m][i]
                subplot_title = self._get_subplot_title(current_params_transformed)

                ########### plot true ###########
                ax[j].stairs(
                    spectra_true_to_vis[m][i], 
                    self.kNPE_bins_edges,
                    label=self.sources_names_to_vis[k],
                    color=self.sources_colors_to_vis[k],
                    alpha=0.6
                )

                ########### plot predicted ###########
                ax[j].stairs(
                    spectra_predicted_to_vis[m][i], 
                    self.kNPE_bins_edges,
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
        
        if dataset_type == 'training':
            suptitle = self._get_suptitle_training(current_epoch, global_step, metric_value)
            fig.suptitle(suptitle, x=0.25, y=0.99, fontsize=20)
            fig.tight_layout()
            fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_tr.png')
            fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_tr.pdf')
        elif dataset_type == 'val1':
            suptitle = self._get_suptitle_val(current_epoch, global_step, metric_value, val_data_type=1)
            fig.suptitle(suptitle, x=0.3, y=0.99, fontsize=20)
            fig.tight_layout()
            fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_v1.png')
            fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_v1.pdf')
        plt.close(fig)

    def plot_rates(
            self,
            rates_predicted_to_vis: torch.Tensor,
            rates_true_to_vis: list,
            current_epoch: int,
            global_step: int,
            metric_value: float,
        ) -> None:
        fig, ax = plt.subplots(1, self.params_dim, 
                               figsize=(self.params_dim*6, self.params_dim*2))
        ax = ax.flatten()
        for i in range(self.params_dim * self.n_sources):
            j = i // self.n_sources
            k = i % self.n_sources
            subplot_title = self._get_subplot_title(
                (self.kB_val2_values[j], self.fC_val2_values[j], self.LY_val2_values[j])
            )

            ########### plot true ###########
            ax[j].stairs(
                rates_true_to_vis[j][k], self.kNPE_bins_edges,
                label=self.sources_names_to_vis[k],
                color=self.sources_colors_to_vis[k],
                linewidth=1.25,
                alpha=0.6
            )

            ########### plot predicted ###########
            ax[j].stairs(
                rates_predicted_to_vis[i], self.kNPE_bins_edges,
                color=self.sources_colors_to_vis[k],
                linestyle='--',
                linewidth=1.25,
                alpha=1.0
            )
            
            if k == 4:
                ax[j].plot([0], [0], color='black', linewidth=2, label="JUNOSW")
                ax[j].plot([0], [0], color='black', linestyle='--', linewidth=2, label="TEDE")
                
                handles, labels = ax[j].get_legend_handles_labels()                
                legend1 = ax[j].legend(handles[:5], labels[:5], frameon=1, ncol=1, fontsize=14, loc="upper right",)
                legend2 = ax[j].legend(handles[5:], labels[5:], frameon=1, ncol=1, fontsize=14, loc=(0.31, 0.82))
                ax[j].add_artist(legend1)
                ax[j].add_artist(legend2)
            
            ax[j].set_title(subplot_title, fontsize=14)
            ax[j].set_ylim(1e-5, 0.4)
            ax[j].set_xlim(0.0, 16.0)
            ax[j].set_yscale("log")
            ax[j].set_xlabel("Number of photo-electrons: " + r"$N_{p.e.} \ / \ 10^3$", fontsize=18)

            if j == 0:
                ax[j].set_ylabel("Prob. density: " + r"$f(N_{p.e.} | k_{B}, f_{C}, Y)$", fontsize=18)

        suptitle = self._get_suptitle_val(current_epoch, global_step, metric_value, val_data_type=2)
        fig.suptitle(suptitle, x=0.3, y=0.99, fontsize=20)
        fig.tight_layout()
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_v2.png')
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_v2.pdf')
        plt.close(fig)

    def plot_rates_with_rel_error(
            self,
            rates_predicted_to_vis: torch.Tensor,
            rates_true_to_vis: list,
            current_epoch: int,
            global_step: int,
            metric_value: float,
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

                predicted = rates_predicted_to_vis[i]
                truth = rates_true_to_vis[j][k]

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
                    (predicted - truth) / (truth + 1e-5),
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
            ax_diff.set_ylim(-1.25, 1.25)
            ax_diff.set_yticks([-1, -0.5, 0, 0.5, 1])
            if j == 0:
                ax_diff.set_ylabel(
                    r"$\Delta = \frac{f_{\rm{TEDE}} - f_{\rm{JUNOSW}}}{f_{\rm{JUNOSW}} + \varepsilon}$",
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

        suptitle = self._get_suptitle_val(current_epoch, global_step, metric_value, val_data_type=2)
        fig.suptitle(suptitle, x=0.35, y=0.99, fontsize=20)
        fig.tight_layout()
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_v2_with_rel_errors.png')
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_v2_with_rel_errors.pdf')
        plt.close(fig)

    def plot_training_process(
            self,
            val1_metric_to_plot: list,
            val2_metric_to_plot: list,
            val_metric_to_plot: list,
            train_loss_to_plot: list,
            train_loss_value: float,
            val1_metric_value: float,
            val2_metric_value: float,
            val_metric_value: float,
            global_step: int,
            current_epoch: int
        ) -> None:
        title = r'$\rm D^{T}_{KL} = $' + f"{train_loss_value:.5f}, "
        title += r'$\rm D^{V_1}_{C} = $' + f"{val1_metric_value:.5f}, "
        title += r'$\rm D^{V_2}_{C} = $' + f"{val2_metric_value:.5f}, "
        title += r'$\rm D^{V}_{C} = \rm \frac{D^{V_1}_{C} + 4 \cdot \rm D^{V_2}_{C}}{5}$' + f' = {val_metric_value:.5f}'

        fig, ax1 = plt.subplots(1, 1, figsize=(16, 5))

        ax1.plot(
            np.arange(1, len(train_loss_to_plot)+1),
            train_loss_to_plot,
            label="Training data: " + r'$\rm D^{C}_{T}$',
            color='black',
            alpha=0.9,
            linewidth=1.25
        )
        ax1.set_ylabel("Training Loss", fontsize=16, color='black')
        ax1.set_xlabel('Epoch', fontsize=16)
        ax1.set_yscale("log")
        # ax1.set_ylim(1e-4, 0.15)
        ax1.tick_params(axis='y', labelsize=14, labelcolor='black')

        ax2 = ax1.twinx()
        ax2.plot(
            np.arange(1, len(val1_metric_to_plot)),
            val1_metric_to_plot[1:],
            label="Validation dataset №1: " + r'$\rm D^{V_1}_{C}$',
            color='royalblue',
            alpha=0.9,
            linewidth=1.25
        )
        ax2.plot(
            np.arange(1, len(val2_metric_to_plot)),
            val2_metric_to_plot[1:],
            label="Validation dataset №2: " + r'$\rm D^{V_2}_{C}$',
            color='darkred',
            alpha=0.9,
            linewidth=1.25
        )
        ax2.plot(
            np.arange(1, len(val_metric_to_plot)),
            val_metric_to_plot[1:],
            label="Validation datasets combined: " + r'$\rm D^{V}_{C}$',
            color='darkgreen',
            alpha=0.9,
            linewidth=1.25
        )
        ax2.set_ylabel("Validation Metrics", fontsize=16, color='green')
        ax2.tick_params(axis='y', labelsize=14, labelcolor='green')

        # Legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=16)

        fig.suptitle(title, x=0.35, y=0.99, fontsize=20)
        fig.tight_layout()
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_training_process.png') 
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_training_process.pdf') 
        plt.show()
