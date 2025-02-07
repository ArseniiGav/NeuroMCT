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
    
    def _get_suptitle_val(self, current_epoch, global_step, val_metrics, metric_names, val_data_type):
        suptitle = f"Validation dataset №{val_data_type}. Epoch: {current_epoch}, "
        suptitle += f"Iteration: {global_step}, "
        for name in metric_names:
            if name == "wasserstein":
                suptitle += r"$d^{V_%s}_{1}$ " %val_data_type
                suptitle += f"= {val_metrics[name]:.4f}; "
            elif name == "cramer":
                suptitle += r"$d^{V_%s}_{2}$ " %val_data_type
                suptitle += f"= {val_metrics[name]:.4f}; "
            elif name == "ks":
                suptitle += r"$d^{V_%s}_{\infty}$ " %val_data_type
                suptitle += f"= {val_metrics[name]:.4f}"
        return suptitle

    def _get_suptitle_training(self, current_epoch, global_step, train_loss):
        suptitle = f"Training dataset. Epoch: {current_epoch}, "
        suptitle += f"Iteration: {global_step}, "
        suptitle += r"$L^{T}_{\rm KL}$ "
        suptitle += f"= {train_loss:.4f}"
        return suptitle

    def plot_spectra(
            self,
            spectra_predicted_to_vis: list,
            spectra_true_to_vis: list,
            params_to_vis_transformed: list, 
            current_epoch: int,
            global_step: int,
            metric_value: float,
            dataset_type: str,
            **kwargs,
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
                ax[j].set_ylim(5e-5, 0.25)
                ax[j].set_yscale("log")
                ax[j].set_xlim(0.0, 16.0)
                
                if j >= self.n_params_values_to_vis * (self.params_dim - 1):
                    ax[j].set_xlabel("Number of photo-electrons: " + r"$N_{p.e.} \ / \ 10^3$")
                if j % self.n_params_values_to_vis == 0:
                    ax[j].set_ylabel("Prob. density: " + r"$f(N_{p.e.} | k_{B}, f_{C}, Y)$")
        
        if dataset_type == 'training':
            suptitle = self._get_suptitle_training(current_epoch, global_step, metric_value)
            fig.suptitle(suptitle, x=0.3, y=0.99, fontsize=20)
            fig.tight_layout()
            fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_tr.png')
            fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_tr.pdf')
        elif dataset_type == 'val1':
            metric_names = kwargs["metric_names"]
            suptitle = self._get_suptitle_val(
                current_epoch, global_step, metric_value, metric_names, val_data_type=1)
            fig.suptitle(suptitle, x=0.4, y=0.99, fontsize=20)
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
            metric_names: list,
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
            ax[j].set_ylim(5e-6, 0.5)
            ax[j].set_xlim(0.0, 16.0)
            ax[j].set_yscale("log")
            ax[j].set_xlabel("Number of photo-electrons: " + r"$N_{p.e.} \ / \ 10^3$", fontsize=18)

            if j == 0:
                ax[j].set_ylabel("Prob. density: " + r"$f(N_{p.e.} | k_{B}, f_{C}, Y)$", fontsize=18)

        suptitle = self._get_suptitle_val(
            current_epoch, global_step, metric_value, metric_names, val_data_type=2)
        fig.suptitle(suptitle, x=0.4, y=0.99, fontsize=20)
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
            metric_names: list,
        ) -> None:
        fig, axes = plt.subplots(2, self.params_dim, 
                                 figsize=(self.params_dim * 6, self.params_dim * 2.25),
                                 gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})
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
                    (predicted - truth) / torch.sqrt(truth),
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
                    legend2 = ax_main.legend(handles[5:], labels[5:], frameon=1, ncol=1, fontsize=12, loc=(0.27, 0.825))
                    ax_main.add_artist(legend1)
                    ax_main.add_artist(legend2)

            ax_diff.set_xlim(0.0, 16.0)
            ax_diff.set_xlabel("Number of photo-electrons: " + r"$N_{p.e.} \ / \ 10^3$", fontsize=16)
            ax_diff.set_ylim(-0.05, 0.05)
            ax_diff.set_yticks([-0.04, -0.02, 0, 0.02, 0.04])
            if j == 0:
                ax_diff.set_ylabel(
                    r"$\Delta = \frac{f_{\rm{TEDE}} - f_{\rm{JUNOSW}}}{\sqrt{f_{\rm{JUNOSW}}}}$",
                    fontsize=17
                )

            ax_main.set_ylim(5e-6, 0.5)
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

        suptitle = self._get_suptitle_val(
            current_epoch, global_step, metric_value, metric_names, val_data_type=2)
        fig.suptitle(suptitle, x=0.45, y=0.99, fontsize=20)
        fig.tight_layout()
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_v2_with_rel_errors.png')
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_v2_with_rel_errors.pdf')
        plt.close(fig)

    def plot_training_process(
            self,
            train_loss_to_plot: list,
            train_loss_value: float,
            global_step: int,
            current_epoch: int
        ) -> None:
        title = "The loss averaged over the last batch: " + r'$L^{B}_{\rm KL} = $' + f"{train_loss_value:.4f}"
        label = (r"$L_{\rm KL} = "
                 r"\frac{1}{|B| \cdot N_b} \sum_{i=1}^{|B|} \ \sum_{j=1}^{N_b} "
                 r"f\left(\mathbf{X}_{i,j}^{\rm{JUNOSW}}\right) \cdot "
                 r"\log\left( "
                 r"\frac{f\left(\mathbf{X}_{i,j}^{\rm{JUNOSW}}\right)}{f\left(\mathbf{X}_{i,j}^{\rm{TEDE}}\right)} "
                 r"\right)$")

        x_to_plot = np.arange(1, global_step+1)
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(
            x_to_plot,
            train_loss_to_plot,
            label=label,
            color='black',
            alpha=0.7,
            linewidth=1.25
        )
        ax.set_ylabel("KL-divergence loss: " + r"$L_{\rm KL}$", fontsize=16, color='black')
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_yscale("log")
        ax.set_ylim(5e-3, 2.0)
        ax.tick_params(axis='y', labelsize=14, labelcolor='black')
        ax.legend(loc="upper right", fontsize=16)

        fig.suptitle(title, x=0.3, y=0.975, fontsize=18)
        fig.tight_layout()
        fig.savefig(f"{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_training_process.png") 
        fig.savefig(f"{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_training_process.pdf") 
        plt.close(fig)

    def plot_val_metrics(
            self,
            val1_metrics_to_plot: list,
            val2_metrics_to_plot: list,
            val_metrics_to_plot: list,
            val1_metric_value: float,
            val2_metric_value: float,
            val_metric_value: float,
            global_step: int,
            current_epoch: int,
            val_metric_name: str
        ) -> None:
        if val_metric_name == "wasserstein":
            title = r"$d^{V_1}_{1}$ " + f"= {val1_metric_value:.4f}; "
            title += r"$d^{V_2}_{1}$ " + f"{val2_metric_value:.4f}; "
            title += r"$d^{V}_{1} = \frac{1}{2} \left(d^{V_1}_{1} + d^{V_2}_{1}\right)$"
            title += f' = {val_metric_value:.4f}'

            ylabel = "Wasserstein distance: " + r"$d^{V}_{1}$"
        elif val_metric_name == "cramer":
            title = r"$d^{V_1}_{2}$ " + f"= {val1_metric_value:.4f}; "
            title += r"$d^{V_2}_{2}$ " + f"{val2_metric_value:.4f}; "
            title += r"$d^{V}_{2} = \frac{1}{2} \left(d^{V_1}_{2} + d^{V_2}_{2}\right)$"
            title += f' = {val_metric_value:.4f}'

            ylabel = "Cramér-von Mises distance: " + r"$d^{V}_{1}$"
        elif val_metric_name == "ks":
            title = r"$d^{V_1}_{\infty}$ " + f"= {val1_metric_value:.4f}; "
            title += r"$d^{V_2}_{\infty}$ " + f"{val2_metric_value:.4f}; "
            title += r"$d^{V}_{\infty} = \frac{1}{2} \left(d^{V_1}_{\infty} + d^{V_2}_{\infty}\right)$"
            title += f' = {val_metric_value:.4f}'

            ylabel = "Kolmogorov-Smirnov distance: " + r"$d^{V}_{\infty}$"

        x_to_plot = np.arange(1, current_epoch+2)
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(
            x_to_plot,
            val1_metrics_to_plot[1:],
            color='royalblue',
            label="Validation dataset №1",
            alpha=0.9,
            linewidth=1.5
        )
        ax.plot(
            x_to_plot,
            val2_metrics_to_plot[1:],
            label="Validation dataset №2",
            color='darkred',
            alpha=0.9,
            linewidth=1.5
        )
        ax.plot(
            x_to_plot,
            val_metrics_to_plot[1:],
            label="Validation datasets average",
            color='darkgreen',
            alpha=0.9,
            linewidth=1.5
        )
        ax.set_ylabel(ylabel, fontsize=16, color='black')
        ax.set_xlabel('Epoch', fontsize=16)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 2e-1)
        ax.tick_params(axis='y', labelsize=14, labelcolor='black')
        ax.legend(loc="upper right", fontsize=15)

        fig.suptitle(title, x=0.3, y=0.975, fontsize=16)
        fig.tight_layout()
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_val_metric_{val_metric_name}.png') 
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_val_metric_{val_metric_name}.pdf') 
        plt.close(fig)

    def plot_val_metrics_combined(
            self,
            val_metrics_to_plot: dict,
            val_metric_values: dict,
            global_step: int,
            current_epoch: int,
            val_metric_names: list
        ) -> None:
        ylabel = "Validation metrics: " + r"$d^{V}_{p}$"
        x_to_plot = np.arange(1, current_epoch+2)
        title = ""

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        for val_metric_name in val_metric_names:
            if val_metric_name == "wasserstein":
                color = "indigo"
                title += r"$d^{V}_{2}$ " + f"= {val_metric_values[val_metric_name]:.4f}; "
                label = "Wasserstein distance: " + r"$d_2^{V}$"
            elif val_metric_name == "cramer":
                color = "#3971ac"
                title += r"$d^{V}_{1}$ " + f"= {val_metric_values[val_metric_name]:.4f}; "
                label = "Cramér-von Mises distance: " + r"$d_1^{V}$"
            elif val_metric_name == "ks":
                color = "lightslategrey"
                title += r"$d^{V}_{\infty}$ " + f"= {val_metric_values[val_metric_name]:.4f}; "
                label = "Kolmogorov-Smirnov distance: " + r"$d_{\infty}^{V}$"

            ax.plot(
                x_to_plot,
                val_metrics_to_plot[val_metric_name][1:],
                label=label,
                color=color,
                alpha=0.9,
                linewidth=1.5
            )
        title = title[:-2] # remove "; " at the end

        ax.set_ylabel(ylabel, fontsize=16, color='black')
        ax.set_xlabel('Epoch', fontsize=16)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 2e-1)
        ax.tick_params(axis='y', labelsize=14, labelcolor='black')
        ax.legend(loc="upper right", fontsize=16)

        fig.suptitle(title, x=0.225, y=0.975, fontsize=16)
        fig.tight_layout()
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_val_metrics.png') 
        fig.savefig(f'{self.path_to_plots}/tede_training/epoch_{current_epoch}_it_{global_step}_val_metrics.pdf') 
        plt.close(fig)
