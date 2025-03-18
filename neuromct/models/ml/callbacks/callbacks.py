import torch
from lightning.pytorch.callbacks import Callback

from ....plot import ModelResultsVisualizer


class ModelResultsVisualizerCallback(Callback):
    def __init__(
            self, 
            res_visualizer: ModelResultsVisualizer, 
            approach_type: str,
            base_path_to_savings: str,
            plots_dir_name: str,
            predictions_dir_name: str,
            val_metric_names: str
        ):
        super().__init__()
        self.res_visualizer = res_visualizer
        self.approach_type = approach_type
        self.path_to_plot_savings = base_path_to_savings + "/" + plots_dir_name
        self.path_to_predictions_savings = base_path_to_savings + "/" + predictions_dir_name

        self.plot_every_n_train_epochs = self.res_visualizer.plot_every_n_train_epochs

        training_data_to_vis = self.res_visualizer.training_data_to_vis
        self.training_spectra_to_vis = training_data_to_vis[0]
        self.training_params_to_vis = training_data_to_vis[1]
        self.training_source_types_to_vis = training_data_to_vis[2]
        self.training_params_to_vis_transformed = self.res_visualizer.training_params_to_vis_transformed

        val1_data_to_vis = self.res_visualizer.val1_data_to_vis
        self.val1_spectra_to_vis = val1_data_to_vis[0]
        self.val1_params_to_vis = val1_data_to_vis[1]
        self.val1_source_types_to_vis = val1_data_to_vis[2]
        self.val1_params_to_vis_transformed = self.res_visualizer.val1_params_to_vis_transformed

        self.val2_rates_to_vis = self.res_visualizer.val2_rates_to_vis
        self.val2_params_to_vis = self.res_visualizer.val2_params_to_vis
        self.val2_source_types_to_vis = self.res_visualizer.val2_source_types_to_vis

        self.x_values = torch.linspace(
            0.4, 16.4, 2000, dtype=torch.float64)

        self.val_metric_names = val_metric_names
        self.val1_metrics_to_plot = {key: [] for key in self.val_metric_names}
        self.val2_metrics_to_plot = {key: [] for key in self.val_metric_names}
        self.val_metrics_to_plot = {key: [] for key in self.val_metric_names}

    def _save_spectra_plots(
            self, 
            train_loss_value,
            val1_metrics_values,
            val2_metrics_values,
            training_spectra, 
            val1_spectra, 
            val2_spectra, 
            current_epoch,
            global_step,
            val_metric_names
        ) -> None:
        self.res_visualizer.plot_spectra(
            spectra_predicted_to_vis=training_spectra,
            spectra_true_to_vis=self.training_spectra_to_vis,
            params_to_vis_transformed=self.training_params_to_vis_transformed,
            current_epoch=current_epoch,
            global_step=global_step,
            metric_value=train_loss_value,
            dataset_type='training',
            path_to_save=self.path_to_plot_savings,
            approach_type=self.approach_type
        )

        self.res_visualizer.plot_spectra(
            spectra_predicted_to_vis=val1_spectra,
            spectra_true_to_vis=self.val1_spectra_to_vis,
            params_to_vis_transformed=self.val1_params_to_vis_transformed,
            current_epoch=current_epoch,
            global_step=global_step,
            metric_value=val1_metrics_values,
            dataset_type='val1',
            metric_names=val_metric_names,
            path_to_save=self.path_to_plot_savings,
            approach_type=self.approach_type
        )

        self.res_visualizer.plot_rates(
            rates_predicted_to_vis=val2_spectra,
            rates_true_to_vis=self.val2_rates_to_vis,
            current_epoch=current_epoch,
            global_step=global_step,
            metric_value=val2_metrics_values,
            metric_names=val_metric_names,
            path_to_save=self.path_to_plot_savings,
            approach_type=self.approach_type
        )

        if self.approach_type == 'tede':
            self.res_visualizer.plot_rates_with_rel_error(
                rates_predicted_to_vis=val2_spectra,
                rates_true_to_vis=self.val2_rates_to_vis,
                current_epoch=current_epoch,
                global_step=global_step,
                metric_value=val2_metrics_values,
                metric_names=val_metric_names,
                path_to_save=self.path_to_plot_savings
            )

    def _save_training_process_plots(
            self,
            train_loss_to_plot,
            val1_metrics_to_plot,
            val2_metrics_to_plot,
            val_metrics_to_plot,
            train_loss_value,
            val1_metrics_values,
            val2_metrics_values,
            val_metrics_values,
            current_epoch,
            global_step,
            val_metric_names
        ) -> None:
        self.res_visualizer.plot_training_process(
            train_loss_to_plot=train_loss_to_plot,
            train_loss_value=train_loss_value,
            global_step=global_step,
            current_epoch=current_epoch,
            path_to_save=self.path_to_plot_savings,
            approach_type=self.approach_type
        )

        for name in val_metric_names:
            self.res_visualizer.plot_val_metrics(
                val1_metrics_to_plot=val1_metrics_to_plot[name],
                val2_metrics_to_plot=val2_metrics_to_plot[name],
                val_metrics_to_plot=val_metrics_to_plot[name],
                val1_metric_value=val1_metrics_values[name],
                val2_metric_value=val2_metrics_values[name],
                val_metric_value=val_metrics_values[name],
                global_step=global_step,
                current_epoch=current_epoch,
                val_metric_name=name,
                path_to_save=self.path_to_plot_savings
            )

        self.res_visualizer.plot_val_metrics_combined(
            val_metrics_to_plot=val_metrics_to_plot,
            val_metric_values=val_metrics_values,
            global_step=global_step,
            current_epoch=current_epoch,
            val_metric_names=val_metric_names,
            path_to_save=self.path_to_plot_savings
        )

    def _get_predictions(
            self, 
            model, 
            device, 
            current_epoch, 
            global_step
        ):
        model.eval()
        with torch.no_grad():
            if self.approach_type == 'tede':
                training_spectra_list = []
                for i in range(len(self.training_params_to_vis)):
                    training_spectra = model(
                        self.training_params_to_vis[i].to(device),
                        self.training_source_types_to_vis[i].to(device)
                    ).detach().cpu()
                    training_spectra_list.append(training_spectra)

                val1_spectra_list = []
                for i in range(len(self.val1_params_to_vis)):
                    val1_spectra = model(
                        self.val1_params_to_vis[i].to(device),
                        self.val1_source_types_to_vis[i].to(device),
                        
                    ).detach().cpu()
                    val1_spectra_list.append(val1_spectra)

                val2_spectra = model(
                    self.val2_params_to_vis.to(device),
                    self.val2_source_types_to_vis.to(device)
                ).detach().cpu()

            elif self.approach_type == 'nfde':
                x = self.x_values.to(device=device)

                training_spectra_list = []
                for i in range(len(self.training_params_to_vis)):
                    training_spectra_per_cond_set = []
                    for j in range(len(self.training_params_to_vis[i])):
                        prob_x = torch.exp(
                            model.log_prob(
                                x, 
                                self.training_params_to_vis[i][j].to(device), 
                                self.training_source_types_to_vis[i][j].to(device)
                            )
                        ).detach().cpu()
                        training_spectra_per_cond_set.append(prob_x)
                    training_spectra_list.append(training_spectra_per_cond_set)

                val1_spectra_list = []
                for i in range(len(self.val1_params_to_vis)):
                    val1_spectra_per_cond_set = []
                    for j in range(len(self.val1_params_to_vis[i])):
                        prob_x = torch.exp(
                            model.log_prob(
                                x, 
                                self.val1_params_to_vis[i][j].to(device),
                                self.val1_source_types_to_vis[i][j].to(device)
                            )
                        ).detach().cpu()
                        val1_spectra_per_cond_set.append(prob_x)
                    val1_spectra_list.append(val1_spectra_per_cond_set)

                val2_spectra = []
                for i in range(len(self.val2_params_to_vis)):
                    val2_spectra_per_cond_set = torch.exp(
                        model.log_prob(
                            x, 
                            self.val2_params_to_vis[i].to(device), 
                            self.val2_source_types_to_vis[i].to(device)
                        )
                    ).detach().cpu()
                    val2_spectra.append(val2_spectra_per_cond_set)
        model.train()

        # save predictions
        file_name_base = f"{self.path_to_predictions_savings}/epoch_{current_epoch}_it_{global_step}"
        torch.save(training_spectra_list, f'{file_name_base}_tr.pt')
        torch.save(val1_spectra_list, f'{file_name_base}_v1.pt')
        torch.save(val2_spectra, f'{file_name_base}_v2.pt')
        return (training_spectra_list, val1_spectra_list, val2_spectra)

    def on_train_epoch_end(self, trainer, pl_training_module):
        train_loss_value = trainer.callback_metrics["training_loss"].item()

        val1_metrics_values = dict()
        val2_metrics_values = dict()
        val_metrics_values = dict()

        for name in self.val_metric_names:
            val1_metrics_value = trainer.callback_metrics[f"val1_{name}_metric"].item()
            val2_metrics_value = trainer.callback_metrics[f"val2_{name}_metric"].item()
            val_metrics_value = trainer.callback_metrics[f"val_{name}_metric"].item()

            val1_metrics_values[name] = val1_metrics_value
            val2_metrics_values[name] = val2_metrics_value
            val_metrics_values[name] = val_metrics_value

        if trainer.is_global_zero:
            device = pl_training_module.device
            current_epoch = pl_training_module.current_epoch
            global_step = pl_training_module.global_step

            for name in self.val_metric_names:
                self.val1_metrics_to_plot[name].append(val1_metrics_values[name])
                self.val2_metrics_to_plot[name].append(val2_metrics_values[name])
                self.val_metrics_to_plot[name].append(val_metrics_values[name])

            condition_to_plot = (
                global_step > 0 and \
                current_epoch % self.plot_every_n_train_epochs == 0
            )
            if condition_to_plot:
                training_spectra, val1_spectra, val2_spectra = self._get_predictions(
                    model=pl_training_module.model, 
                    device=device, 
                    current_epoch=current_epoch, 
                    global_step=global_step
                )
                self._save_spectra_plots(
                    train_loss_value,
                    val1_metrics_values,
                    val2_metrics_values,
                    training_spectra,
                    val1_spectra,
                    val2_spectra,
                    current_epoch,
                    global_step,
                    self.val_metric_names
                )

    def on_train_end(self, trainer, pl_training_module):
        train_loss_value = trainer.callback_metrics["training_loss"].item()

        val1_metrics_values = dict()
        val2_metrics_values = dict()
        val_metrics_values = dict()

        for name in self.val_metric_names:
            val1_metrics_value = trainer.callback_metrics[f"val1_{name}_metric"].item()
            val2_metrics_value = trainer.callback_metrics[f"val2_{name}_metric"].item()
            val_metrics_value = trainer.callback_metrics[f"val_{name}_metric"].item()

            val1_metrics_values[name] = val1_metrics_value
            val2_metrics_values[name] = val2_metrics_value
            val_metrics_values[name] = val_metrics_value

        if trainer.is_global_zero:
            current_epoch = pl_training_module.current_epoch - 1 # counting correction
            global_step = pl_training_module.global_step
            device = pl_training_module.device

            train_loss_to_plot = pl_training_module.train_loss_to_plot

            training_spectra, val1_spectra, val2_spectra = self._get_predictions(
                model=pl_training_module.model, 
                device=device, 
                current_epoch=current_epoch, 
                global_step=global_step
            )
            self._save_spectra_plots(
                train_loss_value,
                val1_metrics_values,
                val2_metrics_values,
                training_spectra,
                val1_spectra,
                val2_spectra,
                current_epoch,
                global_step,
                self.val_metric_names
            )
            self._save_training_process_plots(
                train_loss_to_plot,
                self.val1_metrics_to_plot,
                self.val2_metrics_to_plot,
                self.val_metrics_to_plot,
                train_loss_value,
                val1_metrics_values,
                val2_metrics_values,
                val_metrics_values,
                current_epoch,
                global_step,
                self.val_metric_names
            )
