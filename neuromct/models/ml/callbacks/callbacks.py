import torch
from lightning.pytorch.callbacks import Callback

from ....plot import ModelResultsVisualizer


class ModelResultsVisualizerCallback(Callback):
    def __init__(
            self, 
            res_visualizer: ModelResultsVisualizer, 
            base_path_to_savings: str,
            plots_dir_name: str,
            predictions_dir_name: str,
        ):
        super().__init__()
        self.res_visualizer = res_visualizer
        self.path_to_plot_savings = base_path_to_savings + "/" + plots_dir_name
        self.path_to_predictions_savings = base_path_to_savings + "/" + predictions_dir_name

        self.plot_every_n_steps = self.res_visualizer.plot_every_n_steps

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
            path_to_save=self.path_to_plot_savings
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
            path_to_save=self.path_to_plot_savings
        )

        self.res_visualizer.plot_rates(
            rates_predicted_to_vis=val2_spectra,
            rates_true_to_vis=self.val2_rates_to_vis,
            current_epoch=current_epoch,
            global_step=global_step,
            metric_value=val2_metrics_values,
            metric_names=val_metric_names,
            path_to_save=self.path_to_plot_savings
        )

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
            path_to_save=self.path_to_plot_savings
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
        model.train()

        # save predictions
        file_name_base = f"{self.path_to_predictions_savings}/epoch_{current_epoch}_it_{global_step}"
        torch.save(training_spectra_list, f'{file_name_base}_tr.pt')
        torch.save(val1_spectra_list, f'{file_name_base}_v1.pt')
        torch.save(val2_spectra, f'{file_name_base}_v2.pt')
        return (training_spectra_list, val1_spectra_list, val2_spectra)

    def on_train_epoch_end(self, trainer, pl_training_module):
        current_epoch = pl_training_module.current_epoch
        global_step = pl_training_module.global_step

        condition_to_plot = (
            global_step > 0 and \
            current_epoch % self.plot_every_n_steps == 0
        )
        if condition_to_plot:
            train_loss_value = trainer.callback_metrics["training_loss"].item()
            val1_metrics_values = pl_training_module.val1_metrics_values
            val2_metrics_values = pl_training_module.val2_metrics_values
            val_metric_names = pl_training_module.val_metric_names
            device = pl_training_module.device

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
                val_metric_names
            )

    def on_train_end(self, trainer, pl_training_module):
        train_loss_value = trainer.callback_metrics["training_loss"].item()
        current_epoch = pl_training_module.current_epoch
        global_step = pl_training_module.global_step
        train_loss_to_plot = pl_training_module.train_loss_to_plot
        val1_metrics_to_plot = pl_training_module.val1_metrics_to_plot
        val2_metrics_to_plot = pl_training_module.val2_metrics_to_plot
        val_metrics_to_plot = pl_training_module.val_metrics_to_plot
        val1_metrics_values = pl_training_module.val1_metrics_values
        val2_metrics_values = pl_training_module.val2_metrics_values
        val_metrics_values = pl_training_module.val_metrics_values
        val_metric_names = pl_training_module.val_metric_names
        device = pl_training_module.device

        training_spectra, val1_spectra, val2_spectra = self._get_predictions(
            model=pl_training_module.model, device=device)
        self._save_spectra_plots(
            train_loss_value,
            val1_metrics_values,
            val2_metrics_values,
            training_spectra,
            val1_spectra,
            val2_spectra,
            current_epoch,
            global_step,
            val_metric_names
        )
        self._save_training_process_plots(
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
        )
