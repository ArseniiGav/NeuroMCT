import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule


class TEDELightningTraining(LightningModule):
    def __init__(self,
            model: nn.Module,
            loss_function: nn.Module,
            val_metric_functions: dict[str, nn.Module],
            optimizer: optim.Optimizer,
            lr_scheduler: optim.lr_scheduler.LRScheduler,
            lr: float,
            bins_centers: torch.Tensor,
            **kwargs,
        ):
        super(TEDELightningTraining, self).__init__()

        self.model = model
        self.loss_function = loss_function
        self.val_metric_functions = val_metric_functions
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr = lr
        self.bins_centers = bins_centers
        
        self.val_metric_names = list(val_metric_functions.keys())

        self.kwargs = kwargs
        if self.kwargs['res_visualizator']:
            self.val1_metrics_to_plot = {key: [] for key in self.val_metric_names}
            self.val2_metrics_to_plot = {key: [] for key in self.val_metric_names}
            self.val_metrics_to_plot = {key: [] for key in self.val_metric_names}
            self.train_loss_to_plot = []

            self.res_visualizator = kwargs['res_visualizator']

            training_data_to_vis = self.res_visualizator.training_data_to_vis
            self.training_spectra_to_vis = training_data_to_vis[0]
            self.training_params_to_vis = training_data_to_vis[1]
            self.training_source_types_to_vis = training_data_to_vis[2]
            self.training_params_to_vis_transformed = self.res_visualizator.training_params_to_vis_transformed

            val1_data_to_vis = self.res_visualizator.val1_data_to_vis
            self.val1_spectra_to_vis = val1_data_to_vis[0]
            self.val1_params_to_vis = val1_data_to_vis[1]
            self.val1_source_types_to_vis = val1_data_to_vis[2]
            self.val1_params_to_vis_transformed = self.res_visualizator.val1_params_to_vis_transformed

            self.val2_rates_to_vis = self.res_visualizator.val2_rates_to_vis
            self.val2_params_to_vis = self.res_visualizator.val2_params_to_vis
            self.val2_source_types_to_vis = self.res_visualizator.val2_source_types_to_vis

    def _compute_and_log_losses(self, spectra_predict, spectra_true, data_type):
        loss = self.loss_function(spectra_predict, spectra_true)
        self.log(f"{data_type}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _compute_and_log_val_metrics(self, spectra_predict, spectra_true, data_type):
        bins_centers_repeated = self.bins_centers.repeat(
            (spectra_predict.shape[0], 1)).to(self.device)

        metrics = dict()
        for name, function in self.val_metric_functions.items():
            metric = function(bins_centers_repeated, bins_centers_repeated, 
                              spectra_predict, spectra_true)
            self.log(f"{data_type}_{name}_metric", metric, prog_bar=True, on_epoch=True)
            metrics[name] = metric.item()
        return metrics

    def configure_optimizers(self):      
        opt = self.optimizer(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-3)
        scheduler = self.lr_scheduler(opt, mode='min', factor=0.9, patience=5, verbose=False) # depends on lr_scheduler. Needs more flexibility
        return [opt], [{'scheduler': scheduler, 'monitor': "val_cramer_metric"}]

    def forward(self, params, source_types, t_out=False):
        return self.model(params, source_types, t_out=t_out)

    def training_step(self, batch):
        spectra_true, params, source_types = batch
        spectra_predict = self(params, source_types)
        loss = self._compute_and_log_losses(spectra_predict, spectra_true, "training")
        if self.kwargs['res_visualizator']:
            self.train_loss_to_plot.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        spectra_true, params, source_types = batch
        spectra_predict = self(params, source_types)
        if dataloader_idx == 0:
            metrics_values = self._compute_and_log_val_metrics(spectra_predict, spectra_true, "val1")
            self.val1_metrics_values = metrics_values
            if self.kwargs['res_visualizator']:
                for name, value in metrics_values.items():
                    self.val1_metrics_to_plot[name].append(value)
        elif dataloader_idx == 1:
            metrics_values = self._compute_and_log_val_metrics(spectra_predict, spectra_true, "val2")
            self.val2_metrics_values = metrics_values
            if self.kwargs['res_visualizator']:
                for name, value in metrics_values.items():
                    self.val2_metrics_to_plot[name].append(value)

    def on_validation_epoch_end(self):
        self.val_metrics_values = dict()
        for name in self.val_metric_names:
            self.val_metrics_values[name] = (
                self.val1_metrics_values[name] + self.val2_metrics_values[name]) / 2
            self.log(f"val_{name}_metric", self.val_metrics_values[name], prog_bar=True)
            if self.kwargs['res_visualizator']:
                self.val_metrics_to_plot[name].append(self.val_metrics_values[name])

    def on_train_epoch_end(self):
        if self.kwargs['res_visualizator']:
            train_loss_value = self.trainer.callback_metrics["training_loss"].item()
            if self.global_step > 0:
                self.eval()
                with torch.no_grad():
                    training_spectra_pdfs_to_vis = []
                    for i in range(len(self.training_params_to_vis)):
                        training_spectra_pdf_to_vis = self(
                            self.training_params_to_vis[i].to(self.device),
                            self.training_source_types_to_vis[i].to(self.device)
                        ).detach().cpu()
                        training_spectra_pdfs_to_vis.append(training_spectra_pdf_to_vis)

                    val1_spectra_pdfs_to_vis = []
                    for i in range(len(self.val1_params_to_vis)):
                        val1_spectra_pdf_to_vis = self(
                            self.val1_params_to_vis[i].to(self.device),
                            self.val1_source_types_to_vis[i].to(self.device),
                            
                        ).detach().cpu()
                        val1_spectra_pdfs_to_vis.append(val1_spectra_pdf_to_vis)

                    val2_spectra_pdfs_to_vis = self(
                        self.val2_params_to_vis.to(self.device),
                        self.val2_source_types_to_vis.to(self.device)
                    ).detach().cpu()
                self.train()

                self.res_visualizator.plot_spectra(
                    spectra_predicted_to_vis=training_spectra_pdfs_to_vis,
                    spectra_true_to_vis=self.training_spectra_to_vis,
                    params_to_vis_transformed=self.training_params_to_vis_transformed,
                    current_epoch=self.current_epoch,
                    global_step=self.global_step,
                    metric_value=train_loss_value,
                    dataset_type='training'
                )

                self.res_visualizator.plot_spectra(
                    spectra_predicted_to_vis=val1_spectra_pdfs_to_vis,
                    spectra_true_to_vis=self.val1_spectra_to_vis,
                    params_to_vis_transformed=self.val1_params_to_vis_transformed,
                    current_epoch=self.current_epoch,
                    global_step=self.global_step,
                    metric_value=self.val1_metrics_values,
                    dataset_type='val1',
                    metric_names=self.val_metric_names
                )

                self.res_visualizator.plot_rates(
                    rates_predicted_to_vis=val2_spectra_pdfs_to_vis,
                    rates_true_to_vis=self.val2_rates_to_vis,
                    current_epoch=self.current_epoch,
                    global_step=self.global_step,
                    metric_value=self.val2_metrics_values,
                    metric_names=self.val_metric_names
                )

                self.res_visualizator.plot_rates_with_rel_error(
                    rates_predicted_to_vis=val2_spectra_pdfs_to_vis,
                    rates_true_to_vis=self.val2_rates_to_vis,
                    current_epoch=self.current_epoch,
                    global_step=self.global_step,
                    metric_value=self.val2_metrics_values,
                    metric_names=self.val_metric_names
                )

                self.res_visualizator.plot_training_process(
                    train_loss_to_plot=self.train_loss_to_plot,
                    train_loss_value=train_loss_value,
                    global_step=self.global_step,
                    current_epoch=self.current_epoch
                )
                for name in self.val_metric_names:
                    self.res_visualizator.plot_val_metrics(
                        val1_metrics_to_plot=self.val1_metrics_to_plot[name],
                        val2_metrics_to_plot=self.val2_metrics_to_plot[name],
                        val_metrics_to_plot=self.val_metrics_to_plot[name],
                        val1_metric_value=self.val1_metrics_values[name],
                        val2_metric_value=self.val2_metrics_values[name],
                        val_metric_value=self.val_metrics_values[name],
                        global_step=self.global_step,
                        current_epoch=self.current_epoch,
                        val_metric_name=name
                    )
