import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule


class TEDELightningTraining(LightningModule):
    def __init__(self,
             model: nn.Module,
             loss_function: nn.Module,
             val_metric_function: nn.Module,
             optimizer: optim.Optimizer,
             lr_scheduler: optim.lr_scheduler.LRScheduler,
             lr: float,
             **kwargs,
        ):
        super(TEDELightningTraining, self).__init__()

        self.model = model
        self.loss_function = loss_function
        self.val_metric_function = val_metric_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr = lr
        
        self.train_loss_to_plot = []
        self.val1_metric_to_plot = []
        self.val2_metric_to_plot = []
        self.val_metric_to_plot = []

        self.kwargs = kwargs
        if kwargs['res_visualizator']:
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
        metric = self.val_metric_function(spectra_predict, spectra_true)
        self.log(f"{data_type}_metric", metric, prog_bar=True, on_epoch=True)
        return metric

    def configure_optimizers(self):      
        opt = self.optimizer(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler = self.lr_scheduler(opt, mode='min', factor=0.9, patience=5, verbose=False) # depends on lr_scheduler. Needs more flexibility
        return [opt], [{'scheduler': scheduler, 'monitor': "val_metric"}]

    def forward(self, params, source_types):
        return self.model(params, source_types)

    def training_step(self, batch):
        spectra_true, params, source_types = batch
        spectra_predict = self(params, source_types)
        loss = self._compute_and_log_losses(spectra_predict, spectra_true, "training")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        spectra_true, params, source_types = batch
        spectra_predict = self(params, source_types)
        if dataloader_idx == 0:
            metric = self._compute_and_log_val_metrics(spectra_predict, spectra_true, "val1")
            self.val1_metric_value = metric.item()
            self.val1_metric_to_plot.append(self.val1_metric_value)
        elif dataloader_idx == 1:
            metric = self._compute_and_log_val_metrics(spectra_predict, spectra_true, "val2")
            self.val2_metric_value = metric.item()
            self.val2_metric_to_plot.append(self.val2_metric_value)
        return metric
        
    def on_validation_epoch_end(self):
        self.val_metric = (self.val1_metric_value + 4 * self.val2_metric_value) / 5
        self.val_metric_to_plot.append(self.val_metric)
        self.log('val_metric', self.val_metric, prog_bar=True)

    def on_train_epoch_end(self):
        train_loss_value = self.trainer.callback_metrics["training_loss"].item()
        self.train_loss_to_plot.append(train_loss_value)
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
                        self.val1_source_types_to_vis[i].to(self.device)
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
                metric_value=self.val1_metric_value,
                dataset_type='val1'
            )

            self.res_visualizator.plot_rates(
                rates_predicted_to_vis=val2_spectra_pdfs_to_vis,
                rates_true_to_vis=self.val2_rates_to_vis,
                current_epoch=self.current_epoch,
                global_step=self.global_step,
                metric_value=self.val2_metric_value,
            )

            self.res_visualizator.plot_rates_with_rel_error(
                rates_predicted_to_vis=val2_spectra_pdfs_to_vis,
                rates_true_to_vis=self.val2_rates_to_vis,
                current_epoch=self.current_epoch,
                global_step=self.global_step,
                metric_value=self.val2_metric_value,
            )

            self.res_visualizator.plot_training_process(
                val1_metric_to_plot=self.val1_metric_to_plot,
                val2_metric_to_plot=self.val2_metric_to_plot,
                val_metric_to_plot=self.val_metric_to_plot,
                train_loss_to_plot=self.train_loss_to_plot,
                train_loss_value=train_loss_value,
                val1_metric_value=self.val1_metric_value,
                val2_metric_value=self.val2_metric_value,
                val_metric_value=self.val_metric,
                global_step=self.global_step,
                current_epoch=self.current_epoch
            )
