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
            optimizer_hparams: dict,
            lr: float,
            weight_decay: float,
            bins_centers: torch.Tensor,
            monitor_metric: str,
        ):
        super(TEDELightningTraining, self).__init__()

        self.model = model
        self.loss_function = loss_function
        self.val_metric_functions = val_metric_functions
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.optimizer_hparams = optimizer_hparams
        self.lr = lr
        self.weight_decay = weight_decay
        self.bins_centers = bins_centers
        self.monitor_metric = monitor_metric
        self.val_metric_names = list(val_metric_functions.keys())

        self.val1_metrics_to_plot = {key: [] for key in self.val_metric_names}
        self.val2_metrics_to_plot = {key: [] for key in self.val_metric_names}
        self.val_metrics_to_plot = {key: [] for key in self.val_metric_names}
        self.train_loss_to_plot = []

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
        if self.optimizer == optim.AdamW:
            opt = self.optimizer(
                self.parameters(), 
                lr=self.lr, 
                betas=(
                    self.optimizer_hparams['beta1'], 
                    self.optimizer_hparams['beta2']
                ), 
                weight_decay=self.weight_decay
            )    
        elif self.optimizer == optim.RMSprop:
            opt = self.optimizer(
                self.parameters(), 
                lr=self.lr, 
                alpha=self.optimizer_hparams['alpha'], 
                weight_decay=self.weight_decay
            )
        
        if self.lr_scheduler == None:
            return [opt]
        else:
            if self.lr_scheduler == optim.lr_scheduler.ExponentialLR:
                scheduler = self.lr_scheduler(
                    opt, gamma=self.optimizer_hparams['gamma'], verbose=False)
            elif self.lr_scheduler == optim.lr_scheduler.ReduceLROnPlateau:
                scheduler = self.lr_scheduler(
                    opt, mode='min', factor=self.optimizer_hparams['reduction_factor'], 
                    patience=20, verbose=False)
            elif self.lr_scheduler == optim.lr_scheduler.CosineAnnealingLR: 
                scheduler = self.lr_scheduler(
                    opt, T_max=self.optimizer_hparams['T_max'], 
                    eta_min=1e-6, verbose=False)
            return [opt], [{'scheduler': scheduler, 'monitor': self.monitor_metric}]

    def forward(self, params, source_types):
        return self.model(params, source_types)

    def training_step(self, batch):
        spectra_true, params, source_types = batch
        spectra_predict = self(params, source_types)
        loss = self._compute_and_log_losses(
            spectra_predict, spectra_true, "training")
        self.train_loss_to_plot.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        spectra_true, params, source_types = batch
        spectra_predict = self(params, source_types)
        if dataloader_idx == 0:
            self.val1_metrics_values = self._compute_and_log_val_metrics(
                spectra_predict, spectra_true, "val1")
            for name, value in self.val1_metrics_values.items():
                self.val1_metrics_to_plot[name].append(value)
        elif dataloader_idx == 1:
            self.val2_metrics_values = self._compute_and_log_val_metrics(
                spectra_predict, spectra_true, "val2")
            for name, value in self.val2_metrics_values.items():
                self.val2_metrics_to_plot[name].append(value)

    def on_validation_epoch_end(self):
        self.val_metrics_values = dict()
        for name in self.val_metric_names:
            self.val_metrics_values[name] = (
                self.val1_metrics_values[name] + self.val2_metrics_values[name]) / 2
            self.log(f"val_{name}_metric", self.val_metrics_values[name], prog_bar=True)
            self.val_metrics_to_plot[name].append(self.val_metrics_values[name])
