import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule

from .metrics import LpNormDistance


class NFDELightningTraining(LightningModule):
    def __init__(self,
            model: nn.Module,
            loss_function: str,
            val_metric_functions: dict[str, nn.Module],
            optimizer: optim.Optimizer,
            lr_scheduler: optim.lr_scheduler.LRScheduler,
            optimizer_hparams: dict,
            lr: float,
            weight_decay: float,
            monitor_metric: str,
            n_energies_to_gen: int
        ):
        super(NFDELightningTraining, self).__init__()

        self.model = model
        self.loss_function = loss_function
        self.val_metric_functions = val_metric_functions
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.optimizer_hparams = optimizer_hparams
        self.lr = lr
        self.weight_decay = weight_decay
        self.monitor_metric = monitor_metric
        self.val_metric_names = list(val_metric_functions.keys())
        self.n_energies_to_gen = n_energies_to_gen

        self.val1_metrics_to_plot = {key: [] for key in self.val_metric_names}
        self.val2_metrics_to_plot = {key: [] for key in self.val_metric_names}
        self.val_metrics_to_plot = {key: [] for key in self.val_metric_names}
        self.train_loss_to_plot = []

        if self.loss_function == 'wasserstein':
            self.wasserstein_1d = LpNormDistance(p=1)
        elif self.loss_function == "cramer":
            self.cramer_1d = LpNormDistance(p=2)

    def _compute_and_log_val_metrics(self, spectra_predict, spectra_true, data_type):
        metrics = dict()
        for name, function in self.val_metric_functions.items():
            metric = function(spectra_predict, spectra_true)
            self.log(f"{data_type}_{name}_metric", metric, prog_bar=True, on_epoch=True)
            metrics[name] = metric.item()
        return metrics

    def configure_optimizers(self):
        if self.optimizer == optim.AdamW:
            opt = self.optimizer(
                self.model.parameters(), 
                lr=self.lr, 
                betas=(
                    self.optimizer_hparams['beta1'], 
                    self.optimizer_hparams['beta2']
                ), 
                weight_decay=self.weight_decay
            )    
        elif self.optimizer == optim.RMSprop:
            opt = self.optimizer(
                self.model.parameters(), 
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
        real_energies, params, source_types = batch
        if self.loss_function == 'kl-div':
            base_log_prob, log_det_jacobian = self.model._log_prob_comp(
                real_energies, params, source_types)
            base_log_prob_loss = -base_log_prob.mean()
            log_det_jacobian_loss = -log_det_jacobian.mean()
            loss = base_log_prob_loss + log_det_jacobian_loss
        else:
            if self.model.base_type == 'uniform':
                z = torch.rand(params.shape[0], 1).to(self.device) * 20
            elif self.model.base_type == 'normal':
                z = torch.randn(params.shape[0], 1).to(self.device)
            elif self.model.base_type == 'lognormal':
                z = torch.exp(torch.randn(params.shape[0], 1).to(self.device))

            if self.loss_function == 'wasserstein':
                x = self.inverse(z, params, source_types)
                loss = self.wasserstein_1d(real_energies, x)
            elif self.loss_function == 'cramer':
                x = self.inverse(z, params, source_types)
                loss = self.cramer_1d(real_energies, x)

        self.log(f"training_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_loss_to_plot.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        real_energies, params, source_types = batch

        gen_energies = []
        for i in range(batch.shape[0]):
            gen_energies_per_condition = self.model.generate_energies(
                self.n_energies_to_gen, params[i, :], source_types[i, :])
            gen_energies.append(gen_energies_per_condition)
        gen_energies = torch.cat(gen_energies, dim=0, dtype=torch.float32)

        if dataloader_idx == 0:
            self.val1_metrics_values = self._compute_and_log_val_metrics(
                gen_energies, real_energies, "val1")
            for name, value in self.val1_metrics_values.items():
                self.val1_metrics_to_plot[name].append(value)
        elif dataloader_idx == 1:
            self.val2_metrics_values = self._compute_and_log_val_metrics(
                gen_energies, real_energies, "val2")
            for name, value in self.val2_metrics_values.items():
                self.val2_metrics_to_plot[name].append(value)

    def on_validation_epoch_end(self):
        self.val_metrics_values = dict()
        for name in self.val_metric_names:
            self.val_metrics_values[name] = (
                self.val1_metrics_values[name] + self.val2_metrics_values[name]) / 2
            self.log(f"val_{name}_metric", self.val_metrics_values[name], prog_bar=True)
            self.val_metrics_to_plot[name].append(self.val_metrics_values[name])
