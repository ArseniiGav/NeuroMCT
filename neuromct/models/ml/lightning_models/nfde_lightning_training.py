import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule

from ..metrics import LpNormDistance


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
            n_en_values: int,
            en_limits: tuple[float, float]
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
        
        lb, rb = en_limits
        self.x_values = torch.linspace(
            lb, rb, n_en_values, dtype=torch.float32)

        self.val1_metrics_to_plot = {key: [] for key in self.val_metric_names}
        self.val2_metrics_to_plot = {key: [] for key in self.val_metric_names}
        self.val_metrics_to_plot = {key: [] for key in self.val_metric_names}
        self.train_loss_to_plot = []

        if self.loss_function == 'wasserstein':
            self.wasserstein_loss = LpNormDistance(p=1)
        elif self.loss_function == "cramer":
            self.cramer_loss = LpNormDistance(p=2)
    
    def _compute_and_log_val_metrics(self, x, prob_x_batch, real_energies_batch, data_type):
        metrics = dict()
        for name, function in self.val_metric_functions.items():
            metric_list = []
            for i in range(real_energies_batch.shape[0]):
                no_nan_inds = ~torch.isnan(real_energies_batch[i])
                real_energies_no_nan = real_energies_batch[i][no_nan_inds]
                metric = function(
                    x.unsqueeze(0), 
                    real_energies_no_nan.unsqueeze(0), 
                    prob_x_batch[i].unsqueeze(0), 
                    None
                )
                metric_list.append(metric)
            metric_mean = torch.mean(torch.stack(metric_list))
            metrics[name] = metric_mean.item()
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
        batch_size = real_energies.shape[0]
        x = self.x_values.to(device=real_energies.device)

        losses = []
        for i in range(batch_size):
            no_nan_inds = ~torch.isnan(real_energies[i])
            real_energies_no_nan = real_energies[i][no_nan_inds]
            if self.loss_function == 'kl-div':
                base_log_prob, log_det_jacobian = self.model._log_prob_comp(
                    real_energies_no_nan.unsqueeze(1), 
                    params[i].unsqueeze(0), 
                    source_types[i].unsqueeze(0)
                )
                base_log_prob_loss = -base_log_prob.mean()
                log_det_jacobian_loss = -log_det_jacobian.mean()
                loss = base_log_prob_loss + log_det_jacobian_loss
            else:
                prob_x = torch.exp(
                    self.model.log_prob(
                        x.unsqueeze(1), 
                        params[i].unsqueeze(0), 
                        source_types[i].unsqueeze(0)
                    )
                )
                if self.loss_function == 'wasserstein':
                    loss = self.wasserstein_loss(
                        x.unsqueeze(0), 
                        real_energies_no_nan.unsqueeze(0), 
                        prob_x.unsqueeze(0), 
                        None
                    )
                elif self.loss_function == 'cramer':
                    loss = self.cramer_loss(
                        x.unsqueeze(0), 
                        real_energies_no_nan.unsqueeze(0), 
                        prob_x.unsqueeze(0), 
                        None
                    )
            losses.append(loss)
        mean_loss = torch.mean(torch.tensor(losses))
        self.log(f"training_loss", mean_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_loss_to_plot.append(mean_loss.item())
        return loss

    def on_validation_epoch_start(self):
        self.val1_metrics_within_val_epoch = {
            key: [] for key in self.val_metric_names}
        self.val2_metrics_within_val_epoch = {
            key: [] for key in self.val_metric_names}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataset_type = "val1" if dataloader_idx == 0 else "val2"
        real_energies, params, source_types = batch
        batch_size = real_energies.shape[0]
        x = self.x_values.to(device=real_energies.device)

        prob_x_batch = []
        for i in range(batch_size):
            prob_x = torch.exp(
                self.model.log_prob(
                    x.unsqueeze(1), 
                    params[i].unsqueeze(0), 
                    source_types[i].unsqueeze(0)
                )
            )
            prob_x_batch.append(prob_x)
        prob_x_batch = torch.vstack(prob_x_batch)

        metrics_values = self._compute_and_log_val_metrics(
            x, prob_x_batch, real_energies, dataset_type)
        if dataset_type == 'val1':
            for name, value in metrics_values.items():
                self.val1_metrics_within_val_epoch[name].append(value)
        elif dataset_type == 'val2':
            for name, value in metrics_values.items():
                self.val2_metrics_within_val_epoch[name].append(value)

    def on_validation_epoch_end(self):
        self.val1_metrics_values = dict()
        self.val2_metrics_values = dict()
        self.val_metrics_values = dict()

        for name in self.val_metric_names:
            val1_metrics_value = torch.mean(
                torch.tensor(self.val1_metrics_within_val_epoch[name])
            ).item()
            val2_metrics_value = torch.mean(
                torch.tensor(self.val2_metrics_within_val_epoch[name])
            ).item()
            val_metrics_value = (val1_metrics_value + val2_metrics_value) / 2

            self.val1_metrics_values[name] = val1_metrics_value 
            self.val2_metrics_values[name] = val2_metrics_value
            self.val_metrics_values[name] = val_metrics_value

            self.val1_metrics_to_plot[name].append(val1_metrics_value)
            self.val2_metrics_to_plot[name].append(val2_metrics_value)
            self.val_metrics_to_plot[name].append(val_metrics_value)

            self.log(f"val1_{name}_metric", val1_metrics_value, prog_bar=True)
            self.log(f"val2_{name}_metric", val2_metrics_value, prog_bar=True)
            self.log(f"val_{name}_metric", val_metrics_value, prog_bar=True)

        print(self.val_metrics_values)
