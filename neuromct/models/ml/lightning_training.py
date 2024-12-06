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
        self.kwargs = kwargs
        
        self.train_loss_to_plot = []
        self.val1_loss_to_plot = []
        self.val2_loss_to_plot = []
        self.val_loss_to_plot = []
    
    def _compute_and_log_losses(self, spectra_predict, spectra, data_type):
        loss = self.loss_function(spectra_predict, spectra)
        self.log(f"{data_type}_loss", loss, prog_bar=True)
        return loss

    def _compute_and_log_val_losses(self, spectra_predict, spectra, data_type):
        loss = self.val_metric_function(spectra_predict, spectra)
        self.log(f"{data_type}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch):
        spectra, params, source_types = batch
        spectra_predict = self.model(params, source_types)
        loss = self._compute_and_log_losses(spectra_predict, spectra, "training")
        self.train_loss_to_plot.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        spectra, params, source_types = batch
        spectra_predict = self.model(params, source_types)
        if dataloader_idx == 0:
            loss = self._compute_and_log_val_losses(spectra_predict, spectra, "val1")
            self.val1_epoch_loss = loss.item()
        elif dataloader_idx == 1:
            loss = self._compute_and_log_val_losses(spectra_predict, spectra, "val2")
            self.val2_epoch_loss = loss.item()
        return loss
        
    def on_validation_epoch_end(self):
        self.val_epoch_loss = (self.val1_epoch_loss + 4 * self.val2_epoch_loss) / 5
        self.val1_loss_to_plot.append(self.val1_epoch_loss)
        self.val2_loss_to_plot.append(self.val2_epoch_loss)
        self.val_loss_to_plot.append(self.val_epoch_loss)
        self.log('val_loss', self.val_epoch_loss, prog_bar=True)
        
    def configure_optimizers(self):      
        opt = self.optimizer(self.parameters(), lr=self.lr, maximize=False)
        scheduler = self.lr_scheduler(opt, mode='min', factor=0.95, patience=5, verbose=False) # depends on lr_scheduler. Needs more flexibility
        return [opt], [{'scheduler': scheduler, 'monitor': "val_loss"}]
