import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule


class LightningTrainingTransformer(LightningModule):
    def __init__(self,
             transformer: nn.Module,
             loss_function: nn.Module,
             metric_function: nn.Module,
             optimizer: optim.Optimizer,
             lr_scheduler: optim.lr_scheduler.LRScheduler,
             lr: float,
             **kwargs,
        ):
        super(LightningTrainingTransformer, self).__init__()
        self.transformer = transformer
        self.loss_function = loss_function
        self.metric_function = metric_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr = lr    
        self.kwargs = kwargs
        
        self.train_loss_to_plot = []
        self.val1_loss_to_plot = []
        self.val2_loss_to_plot = []
        self.val_loss_to_plot = []
    
    def _compute_and_log_losses(self, y_pred, y, data_type):
        loss = self.loss_function(y_pred, y)
        self.log(f"{data_type}_loss", loss, prog_bar=True)
        return loss

    def _compute_and_log_val_losses(self, y_pred, y, data_type):
        loss = self.metric_function(y_pred, y)
        self.log(f"{data_type}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self._compute_and_log_losses(y_pred, y, "training")
        self.train_loss_to_plot.append(loss.item())
        return loss

    def validation_step(self, batch, dataloader_idx=0):
        x, y = batch
        y_pred = self(x)
        if dataloader_idx == 0:
            loss = self._compute_and_log_val_losses(y_pred, y, "val1")
            self.val1_epoch_loss = loss.item()
        elif dataloader_idx == 1:
            loss = self._compute_and_log_val_losses(y_pred, y, "val2")
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
        scheduler = lr_scheduler(opt, mode='min', factor=0.95, patience=5, verbose=False) # depends on lr_scheduler. Needs more flexibility
        return [opt], [{'scheduler': scheduler, 'monitor': "val_loss"}]
