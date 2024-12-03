import torch.optim as optim
from lightning import LightningModule
from neuromct.models.ml import TransformerRegressor


class LightningTrainingTransformer(LightningModule):
    def __init__(self,
             optimizer: optim.Optimizer,
             lr: float,
             transformer: nn.Module,
             **kwargs,
        ):
        super(LightningTrainingTransformer, self).__init__()
        self.kwargs = kwargs
        self.lr = lr    
        self.train_loss_to_plot = []
        self.val1_loss_to_plot = []
        self.val2_loss_to_plot = []
        self.val_loss_to_plot = []
    
    def load_scaler(self, filepath="models/minmax_scaler.pkl"):
        return pickle.load(open(filepath, "rb"))

    def load_val1_data_to_vis(self, filepath):
        val1_data = torch.Tensor(np.load(filepath)['arr_0'])
        return val1_data
        
    def load_val2_data_to_vis(self, filepath):
        val2_data = torch.Tensor(np.load(filepath)['arr_0'])
        val2_data[:, :self.output_dim] = val2_data[:, :self.output_dim] / val2_data[:, :self.output_dim].sum(1)[:, None]
        val2_data_lambdas = val2_data.reshape(self.n_classes, 1000, self.output_dim+self.num_conditions).mean(1)
        return val2_data_lambdas
    
    def _compute_and_log_losses(self, y_pred, y, data_type):
        loss = self.loss_function(y_pred, y)
        self.log(f"{data_type}_loss", loss, prog_bar=True)
        return loss

    def _compute_and_log_val_losses(self, y_pred, y, data_type):
        loss = self.val_loss_function(y_pred, y)
        self.log(f"{data_type}_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def forward(self, src):
        src = self.input_linear(src).unsqueeze(0) 
        src = self.pos_encoder(src)
        tgt = torch.zeros_like(src)
        output = self.transformer_decoder(tgt, src)
        output = self.output_linear(output.squeeze(0))
        output = self.output_activation(output)
        output = output / output.sum(1)[:, None]
        return output

    def training_step(self, batch):
        x, y = batch[:, self.output_dim:], batch[:, :self.output_dim]
        y_pred = self(x)
        loss = self._compute_and_log_losses(y_pred, y, "training")
        self.train_loss_to_plot.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch[:, self.output_dim:], batch[:, :self.output_dim]
        y_pred = self(x)
        if dataloader_idx == 0:
            loss = self._compute_and_log_val_losses(y_pred, y, "val1")
            self.val1_epoch_loss = loss.item()
        elif dataloader_idx == 1:
            loss = self._compute_and_log_val_losses(y_pred, y, "val2")
            self.val2_epoch_loss = loss.item()
        return loss
        
    def on_validation_epoch_end(self):
        if self.current_epoch % 1 == 0:
            save = True
        else:
            save = False

        self.val_epoch_loss = (self.val1_epoch_loss + 4 * self.val2_epoch_loss) / 5
        self.log('val_loss', self.val_epoch_loss, prog_bar=True)
        
        clear_output(wait=True)
        self.plot_val1_spectra(save);
        self.plot_val2_spectra(save);
        self.plot_val_metrics(save)
        self.plot_train_loss(save)

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2
        
        opt = optim.Adam(self.parameters(), lr=lr, betas=(b1, b2), maximize=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.95, patience=5, verbose=False)
        return [opt], [{'scheduler': scheduler, 'monitor': "val_loss"}]

