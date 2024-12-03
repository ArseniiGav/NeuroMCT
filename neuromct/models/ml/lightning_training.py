from lightning import LightningModule
from neuromct.models.ml import TransformerRegressor

class LightningTrainingTransformer(LightningModule):
    def __init__(self,
             output_dim,
             param_dim,
             n_sources,
             lr,
             b1,
             b2,
             d_model,
             nhead,
             num_encoder_layers,
             dim_feedforward,
             dropout
        ):
        super(LightningTrainingTransformer, self).__init__()

        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        
        self.output_dim = output_dim
        self.param_dim = param_dim
        self.n_sources = n_sources
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
        self.transformer = TransformerRegressor(
             output_dim=self.output_dim,
             param_dim=self.param_dim,
             n_sources=self.n_sources,
             d_model=self.d_model,
             nhead=self.nhead,
             num_encoder_layers=self.num_encoder_layers,
             dim_feedforward=self.dim_feedforward,
             dropout=self.dropout,
        )

        self.loss_function = CosineDistanceLoss() #nn.MSELoss() #GeneralizedPoissonNLLLoss(log=True)
        self.val_loss_function = CosineDistanceLoss() #nn.PoissonNLLLoss(log_input=False)
        self.train_loss_to_plot = []
        self.val1_loss_to_plot = []
        self.val2_loss_to_plot = []
        self.val_loss_to_plot = []
        self.data_path = data_path
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
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

    def plot_val1_spectra(self, save=False):
        scaler = self.load_scaler()

        ### validation dataset 1 ###
        
        val1_data = self.load_val1_data_to_vis(f"{self.data_path}/validation_data_three_params_npe.npz")
        val1_data[:, :self.output_dim] = val1_data[:, :self.output_dim] / val1_data[:, :self.output_dim].sum(1)[:, None]
        val1_spectra = val1_data[:, :self.output_dim]
        val1_conditions = val1_data[:, self.output_dim:]

        val_cond_values = [0.1750, 0.5750, 0.8750]
        kB_vary_condition = torch.logical_or((val1_conditions[:, 0] == val_cond_values[0]), (val1_conditions[:, 0] == val_cond_values[1]))
        kB_vary_condition = torch.logical_or(kB_vary_condition, (val1_conditions[:, 0] == val_cond_values[2]))
        kB_vary_condition = torch.logical_and(kB_vary_condition, (val1_conditions[:, 1] == val_cond_values[1]))
        kB_vary_condition = torch.logical_and(kB_vary_condition, (val1_conditions[:, 2] == val_cond_values[1]))
        kB_vary_indexes = torch.where(kB_vary_condition)[0]

        fC_vary_condition = torch.logical_or((val1_conditions[:, 1] == val_cond_values[0]), (val1_conditions[:, 1] == val_cond_values[1]))
        fC_vary_condition = torch.logical_or(fC_vary_condition, (val1_conditions[:, 1] == val_cond_values[2]))
        fC_vary_condition = torch.logical_and(fC_vary_condition, (val1_conditions[:, 0] == val_cond_values[1]))
        fC_vary_condition = torch.logical_and(fC_vary_condition, (val1_conditions[:, 2] == val_cond_values[1]))
        fC_vary_indexes = torch.where(fC_vary_condition)[0]

        LY_vary_condition = torch.logical_or((val1_conditions[:, 2] == val_cond_values[0]), (val1_conditions[:, 2] == val_cond_values[1]))
        LY_vary_condition = torch.logical_or(LY_vary_condition, (val1_conditions[:, 2] == val_cond_values[2]))
        LY_vary_condition = torch.logical_and(LY_vary_condition, (val1_conditions[:, 0] == val_cond_values[1]))
        LY_vary_condition = torch.logical_and(LY_vary_condition, (val1_conditions[:, 1] == val_cond_values[1]))
        LY_vary_indexes = torch.where(LY_vary_condition)[0]

        val1_conditions_transformed = scaler.inverse_transform(val1_conditions[:, :self.n_params].cpu().numpy())
        val1_conditions_transformed = np.concatenate((val1_conditions_transformed, val1_conditions[:, self.n_params:]), axis=1)
        
        indexes_to_plot = [kB_vary_indexes, fC_vary_indexes, LY_vary_indexes]
        fig, ax = plt.subplots(3, 3, figsize=(16, 10))
        ax = ax.flatten()
        for m in range(self.n_params):
            for i in range(kB_vary_indexes.shape[0]):
                j = i % 3 + m * 3
                ith_condition = val1_conditions[indexes_to_plot[m][i], :].to(self.device)

                self.eval()
                with torch.no_grad():
                    pred_spectra = self(ith_condition.unsqueeze(0)).squeeze(0)
                self.train()
                pred_spectra = pred_spectra.detach().cpu().numpy()

                ########### plot truth ###########
                if i // 3 == 0:
                    label = "G4, all sources"
                else:
                    label = None
                ax[j].stairs(val1_spectra[indexes_to_plot[m][i], :], self.bins / 1000, label=label, color='black', alpha=0.7)

                ########### plot predicted ###########
                if i // 3 == 0:
                    label = "Regressor, Cs137"
                    color = 'darkgreen'
                elif i // 3 == 1:
                    label = "Regressor, K40"
                    color = 'royalblue'
                elif i // 3 == 2:
                    label = "Regressor, Co60"
                    color = 'darkred'
                elif i // 3 == 3:
                    label = "Regressor, AmBe"
                    color = 'purple'
                elif i // 3 == 4:
                    label = "Regressor, AmC"
                    color = 'orange'
                ax[j].stairs(pred_spectra, self.bins / 1000, label=label, color=color, alpha=0.7)

                title  = f"kB: {val1_conditions_transformed[indexes_to_plot[m][i]][0]:.2f} [g/cm2/GeV], "
                title += f"fC: {val1_conditions_transformed[indexes_to_plot[m][i]][1]:.3f}, "
                title += f"LY: {val1_conditions_transformed[indexes_to_plot[m][i]][2]:.0f} [1/MeV]"
                
                ax[j].legend(loc="upper right", ncol=2)
                ax[j].set_title(title)
                ax[j].set_ylim(1e-4, 1)
                ax[j].set_yscale("log")
                ax[j].set_xlim(-0.5, 17)
                
                if j >= 6:
                    ax[j].set_xlabel("kNPE")
                if j % self.n_params == 0:
                    ax[j].set_ylabel("Normalized counts")
                    
        fig.suptitle(f"Validation dataset №1. Epoch: {self.current_epoch}, " + r"$\rm D^{C}_{V_1}$" + f" = {self.val1_epoch_loss:.5f}", x=0.25, y=0.99, fontsize=20)
        fig.tight_layout()
        if save:
            fig.savefig(f'plots/Transformer_training_process/epoch_{self.current_epoch}_v1.png') 
        plt.show()

    def plot_val2_spectra(self, save=False):
        scaler = self.load_scaler()
        
        ### validation dataset 2 ###
        
        val2_spectra_list = []
        val2_conditions_list = []
        val2_conditions_transformed_list = []
        for i in range(3):
            val2_data = self.load_val2_data_to_vis(f"{self.data_path}/validation_data2_{i+1}_three_params_npe.npz")
            val2_spectra = val2_data[:, :self.output_dim]
            val2_conditions = val2_data[:, self.output_dim:]
            val2_conditions_transformed = scaler.inverse_transform(val2_conditions[:, :self.n_params].cpu().numpy())
            val2_conditions_transformed = np.concatenate((val2_conditions_transformed, val2_conditions[:, self.n_params:]), axis=1)

            val2_spectra_list.append(val2_spectra)
            val2_conditions_list.append(val2_conditions)
            val2_conditions_transformed_list.append(val2_conditions_transformed)
        
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        ax = ax.flatten()
        for i in range(3 * self.n_classes):
            j = i // 5
            k = i % 5
            ith_condition = val2_conditions_list[j][k].to(self.device)
            self.eval()
            with torch.no_grad():
                pred_spectra = self(ith_condition.unsqueeze(0)).squeeze(0)
            self.train()
            pred_spectra = pred_spectra.detach().cpu().numpy()
            
            ########### plot truth ###########
            if i % 5 == 0:
                label = "G4, all sources"
            else:
                label = None
            ax[j].stairs(val2_spectra_list[j][k], self.bins / 1000, label=label, color='black', alpha=0.7)

            ########### plot predicted ###########
            if k == 0:
                label = "Regressor, Cs137"
                color = 'darkgreen'
            elif k == 1:
                label = "Regressor, K40"
                color = 'royalblue'
            elif k == 2:
                label = "Regressor, Co60"
                color = 'darkred'
            elif k == 3:
                label = "Regressor, AmBe"
                color = 'purple'
            elif k == 4:
                label = "Regressor, AmC"
                color = 'orange'
            ax[j].stairs(pred_spectra, self.bins / 1000, label=label, color=color, alpha=0.7)

            title  = f"kB: {val2_conditions_transformed_list[j][k][0]:.2f} [g/cm2/GeV], "
            title += f"fC: {val2_conditions_transformed_list[j][k][1]:.3f}, "
            title += f"LY: {val2_conditions_transformed_list[j][k][2]:.0f} [1/MeV]"
            
            ax[j].legend(loc="upper right", ncol=2)
            ax[j].set_title(title)
            ax[j].set_ylim(1e-4, 1)
            ax[j].set_xlim(-0.5, 17)
            ax[j].set_yscale("log")
            ax[j].set_xlabel("kNPE")
            if j == 0:
                ax[j].set_ylabel("Normalized counts")

        fig.suptitle(f"Validation dataset №2. Epoch: {self.current_epoch}, " + r"$\rm D^{C}_{V_2}$" + f" = {self.val2_epoch_loss:.5f}", x=0.25, y=0.99, fontsize=20)
        fig.tight_layout()
        if save:
            fig.savefig(f'plots/Transformer_training_process/epoch_{self.current_epoch}_v2.png') 
        plt.show()

    def plot_val_metrics(self, save=False):
        self.val1_loss_to_plot.append(self.val1_epoch_loss)
        self.val2_loss_to_plot.append(self.val2_epoch_loss)
        self.val_loss_to_plot.append(self.val_epoch_loss)

        title = r'$\rm D^{C}_{V_1} = $' + f"{self.val1_epoch_loss:.5f}, "
        title += r'$\rm D^{C}_{V_2} = $' + f"{self.val2_epoch_loss:.5f}; "
        title += r'$\rm D^{C}_{V} = \rm \frac{D^{C}_{V_1} + 4 \cdot \rm D^{C}_{V_2}}{5}$' + f' = {self.val_epoch_loss:.5f}'

        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        ax.plot(range(len(self.val1_loss_to_plot)), self.val1_loss_to_plot, label="Validation dataset №1: " + r'$\rm D^{C}_{V_1}$', color='royalblue', linewidth=1.25)
        ax.plot(range(len(self.val2_loss_to_plot)), self.val2_loss_to_plot, label="Validation dataset №2: " + r'$\rm D^{C}_{V_2}$', color='darkred', linewidth=1.25)
        ax.plot(range(len(self.val_loss_to_plot)), self.val_loss_to_plot, label="Validation datasets combined: " + r'$\rm D^{C}_{V}$', color='darkgreen', linewidth=1.25)

        ax.set_ylabel("Cosine distance", fontsize=16)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.1)
        ax.set_xlabel('Epochs', fontsize=16)
        ax.legend(loc="upper right", fontsize=16)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        # ax.set_axes(fontsize=14)

        fig.suptitle(title, x=0.25, y=0.99, fontsize=20)
        fig.tight_layout()
        if save:
            fig.savefig(f'plots/Transformer_training_process/epoch_{self.current_epoch}_val_metrics.png') 
        plt.show()

    def plot_train_loss(self, save=False):

        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        ax.plot(range(len(self.train_loss_to_plot)), self.train_loss_to_plot, label="Training data: " + r'$\rm D^{C}_{T}$', color='black', linewidth=1.25)

        ax.set_ylabel("Cosine distance", fontsize=16)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.1)
        ax.set_xlabel('Training steps', fontsize=16)
        ax.legend(loc="upper right", fontsize=16)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        # ax.set_axes(fontsize=14)

        fig.suptitle("Training dataset", x=0.1, y=0.99, fontsize=20)
        fig.tight_layout()
        if save:
            fig.savefig(f'plots/Transformer_training_process/epoch_{self.current_epoch}_train_loss.png') 
        plt.show()
