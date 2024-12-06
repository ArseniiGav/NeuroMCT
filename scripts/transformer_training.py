import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from neuromct.models.ml import TransformerRegressor
from neuromct.dataset import PoissonResampling, NormalizeToUnity, JMLDataset
from neuromct.configs import configs
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning import Trainer


path_to_processed_data = configs['path_to_processed_data']
path_to_saved_models = configs['path_to_saved_models']

transform = transforms.Compose([
    PoissonResampling(N_resamplings=1000),  # Apply Poisson resampling
    NormalizeToUnity()                      # Normalize to unity
])

train_data = JMLDataset(dataset_type="training", path_to_processed_data=path_to_processed_data, transform=transform)
val1_data = JMLDataset(dataset_type="val1", path_to_processed_data=path_to_processed_data)
val2_1_data = JMLDataset(dataset_type="val2_1", path_to_processed_data=path_to_processed_data)
val2_2_data = JMLDataset(dataset_type="val2_2", path_to_processed_data=path_to_processed_data)
val2_3_data = JMLDataset(dataset_type="val2_3", path_to_processed_data=path_to_processed_data)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=20)
val1_loader = DataLoader(val1_data, batch_size=val1_data.shape[0], shuffle=False)
val2_1_loader = DataLoader(val2_1_data, batch_size=val2_1_data.shape[0], shuffle=False)
val2_2_loader = DataLoader(val2_2_data, batch_size=val2_2_data.shape[0], shuffle=False)
val2_3_loader = DataLoader(val2_3_data, batch_size=val2_3_data.shape[0], shuffle=False)

regressor = TransformerRegressor(
     output_dim=output_dim,
     param_dim=param_dim,
     n_sources=n_sources,
     d_model=d_model,
     nhead=nhead,
     num_encoder_layers=num_encoder_layers,
     dim_feedforward=dim_feedforward,
     dropout=dropout,
)

trainer_regressor = Trainer(
    max_epochs=1000,
    deterministic=True,
    accelerator="gpu",
    devices=[0],
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        LearningRateMonitor(),
    ],
    # logger=neptune_logger,
    enable_checkpointing=True,
)

trainer_regressor.fit(
    regressor,
    train_dataloaders=train_loader,
    val_dataloaders=[
      val1_loader,
      val2_1_loader,
      val2_2_loader,
      val2_3_loader,
    ]
)

print(checkpoint_callback.best_model_path)
print(checkpoint_callback.best_model_score)
best_regressor_model = TransformerRegressor.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    output_dim=output_dim,
    param_dim=param_dim,
    n_sources=n_sources,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
)

# Save the regressor model
torch.save(best_regressor_model.state_dict(), f"{path_to_saved_models}/Regressor.pth")