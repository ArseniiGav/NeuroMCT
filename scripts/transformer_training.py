import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from neuromct.models.ml import TEDE
from neuromct.dataset import PoissonResampling, NormalizeToUnity, JMLDataset
from neuromct.configs import configs
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning import Trainer


parser = argparse.ArgumentParser(description='Train the TEDE model')
parser.add_argument("--N_resamplings", type=int, default=1000, help='Number of Possion resamples for data augmentation')
parser.add_argument("--batch_size", type=int, default=32, help='Batch size')
parser.add_argument("--d_model", type=int, default=128, help='D model')
parser.add_argument("--nhead", type=int, default=8, help='nhead')
parser.add_argument("--num_encoder_layers", type=int, default=5, help='Number of encoder layers')
parser.add_argument("--dim_feedforward", type=int, default=32, help='Dim feedforward')
parser.add_argument("--dropout", type=float, default=0.1, help='Dropout in the encoder layers')
args = parser.parse_args()

transform = transforms.Compose([
    PoissonResampling(N_resamplings=args.N_resamplings),  # Apply Poisson resampling
    NormalizeToUnity()                                    # Normalize to unity
])

train_data = JMLDataset(dataset_type="training", path_to_processed_data=configs['path_to_processed_data'], transform=transform)
val1_data = JMLDataset(dataset_type="val1", path_to_processed_data=configs['path_to_processed_data'])
val2_1_data = JMLDataset(dataset_type="val2_1", path_to_processed_data=configs['path_to_processed_data'])
val2_2_data = JMLDataset(dataset_type="val2_2", path_to_processed_data=configs['path_to_processed_data'])
val2_3_data = JMLDataset(dataset_type="val2_3", path_to_processed_data=configs['path_to_processed_data'])

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=20)
val1_loader = DataLoader(val1_data, batch_size=val1_data.shape[0], shuffle=False)
val2_1_loader = DataLoader(val2_1_data, batch_size=val2_1_data.shape[0], shuffle=False)
val2_2_loader = DataLoader(val2_2_data, batch_size=val2_2_data.shape[0], shuffle=False)
val2_3_loader = DataLoader(val2_3_data, batch_size=val2_3_data.shape[0], shuffle=False)

regressor = TEDE(
     output_dim=configs['output_dim'],
     param_dim=configs['params_dim'],
     n_sources=configs['n_sources'],
     d_model=args.d_model,
     nhead=args.nhead,
     num_encoder_layers=args.num_encoder_layers,
     dim_feedforward=args.dim_feedforward,
     dropout=args.dropout,
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
best_model = TEDE.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    output_dim=configs['output_dim'],
    param_dim=configs['params_dim'],
    n_sources=configs['n_sources'],
    d_model=args.d_model,
    nhead=args.nhead,
    num_encoder_layers=args.num_encoder_layers,
    dim_feedforward=args.dim_feedforward,
    dropout=args.dropout,
)

# Save the regressor model
torch.save(best_model.state_dict(), f"{configs['path_to_saved_models']}/Regressor.pth")