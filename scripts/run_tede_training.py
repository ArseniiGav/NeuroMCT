import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from neuromct.models.ml import TEDE, TEDELightningTraining
from neuromct.models.ml.losses import CosineDistanceLoss
from neuromct.configs import data_configs
from neuromct.utils import tede_argparse, create_dataset, define_transformations
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning import Trainer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = tede_argparse()

training_data_transformations = define_transformations("training")
val_data_transformations = define_transformations("val")

train_data = create_dataset("training", training_data_transformations)
val1_data = create_dataset("val1", val_data_transformations)
val2_1_data = create_dataset("val2_1", val_data_transformations)
val2_2_data = create_dataset("val2_2", val_data_transformations)
val2_3_data = create_dataset("val2_3", val_data_transformations)
val2_data = ConcatDataset([val2_1_data, val2_2_data, val2_3_data])

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)
val1_loader = DataLoader(val1_data, batch_size=val1_data.__len__(), shuffle=False, pin_memory=True)
val2_loader = DataLoader(val2_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

loss_function = CosineDistanceLoss()
val_metric_function = CosineDistanceLoss()

optimizer = optim.Adam
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau

monitor_metric_es = "val_loss"
monitor_metric_checkpoint = "val_loss"
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=monitor_metric_checkpoint, mode="min")
early_stopping_callback = EarlyStopping(monitor=monitor_metric_es, mode="min", patience=50)

tede_model = TEDE(
    param_dim=data_configs['params_dim'],
    n_sources=data_configs['n_sources'],
    output_dim=args.output_dim,
    d_model=args.d_model,
    nhead=args.nhead,
    num_encoder_layers=args.num_encoder_layers,
    dim_feedforward=args.dim_feedforward,
    dropout=args.dropout,
).double().to(device)

tede_model_lightning_training = TEDELightningTraining(
    model=tede_model,
    loss_function=loss_function,
    val_metric_function=val_metric_function,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    lr=args.lr
)

trainer_tede = Trainer(
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

trainer_tede.fit(
    tede_model_lightning_training,
    train_dataloaders=train_loader,
    val_dataloaders=[
      val1_loader,
      val2_loader,
    ]
)

best_tede_model = TEDELightningTraining.load_from_checkpoint(
    checkpoint_path=checkpoint_callback.best_model_path,
    model=tede_model,
    loss_function=loss_function,
    val_metric_function=val_metric_function,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    lr=args.lr,
)

# Save the tede model
torch.save(best_tede_model.model.state_dict(), f"{data_configs['path_to_models']}/tede_model.pth")
