import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning import Trainer

from neuromct.models.ml import TEDE, TEDELightningTraining
from neuromct.models.ml.losses import GeneralizedKLDivLoss
from neuromct.models.ml.metrics import LpNormDistance
from neuromct.configs import data_configs
from neuromct.utils import tede_argparse
from neuromct.utils import create_dataset
from neuromct.utils import define_transformations
from neuromct.utils import res_visualizator_setup

path_to_models = data_configs['path_to_models']
path_to_processed_data = data_configs['path_to_processed_data']
n_sources = data_configs['n_sources']
model_res_visualizator = res_visualizator_setup(data_configs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = tede_argparse()

training_data_transformations = define_transformations("training") # poisson noise + normalization
val_data_transformations = define_transformations("val") # normalization only

train_data_initial = create_dataset(
    "training", path_to_processed_data, training_data_transformations)
train_data = ConcatDataset([train_data_initial] * args.n_data_aug)

val1_data = create_dataset("val1", path_to_processed_data, val_data_transformations)

val2_data = []
for i in range(3):
    val2_i_data = create_dataset(
        f"val2_{i+1}", path_to_processed_data, val_data_transformations, val2_rates=True)
    val2_data.append(val2_i_data)
val2_data = ConcatDataset(val2_data)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)
val1_loader = DataLoader(val1_data, batch_size=val1_data.__len__(), shuffle=False, pin_memory=True)
val2_loader = DataLoader(val2_data, batch_size=val2_data.__len__(), shuffle=False, pin_memory=True)

loss_function = GeneralizedKLDivLoss(log_input=False, log_target=False, reduction='batchmean')
val_metric_function = LpNormDistance(p=2) # Cramér-von Mises distance

optimizer = optim.AdamW
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau

monitor_metric_es = "val_metric"
monitor_metric_checkpoint = "val_metric"
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=monitor_metric_checkpoint, mode="min")
early_stopping_callback = EarlyStopping(monitor=monitor_metric_es, mode="min", patience=50)

tede_model = TEDE(
    n_sources=n_sources,
    output_dim=args.output_dim,
    d_model=args.d_model,
    nhead=args.nhead,
    num_encoder_layers=args.num_encoder_layers,
    dim_feedforward=args.dim_feedforward,
    dropout=args.dropout,
    temperature=args.temperature
)

tede_model_lightning_training = TEDELightningTraining(
    model=tede_model,
    loss_function=loss_function,
    val_metric_function=val_metric_function,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    lr=args.lr,
    res_visualizator=model_res_visualizator,
)

trainer_tede = Trainer(
    max_epochs=1000,
    deterministic=True,
    accelerator="gpu",
    devices="auto",
    precision=64,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        LearningRateMonitor(),
    ],
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
    res_visualizator=model_res_visualizator,
)

torch.save(best_tede_model.model.state_dict(), f"{path_to_models}/tede_model.pth")
