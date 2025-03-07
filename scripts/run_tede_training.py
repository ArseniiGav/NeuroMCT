import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from neuromct.models.ml import TEDE
from neuromct.models.ml.callbacks import ModelResultsVisualizerCallback
from neuromct.models.ml.lightning_models import TEDELightningTraining
from neuromct.models.ml.losses import GeneralizedKLDivLoss
from neuromct.models.ml.metrics import LpNormDistance
from neuromct.configs import data_configs
from neuromct.utils import (
    tede_argparse,
    create_dataset,
    define_transformations,
    res_visualizator_setup
)

approach_type = 'tede'
base_path_to_models = data_configs['base_path_to_models']
path_to_processed_data = data_configs['path_to_processed_data']
path_to_tede_training_results = data_configs['path_to_tede_training_results']

os.makedirs(f'{path_to_tede_training_results}', exist_ok=True)
os.makedirs(f'{path_to_tede_training_results}/plots', exist_ok=True)
os.makedirs(f'{path_to_tede_training_results}/predictions', exist_ok=True)

kNPE_bins_edges = data_configs['kNPE_bins_edges']
kNPE_bins_centers = (kNPE_bins_edges[:-1] + kNPE_bins_edges[1:]) / 2
kNPE_bins_centers = torch.tensor(kNPE_bins_centers, dtype=torch.float32)
bin_size = data_configs['bin_size']

model_res_visualizator = res_visualizator_setup(data_configs)

args = tede_argparse()
seed_everything(args.seed, workers=True)

# Poisson noise + pdf constuction
training_data_transformations = define_transformations("training", bin_size)

# pdf constuction only
val_data_transformations = define_transformations("val", bin_size) 

train_data = create_dataset(
    "training", 
    path_to_processed_data, 
    approach_type,
    training_data_transformations
)
val1_data = create_dataset(
    "val1", 
    path_to_processed_data, 
    approach_type,
    val_data_transformations
)
val2_data = []
for i in range(3):
    val2_i_data = create_dataset(
        f"val2_{i+1}", 
        path_to_processed_data, 
        approach_type,
        val_data_transformations, 
        val2_rates=True
    )
    val2_data.append(val2_i_data)
val2_data = ConcatDataset(val2_data)

train_loader = DataLoader(
    train_data, 
    batch_size=args.batch_size,
    shuffle=True, 
    num_workers=20, 
    pin_memory=True
)

val1_loader = DataLoader(
    val1_data, 
    batch_size=val1_data.__len__(), 
    shuffle=False, 
    pin_memory=True
)

val2_loader = DataLoader(
    val2_data, 
    batch_size=val2_data.__len__(), 
    shuffle=False, 
    pin_memory=True
)

kl_div = GeneralizedKLDivLoss(
    log_input=False, log_target=False, reduction='batchmean')
wasserstein_distance = LpNormDistance(p=1) # Wasserstein distance
cramer_distance = LpNormDistance(p=2) # Cramér-von Mises distance
ks_distance = LpNormDistance(p=torch.inf) # Kolmogorov-Smirnov distance
val_metric_functions = {
    "wasserstein": wasserstein_distance, 
    "cramer": cramer_distance,
    "ks": ks_distance
}

optimizer_hparams = {}
if args.optimizer == 'RMSprop':
    optimizer = optim.RMSprop
    optimizer_hparams['alpha'] = args.alpha
elif args.optimizer == 'AdamW':
    optimizer = optim.AdamW
    optimizer_hparams['beta1'] = args.beta1
    optimizer_hparams['beta2'] = args.beta2

if args.lr_scheduler == 'ExponentialLR':
    lr_scheduler = optim.lr_scheduler.ExponentialLR
    optimizer_hparams['gamma'] = args.gamma
elif args.lr_scheduler == 'CosineAnnealingLR':
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR
    optimizer_hparams['T_max'] = args.T_max
elif args.lr_scheduler == 'ReduceLROnPlateau':
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau
    optimizer_hparams['reduction_factor'] = args.reduction_factor
else:
    lr_scheduler = None

monitor_metric = "val_cramer_metric"
checkpoint_callback = ModelCheckpoint(
    save_top_k=1, monitor=monitor_metric, mode="min")
early_stopping_callback = EarlyStopping(
    monitor=monitor_metric, mode="min", patience=200)
res_visualizer_callback = ModelResultsVisualizerCallback(
    res_visualizer=model_res_visualizator,
    base_path_to_savings=path_to_tede_training_results,
    plots_dir_name='plots',
    predictions_dir_name='predictions'
)

logger = CSVLogger(
    save_dir=path_to_tede_training_results, 
    name=f"training_logs"
)

tede_model = TEDE(
    n_sources=args.n_sources,
    output_dim=args.output_dim,
    d_model=args.d_model,
    activation=args.activation_function,
    nhead=args.nhead,
    num_encoder_layers=args.num_encoder_layers,
    dim_feedforward=args.dim_feedforward,
    dropout=args.dropout,
    temperature=args.temperature,
    bin_size=bin_size
)

tede_model_lightning_training = TEDELightningTraining(
    model=tede_model,
    loss_function=kl_div,
    val_metric_functions=val_metric_functions,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    optimizer_hparams=optimizer_hparams,
    lr=args.lr,
    weight_decay=args.weight_decay,
    bins_centers=kNPE_bins_centers,
    monitor_metric=args.monitor_metric
)

trainer_tede = Trainer(
    max_epochs=2000,
    accelerator=args.accelerator,
    devices="auto",
    precision="16-mixed",
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        res_visualizer_callback,
        LearningRateMonitor(),
    ],
    logger=logger,
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
    loss_function=kl_div,
    val_metric_functions=val_metric_functions,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    optimizer_hparams=optimizer_hparams,
    lr=args.lr,
    weight_decay=args.weight_decay,
    bins_centers=kNPE_bins_centers,
    monitor_metric=args.monitor_metric
)

torch.save(
    best_tede_model.model.state_dict(), 
    f"{base_path_to_models}/models/tede_model.pth"
)
