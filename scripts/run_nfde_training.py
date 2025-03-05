import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from neuromct.models.ml import NFDE
from neuromct.models.ml.callbacks import ModelResultsVisualizerCallback
from neuromct.models.ml.lightning_models import NFDELightningTraining
from neuromct.models.ml.metrics import LpNormDistance
from neuromct.configs import data_configs
from neuromct.utils import (
    nfde_argparse,
    create_dataset,
    res_visualizator_setup
)

approach_type = 'nfde'
base_path_to_models = data_configs['base_path_to_models']
path_to_processed_data = data_configs['path_to_processed_data']
path_to_nfde_training_results = data_configs['path_to_nfde_training_results']

os.makedirs(f'{path_to_nfde_training_results}', exist_ok=True)
os.makedirs(f'{path_to_nfde_training_results}/plots', exist_ok=True)
os.makedirs(f'{path_to_nfde_training_results}/predictions', exist_ok=True)

# model_res_visualizator = res_visualizator_setup(data_configs)

args = nfde_argparse()
seed_everything(args.seed, workers=True)

train_data = create_dataset(
    "training", 
    path_to_processed_data, 
    approach_type
)
val1_data = create_dataset(
    "val1", 
    path_to_processed_data, 
    approach_type
)
val2_data = []
for i in range(3):
    val2_i_data = create_dataset(
        f"val2_{i+1}", 
        path_to_processed_data, 
        approach_type, 
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

loss_function = "kl-div"
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
# res_visualizer_callback = ModelResultsVisualizerCallback(
#     res_visualizer=model_res_visualizator,
#     base_path_to_savings=path_to_nfde_training_results,
#     plots_dir_name='plots',
#     predictions_dir_name='predictions'
# )

logger = CSVLogger(
    save_dir=path_to_nfde_training_results, 
    name=f"training_logs"
)

nfde_model = NFDE(
    n_flows=args.n_flows,
    n_conditions=data_configs['n_conditions'],
    n_sources=args.n_sources,
    n_units=args.n_units,
    activation=args.activation_function,
    flow_type=args.flow_type,
    base_type=args.base_type
)

nfde_model_lightning_training = NFDELightningTraining(
    model=nfde_model,
    loss_function=loss_function,
    val_metric_functions=val_metric_functions,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    optimizer_hparams=optimizer_hparams,
    lr=args.lr,
    weight_decay=args.weight_decay,
    monitor_metric=args.monitor_metric,
    n_energies_to_gen=args.n_energies_to_gen
)

trainer_nfde = Trainer(
    max_epochs=2000,
    accelerator=args.accelerator,
    devices="auto",
    precision="16-mixed",
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        # res_visualizer_callback,
        LearningRateMonitor(),
    ],
    logger=logger,
    enable_checkpointing=True,
)

trainer_nfde.fit(
    nfde_model_lightning_training,
    train_dataloaders=train_loader,
    val_dataloaders=[
      val1_loader,
      val2_loader,
    ]
)

best_nfde_model = NFDELightningTraining.load_from_checkpoint(
    checkpoint_path=checkpoint_callback.best_model_path,
    model=nfde_model,
    loss_function=loss_function,
    val_metric_functions=val_metric_functions,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    optimizer_hparams=optimizer_hparams,
    lr=args.lr,
    weight_decay=args.weight_decay,
    monitor_metric=args.monitor_metric,
    n_energies_to_gen=args.n_energies_to_gen
)

torch.save(
    best_nfde_model.model.state_dict(), 
    f"{base_path_to_models}/models/nfde_model.pth"
)
