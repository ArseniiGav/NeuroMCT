import os
import argparse
import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning import Trainer

import optuna
from optuna.integration import PyTorchLightningPruningCallback
optuna.logging.set_verbosity(optuna.logging.WARNING)

from neuromct.models.ml import TEDE, TEDELightningTraining
from neuromct.models.ml.losses import GeneralizedKLDivLoss
from neuromct.models.ml.metrics import LpNormDistance
from neuromct.configs import data_configs
from neuromct.utils import create_dataset
from neuromct.utils import define_transformations
from neuromct.utils import res_visualizator_setup

parser = argparse.ArgumentParser(description='Run the hyperparameter optimization for the TEDE model')
parser.add_argument("--n_trials", type=int, default=100,
                        help='The number of Optuna trials (default=100).')
parser.add_argument("--accelerator", type=str, default="gpu",
                        help='Device used to train the TEDE model ("cpu" or "gpu", default="gpu").')
parser.add_argument("--seed", type=int, default=22,
                        help='Seed to seed everything (default=22).')
args = parser.parse_args()

path_to_models = data_configs['path_to_models']
path_to_processed_data = data_configs['path_to_processed_data']
path_to_optuna_results = data_configs['path_to_optuna_results']

dirs = [
    f'{path_to_optuna_results}/seed_{args.seed}',
    f'{path_to_optuna_results}/seed_{args.seed}/plots', 
    f'{path_to_optuna_results}/seed_{args.seed}/results', 
]
os.makedirs(dirs, exist_ok=True)

for run_index in range(args.n_trials):
    os.makedirs(
        f"{path_to_optuna_results}/seed_{args.seed}/plots/plots_{run_index}", 
        exist_ok=True
    )
    os.makedirs(
        f"{path_to_optuna_results}/seed_{args.seed}/results/results_{run_index}", 
        exist_ok=True
    )

kNPE_bins_edges = data_configs['kNPE_bins_edges']
kNPE_bins_centers = (kNPE_bins_edges[:-1] + kNPE_bins_edges[1:]) / 2
kNPE_bins_centers = torch.tensor(kNPE_bins_centers, dtype=torch.float64)

model_res_visualizator = res_visualizator_setup(data_configs)

seed_everything(args.seed, workers=True)

# Poisson noise + normalization
training_data_transformations = define_transformations("training")

# normalization only
val_data_transformations = define_transformations("val") 

train_data = create_dataset(
    "training", 
    path_to_processed_data, 
    training_data_transformations
)
val1_data = create_dataset(
    "val1", 
    path_to_processed_data, 
    val_data_transformations
)
val2_data = []
for i in range(3):
    val2_i_data = create_dataset(
        f"val2_{i+1}", 
        path_to_processed_data, 
        val_data_transformations, 
        val2_rates=True
    )
    val2_data.append(val2_i_data)
val2_data = ConcatDataset(val2_data)

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

monitor_metric = "val_cramer_metric"
checkpoint_callback = ModelCheckpoint(
    save_top_k=1, monitor=monitor_metric, mode="min")
early_stopping_callback = EarlyStopping(
    monitor=monitor_metric, mode="min", patience=200)

def objective(trial):
    # Hyperparameter search space
    dependent_params = {}

    common_params = {
        'd_model': trial.suggest_categorical(
            'd_model', [50, 100, 200, 400]),
        'nhead': trial.suggest_categorical(
            'nhead', [5, 10, 25, 50]),
        'num_encoder_layers': trial.suggest_int(
            'num_encoder_layers', 1, 5, step=1),
        'dim_feedforward': trial.suggest_int(
            'dim_feedforward', 32, 512, step=32),
        'activation_function': trial.suggest_categorical(
            'activation_function', ['relu', 'gelu']),
        'learning_rate': trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True),
        'optimizer': trial.suggest_categorical(
            'optimizer', ['AdamW', 'RMSprop']),
        'lr_scheduler': trial.suggest_categorical(
            'lr_scheduler', 
            ['ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'None']),
        'dropout': trial.suggest_float(
            'dropout', 0.0, 0.5, step=0.05),
        'entmax_alpha': trial.suggest_float(
            'entmax_alpha', 1.0, 1.5, step=0.05), # if 1.0, nn.Softmax is used
        'batch_size': trial.suggest_categorical(
            'batch_size', [16, 32, 64, 128, 256, 512]),
    }

    if common_params['scheduler'] == 'ExponentialLR':
        dependent_params['gamma'] = trial.suggest_float(
            'gamma', 0.80, 0.99, step=0.01)
    elif common_params['scheduler'] == 'CosineAnnealingLR':
        dependent_params['T_max'] = trial.suggest_int(
            'T_max', 5, 75, step=5)
    elif common_params['scheduler'] == 'ReduceLROnPlateau':
        dependent_params['reduction_factor'] = trial.suggest_float(
            'reduction_factor', 0.80, 0.99, step=0.01)

    common_params['weight_decay'] = trial.suggest_float(
        'weight_decay', 1e-5, 1e-1, log=True)
    if common_params['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop
        dependent_params['alpha'] = trial.suggest_float(
            'alpha', 0.9, 0.999, step=0.001)
    elif common_params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW
        dependent_params['beta1'] = trial.suggest_float(
            'beta1', 0.5, 0.95, step=0.01)
        dependent_params['beta2'] = trial.suggest_float(
            'beta2', 0.9, 0.999, step=0.001)

    if common_params['lr_scheduler'] == 'ExponentialLR':
        lr_scheduler = optim.lr_scheduler.ExponentialLR
    elif common_params['lr_scheduler'] == 'ReduceLROnPlateau': 
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau
    elif common_params['lr_scheduler'] == 'CosineAnnealingLR': 
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR

    train_loader = DataLoader(
        train_data, 
        batch_size=common_params['batch_size'], 
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

    # Create and train the model
    tede_model = TEDE(
        n_sources=data_configs['n_sources'],
        output_dim=data_configs['n_bins'],
        d_model=common_params['d_model'],
        activation=common_params['activation'],
        nhead=common_params['nhead'],
        num_encoder_layers=common_params['num_encoder_layers'],
        dim_feedforward=common_params['dim_feedforward'],
        dropout=common_params['dropout'],
        entmax_alpha=common_params['entmax_alpha']
    )

    tede_model_lightning_training = TEDELightningTraining(
        model=tede_model,
        loss_function=kl_div,
        val_metric_functions=val_metric_functions,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr=common_params['learning_rate'],
        bins_centers=kNPE_bins_centers,
        monitor_metric=monitor_metric,
        model_res_visualizator=model_res_visualizator,
        dependent_params=dependent_params
    )

    trainer_tede = Trainer(
        max_epochs=1200,
        deterministic=True,
        accelerator=args.accelerator,
        devices="auto",
        precision=64,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            LearningRateMonitor(),
            PyTorchLightningPruningCallback(
                trial, monitor=monitor_metric), #monitor metric between trials
        ],
        enable_checkpointing=True,
    )

    print(tede_model)
    print(common_params)
    print(dependent_params)
    trainer_tede.fit(
        tede_model_lightning_training,
        train_dataloaders=train_loader,
        val_dataloaders=[
        val1_loader,
        val2_loader,
        ]
    )

    return trainer_tede.callback_metrics[monitor_metric].item()


# Create an Optuna study and run optimization
study = optuna.create_study(
    study_name="my_study",
    storage=f"sqlite:///{path_to_optuna_results}/seed_{args.seed}/tede_study.db",
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=50, max_resource="auto", reduction_factor=3),
)
study.optimize(objective, n_trials=args.n_trials) 

with open(f'{path_to_optuna_results}/seed_{args.seed}/tede_study_output.pkl', "wb") as f:
    pickle.dump(study, f)

best_params = study.best_params
best_params['seed'] = args.seed
best_params['best_value'] = study.best_value

with open(f'{path_to_optuna_results}/seed_{args.seed}/tede_best_hparams.pkl', 'wb') as fp:
    pickle.dump(best_params, fp) 

trials_dataframe = study.trials_dataframe()
trials_dataframe.to_csv(
    f'{path_to_optuna_results}/seed_{args.seed}/tede_trials_dataframe.csv',
    index=False
)

hparam_importances = optuna.importance.get_param_importances(study)
with open(f'{path_to_optuna_results}/seed_{args.seed}/tede_hparam_importances.pkl', 'wb') as fp:
    pickle.dump(hparam_importances, fp)  
