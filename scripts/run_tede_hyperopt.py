import os
import argparse
import pickle
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
optuna.logging.set_verbosity(optuna.logging.WARNING)

from neuromct.models.ml import TEDE
from neuromct.models.ml.callbacks import ModelResultsVisualizerCallback
from neuromct.models.ml.lightning_models import TEDELightningTraining
from neuromct.models.ml.losses import GeneralizedKLDivLoss
from neuromct.models.ml.metrics import LpNormDistance
from neuromct.configs import data_configs
from neuromct.utils import (
    create_dataset,
    define_transformations,
    res_visualizator_setup
)

parser = argparse.ArgumentParser(description='Run the hyperparameter optimization for the TEDE model')
parser.add_argument("--n_trials", type=int, default=100,
                        help='The number of Optuna trials (default=100).')
parser.add_argument("--accelerator", type=str, default="gpu",
                        help='Device used to train the TEDE model ("cpu" or "gpu", default="gpu").')
parser.add_argument("--seed", type=int, default=22,
                        help='Seed for reproducibility (default=22).')
parser.add_argument("--monitor_metric", type=str, default="val_cramer_metric",
                        help='''The main validation metric used during the hyperopt search
                                (i.e., used for the early stopping condition). Can be val_*_metric, 
                                where * is either "cramer" or "wasserstein", or "ks". 
                                Default=val_cramer_metric.''')
args = parser.parse_args()

approach_type = 'tede'
path_to_processed_data = data_configs['path_to_processed_data']
path_to_tede_hopt_results = data_configs['path_to_tede_hopt_results']

os.makedirs(f'{path_to_tede_hopt_results}/seed_{args.seed}', exist_ok=True)
for run_index in range(args.n_trials):
    os.makedirs(
        f"{path_to_tede_hopt_results}/seed_{args.seed}/trial_{run_index}/plots", 
        exist_ok=True
    )
    os.makedirs(
        f"{path_to_tede_hopt_results}/seed_{args.seed}/trial_{run_index}/predictions", 
        exist_ok=True
    )

kNPE_bins_edges = data_configs['kNPE_bins_edges']
kNPE_bins_centers = (kNPE_bins_edges[:-1] + kNPE_bins_edges[1:]) / 2
kNPE_bins_centers = torch.tensor(kNPE_bins_centers, dtype=torch.float32)
bin_size = data_configs['bin_size']

model_res_visualizator = res_visualizator_setup(data_configs)

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

def objective(trial):
    # path to save the results of the current trial
    path_to_savings = f"{path_to_tede_hopt_results}/seed_{args.seed}/trial_{trial.number}"

    # define callbacks and the logger
    logger = CSVLogger(save_dir=path_to_savings, name=f"tede_{trial.number}")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor=args.monitor_metric, mode="min")
    
    early_stopping_callback = EarlyStopping(
        monitor=args.monitor_metric, mode="min", patience=200)
    
    res_visualizer_callback = ModelResultsVisualizerCallback(
        res_visualizer=model_res_visualizator,
        base_path_to_savings=path_to_savings,
        plots_dir_name='plots',
        predictions_dir_name='predictions'
    )

    # Hyperparameter search space
    main_hparams = {
        'd_model': trial.suggest_int(
            'd_model', 50, 500, step=50),
        'nhead': trial.suggest_categorical(
            'nhead', [5, 10, 25]),
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
        'temperature': trial.suggest_float(
            'temperature', 0.25, 3.0, step=0.01), # if 1.0, pure nn.Softmax is used
        'batch_size': trial.suggest_categorical(
            'batch_size', [16, 32, 64, 128, 256, 512]),
    }

    optimizer_hparams = {}
    if main_hparams['lr_scheduler'] == 'ExponentialLR':
        lr_scheduler = optim.lr_scheduler.ExponentialLR
        optimizer_hparams['gamma'] = trial.suggest_float(
            'gamma', 0.80, 0.99, step=0.01)
    elif main_hparams['lr_scheduler'] == 'CosineAnnealingLR':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR
        optimizer_hparams['T_max'] = trial.suggest_int(
            'T_max', 5, 75, step=5)
    elif main_hparams['lr_scheduler'] == 'ReduceLROnPlateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau
        optimizer_hparams['reduction_factor'] = trial.suggest_float(
            'reduction_factor', 0.80, 0.99, step=0.01)
    else:
        lr_scheduler = None

    main_hparams['weight_decay'] = trial.suggest_float(
        'weight_decay', 1e-5, 1e-1, log=True)
    if main_hparams['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop
        optimizer_hparams['alpha'] = trial.suggest_float(
            'alpha', 0.9, 0.999, step=0.001)
    elif main_hparams['optimizer'] == 'AdamW':
        optimizer = optim.AdamW
        optimizer_hparams['beta1'] = trial.suggest_float(
            'beta1', 0.5, 0.95, step=0.01)
        optimizer_hparams['beta2'] = trial.suggest_float(
            'beta2', 0.9, 0.999, step=0.001)

    train_loader = DataLoader(
        train_data, 
        batch_size=main_hparams['batch_size'], 
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
        d_model=main_hparams['d_model'],
        activation=main_hparams['activation_function'],
        nhead=main_hparams['nhead'],
        num_encoder_layers=main_hparams['num_encoder_layers'],
        dim_feedforward=main_hparams['dim_feedforward'],
        dropout=main_hparams['dropout'],
        temperature=main_hparams['temperature'],
        bin_size=bin_size
    )

    tede_model_lightning_training = TEDELightningTraining(
        model=tede_model,
        loss_function=kl_div,
        val_metric_functions=val_metric_functions,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        optimizer_hparams=optimizer_hparams,
        lr=main_hparams['learning_rate'],
        weight_decay=main_hparams['weight_decay'],
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
            PyTorchLightningPruningCallback(
                trial, monitor=args.monitor_metric), #monitor metric between trials
        ],
        logger=logger,
        enable_checkpointing=True,
    )

    print(tede_model)
    print(main_hparams)
    print(optimizer_hparams)
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
        lr=main_hparams['learning_rate'],
        weight_decay=main_hparams['weight_decay'],
        bins_centers=kNPE_bins_centers,
        monitor_metric=args.monitor_metric,
    )
    torch.save(
        best_tede_model.model.state_dict(), 
        f"{path_to_savings}/tede_model.pth"
    )

    trial_value = checkpoint_callback.best_model_score.item()
    trial_info = {
        'trial_number': trial.number,
        'main_hparams': main_hparams,
        'optimizer_hparams': optimizer_hparams,
        'objective_value': trial_value,
    }

    params_filepath = os.path.join(
        path_to_savings, f'hparams_{trial.number}.json')
    with open(params_filepath, 'w') as f:
        json.dump(trial_info, f, indent=6)

    return trial_value

# Create an Optuna study and run optimization
storage_path = f"{path_to_tede_hopt_results}/seed_{args.seed}/tede_study.db"

if os.path.exists(storage_path) == False:
    use_initials = True
else:
    use_initials = False

study = optuna.create_study(
    study_name="tede_hp_optimization",
    storage=f"sqlite:///{storage_path}",
    direction='minimize',
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=args.seed),
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=50, max_resource="auto", reduction_factor=3),
)

if use_initials:
    # initial hyperparameters
    initial_hparams = {
        'd_model': 100,
        'nhead': 5,
        'num_encoder_layers': 4,
        'dim_feedforward': 128,
        'activation_function': 'relu',
        'learning_rate': 1e-4,
        'optimizer': 'AdamW',
        'lr_scheduler': 'CosineAnnealingLR',
        'dropout': 0.1,
        'temperature': 2.0,
        'batch_size': 32,
        'weight_decay': 1e-4,
        'T_max': 50,
        'beta1': 0.9,
        'beta2': 0.99,
    }
    study.enqueue_trial(initial_hparams)
study.optimize(objective, n_trials=args.n_trials) 

with open(f'{path_to_tede_hopt_results}/seed_{args.seed}/tede_study_output.pkl', "wb") as f:
    pickle.dump(study, f)

best_params = study.best_params
best_params['seed'] = args.seed
best_params['best_value'] = study.best_value

with open(f'{path_to_tede_hopt_results}/seed_{args.seed}/tede_best_hparams.pkl', 'wb') as fp:
    pickle.dump(best_params, fp) 

trials_dataframe = study.trials_dataframe()
trials_dataframe.to_csv(
    f'{path_to_tede_hopt_results}/seed_{args.seed}/tede_trials_dataframe.csv',
    index=False
)

hparam_importances = optuna.importance.get_param_importances(study)
with open(f'{path_to_tede_hopt_results}/seed_{args.seed}/tede_hparams_importances.pkl', 'wb') as fp:
    pickle.dump(hparam_importances, fp)  
