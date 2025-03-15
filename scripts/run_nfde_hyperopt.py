import os
import argparse
import pickle
import json

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import (
    DataLoader, 
    ConcatDataset
)

from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    LearningRateMonitor
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
optuna.logging.set_verbosity(optuna.logging.WARNING)

from neuromct.models.ml import NFDE
from neuromct.models.ml.callbacks import ModelResultsVisualizerCallback
from neuromct.models.ml.lightning_models import NFDELightningTraining
from neuromct.models.ml.metrics import LpNormDistance
from neuromct.configs import data_configs
from neuromct.utils import (
    create_dataset,
    res_visualizator_setup
)

mp.set_start_method('spawn', force=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def run_main():
    parser = argparse.ArgumentParser(description='Run the hyperparameter optimization for the NFDE model')
    parser.add_argument("--n_trials", type=int, default=100,
                            help='The number of Optuna trials (default=100).')
    parser.add_argument("--accelerator", type=str, default="cpu",
                            help='Device used to train the NFDE model ("cpu" or "gpu", default="cpu").')
    parser.add_argument("--seed", type=int, default=22,
                            help='Seed for reproducibility (default=22).')
    parser.add_argument("--monitor_metric", type=str, default="val_cramer_metric",
                            help='''The main validation metric used during the hyperopt search
                                    (i.e., used for the early stopping condition). Can be val_*_metric, 
                                    where * is either "cramer" or "wasserstein", or "ks". 
                                    Default=val_cramer_metric.''')
    args = parser.parse_args()

    approach_type = 'nfde'
    plot_every_n_train_epochs = 1
    path_to_processed_data = data_configs['path_to_processed_data']
    path_to_nfde_hopt_results = data_configs['path_to_nfde_hopt_results']
    en_limits = 0.0, 20.0
    n_en_values = 100000

    os.makedirs(f'{path_to_nfde_hopt_results}/seed_{args.seed}', exist_ok=True)
    for run_index in range(args.n_trials):
        os.makedirs(
            f"{path_to_nfde_hopt_results}/seed_{args.seed}/trial_{run_index}/plots", 
            exist_ok=True
        )
        os.makedirs(
            f"{path_to_nfde_hopt_results}/seed_{args.seed}/trial_{run_index}/predictions", 
            exist_ok=True
        )

    model_res_visualizator = res_visualizator_setup(
        data_configs, plot_every_n_train_epochs)

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

    loss_function = 'kl-div'
    wasserstein_distance = LpNormDistance(p=1) # Wasserstein distance
    cramer_distance = LpNormDistance(p=2) # Cramér-von Mises distance
    ks_distance = LpNormDistance(p=torch.inf) # Kolmogorov-Smirnov distance
    val_metric_functions = {
        "wasserstein": wasserstein_distance, 
        "cramer": cramer_distance,
        "ks": ks_distance
    }

    def objective(trial):   
        # Hyperparameter search space
        main_hparams = {
            'n_flows': trial.suggest_int(
                'n_flows', 2, 4, step=1),#20, 200, step=10),
            'n_units': trial.suggest_categorical(
                'n_units', [5, 10, 20, 50]),
            'activation_function': trial.suggest_categorical(
                'activation_function', 
                ['relu', 'gelu', 'tanh', 'silu']),
            'flow_type': trial.suggest_categorical(
                'flow_type', ['planar', 'radial']),
            'learning_rate': trial.suggest_float(
                'learning_rate', 1e-6, 1e-3, log=True),
            'optimizer': trial.suggest_categorical(
                'optimizer', ['AdamW', 'RMSprop']),
            'lr_scheduler': trial.suggest_categorical(
                'lr_scheduler', 
                ['ExponentialLR', 'ReduceLROnPlateau', 
                'CosineAnnealingLR', 'None']),
            'dropout': trial.suggest_float(
                'dropout', 0.0, 0.25, step=0.01),
            'batch_size': trial.suggest_int(
                'batch_size', 1, 5, step=1),
        }

        optimizer_hparams = {}
        if main_hparams['lr_scheduler'] == 'ExponentialLR':
            lr_scheduler = optim.lr_scheduler.ExponentialLR
            optimizer_hparams['gamma'] = trial.suggest_float(
                'gamma', 0.80, 0.99, step=0.01)
        elif main_hparams['lr_scheduler'] == 'CosineAnnealingLR':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR
            optimizer_hparams['T_max'] = trial.suggest_int(
                'T_max', 5, 20, step=1)
        elif main_hparams['lr_scheduler'] == 'ReduceLROnPlateau':
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau
            optimizer_hparams['reduction_factor'] = trial.suggest_float(
                'reduction_factor', 0.80, 0.99, step=0.01)
        else:
            lr_scheduler = None

        main_hparams['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-6, 1e-2, log=True)
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

        # path to save the results of the current trial
        path_to_savings = f"{path_to_nfde_hopt_results}/seed_{args.seed}/trial_{trial.number}"

        # define callbacks and the logger
        logger = CSVLogger(
            save_dir=path_to_savings, 
            name=f"nfde_{trial.number}"
        )

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, monitor=args.monitor_metric, mode="min")
        
        early_stopping_callback = EarlyStopping(
            monitor=args.monitor_metric, mode="min", patience=10)
        
        res_visualizer_callback = ModelResultsVisualizerCallback(
            res_visualizer=model_res_visualizator,
            approach_type=approach_type,
            base_path_to_savings=path_to_savings,
            plots_dir_name='plots',
            predictions_dir_name='predictions',
            val_metric_names=list(val_metric_functions.keys())
        )

        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor=args.monitor_metric)

        train_loader = DataLoader(
            train_data, 
            batch_size=main_hparams['batch_size'],
            shuffle=True, 
            pin_memory=True
        )

        val1_loader = DataLoader(
            val1_data, 
            batch_size=main_hparams['batch_size'],
            shuffle=False, 
            pin_memory=True
        )

        val2_loader = DataLoader(
            val2_data, 
            batch_size=main_hparams['batch_size'],
            shuffle=False, 
            pin_memory=True
        )

        # Create and train the model
        nfde_model = NFDE(
            n_flows=main_hparams['n_flows'],
            n_conditions=data_configs['n_conditions'],
            n_sources=data_configs['n_sources'],
            n_units=main_hparams['n_units'],
            activation=main_hparams['activation_function'],
            flow_type=main_hparams['flow_type'],
            dropout=main_hparams['dropout'],
        )

        nfde_model_lightning_training = NFDELightningTraining(
            model=nfde_model,
            loss_function=loss_function,
            val_metric_functions=val_metric_functions,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            optimizer_hparams=optimizer_hparams,
            lr=main_hparams['learning_rate'],
            weight_decay=main_hparams['weight_decay'],
            monitor_metric=args.monitor_metric,
            n_en_values=n_en_values,
            en_limits=en_limits
        )

        trainer_nfde = Trainer(
            max_epochs=100,
            accelerator=args.accelerator,
            strategy="ddp_spawn",
            devices=50,
            precision="64",
            callbacks=[
                checkpoint_callback,
                early_stopping_callback,
                res_visualizer_callback,
                LearningRateMonitor(),
                pruning_callback
            ],
            logger=logger,
            enable_checkpointing=True,
        )

        print(nfde_model)
        print(main_hparams)
        print(optimizer_hparams)
        trainer_nfde.fit(
            nfde_model_lightning_training,
            train_dataloaders=train_loader,
            val_dataloaders=[
                val1_loader,
                val2_loader,
            ]
        )

        pruning_callback.check_pruned()

        # Load the full checkpoint dictionary
        checkpoint_dict = torch.load(
            checkpoint_callback.best_model_path, map_location="cpu")

        for key in checkpoint_dict['callbacks'].keys():
            if "ModelCheckpoint" in key:
                model_checkpoint_key = key
        best_model_score = checkpoint_dict['callbacks'][model_checkpoint_key]["best_model_score"]
        trial_value = best_model_score.item()
        print(trial_value)
        
        best_nfde_model = NFDELightningTraining.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path,
            model=nfde_model,
            loss_function=loss_function,
            val_metric_functions=val_metric_functions,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            optimizer_hparams=optimizer_hparams,
            lr=main_hparams['learning_rate'],  
            weight_decay=main_hparams['weight_decay'],
            monitor_metric=args.monitor_metric,
            n_en_values=n_en_values,
            en_limits=en_limits
        )

        torch.save(
            best_nfde_model.model.state_dict(), 
            f"{path_to_savings}/nfde_model.pth"
        )

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
    storage_path = f"{path_to_nfde_hopt_results}/seed_{args.seed}/nfde_study.db"

    if os.path.exists(storage_path) == False:
        use_initials = True
    else:
        use_initials = False

    study = optuna.create_study(
        study_name="nfde_hp_optimization",
        storage=f"sqlite:///{storage_path}",
        direction='minimize',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=10, max_resource="auto", reduction_factor=3),
    )

    if use_initials:
        # initial hyperparameters
        initial_hparams = {
            'n_flows': 125,
            'n_units': 20,
            'flow_type': "planar",
            'activation_function': 'silu',
            'learning_rate': 1e-4,
            'optimizer': 'AdamW',
            'lr_scheduler': 'ReduceLROnPlateau',
            'dropout': 0.0,
            'batch_size': 1,
            'weight_decay': 1e-4,
            'reduction_factor': 0.9,
            'beta1': 0.9,
            'beta2': 0.999,
        }
        study.enqueue_trial(initial_hparams)
    study.optimize(objective, n_trials=args.n_trials, timeout=600) 

    with open(f'{path_to_nfde_hopt_results}/seed_{args.seed}/nfde_study_output.pkl', "wb") as f:
        pickle.dump(study, f)

    best_params = study.best_params
    best_params['seed'] = args.seed
    best_params['best_value'] = study.best_value

    with open(f'{path_to_nfde_hopt_results}/seed_{args.seed}/nfde_best_hparams.pkl', 'wb') as fp:
        pickle.dump(best_params, fp) 

    trials_dataframe = study.trials_dataframe()
    trials_dataframe.to_csv(
        f'{path_to_nfde_hopt_results}/seed_{args.seed}/nfde_trials_dataframe.csv',
        index=False
    )

    hparam_importances = optuna.importance.get_param_importances(study)
    with open(f'{path_to_nfde_hopt_results}/seed_{args.seed}/nfde_hparams_importances.pkl', 'wb') as fp:
        pickle.dump(hparam_importances, fp)  

if __name__ == "__main__":
    run_main()
