"""
This script provides an interface for running hyperparameter optimization
for both Normalizing Flows Density Estimator (NFDE) and Transformer Encoder Density Estimator (TEDE)
models. It uses Optuna for hyperparameter optimization and PyTorch Lightning for training.

Example usage:
    # For NFDE optimization
    python run_hyperopt.py --approach_type nfde --n_trials 100 --accelerator cpu --seed 22

    # For TEDE optimization
    python run_hyperopt.py --approach_type tede --n_trials 250 --accelerator gpu --seed 222

The script saves all trial results, best parameters, and model states in the specified output directories.
"""

import os
import argparse
import pickle
import json

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, ConcatDataset

from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from neuromct.models.ml import NFDE, TEDE
from neuromct.models.ml.callbacks import ModelResultsVisualizerCallback
from neuromct.models.ml.lightning_models import NFDELightningTraining, TEDELightningTraining
from neuromct.models.ml.losses import GeneralizedKLDivLoss
from neuromct.models.ml.metrics import LpNormDistance
from neuromct.configs import data_configs
from neuromct.utils import (
    create_dataset,
    define_transformations,
    res_visualizator_setup
)

def setup_environment(approach_type):
    """Set up environment-specific configurations for different approaches.

    Args:
        approach_type (str): Type of approach ('nfde' or 'tede'). NFDE requires specific
            multiprocessing and threading settings.

    Note:
        For NFDE, this function:
        - Sets multiprocessing start method to 'spawn'
        - Limits number of threads for better performance with DDP
    """
    if approach_type == 'nfde':
        mp.set_start_method('spawn', force=True)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

def setup_data_and_paths(args, approach_type):
    """Set up data paths and model-specific configurations.

    Args:
        args (argparse.Namespace): Command line arguments containing seed and n_trials
        approach_type (str): Type of approach ('nfde' or 'tede')

    Returns:
        tuple: Contains:
            - path_to_processed_data (str): Path to the processed dataset
            - path_to_hopt_results (str): Path to save hyperopt results
            - plot_every_n_train_epochs (int): Frequency of plotting during training
            - extra_config (dict): Additional configuration parameters specific to each approach
    """
    path_to_processed_data = data_configs['path_to_processed_data']
    path_to_hopt_results = (data_configs['path_to_nfde_hopt_results'] 
                           if approach_type == 'nfde' 
                           else data_configs['path_to_tede_hopt_results'])

    # Create directories for trials
    os.makedirs(f'{path_to_hopt_results}/seed_{args.seed}', exist_ok=True)
    for run_index in range(args.n_trials):
        os.makedirs(
            f"{path_to_hopt_results}/seed_{args.seed}/trial_{run_index}/plots", 
            exist_ok=True
        )
        os.makedirs(
            f"{path_to_hopt_results}/seed_{args.seed}/trial_{run_index}/predictions", 
            exist_ok=True
        )

    # Set up model-specific configurations
    if approach_type == 'nfde':
        en_limits = (0.0, 20.0)
        n_en_values = 100000
        extra_config = {
            'en_limits': en_limits,
            'n_en_values': n_en_values
        }
    else:  # tede
        kNPE_bins_edges = data_configs['kNPE_bins_edges']
        kNPE_bins_centers = torch.tensor(
            (kNPE_bins_edges[:-1] + kNPE_bins_edges[1:]) / 2, 
            dtype=torch.float64
        )
        bin_size = data_configs['bin_size']
        training_data_transforms = define_transformations("training", bin_size)
        val_data_transforms = define_transformations("val", bin_size)
        extra_config = {
            'kNPE_bins_centers': kNPE_bins_centers,
            'bin_size': bin_size,
            'training_data_transforms': training_data_transforms,
            'val_data_transforms': val_data_transforms
        }

    return path_to_processed_data, path_to_hopt_results, extra_config

def create_dataloaders(approach_type, path_to_processed_data, 
                       batch_size, data_transforms=False):
    """Create data loaders for training and validation.

    Args:
        approach_type (str): Type of approach ('nfde' or 'tede')
        path_to_processed_data (str): Path to the processed dataset
        batch_size (int): Batch size for training
        data_transforms (dict, optional): Data transformations for TEDE approach

    Returns:
        tuple: Contains:
            - train_loader (DataLoader): Training data
            - val1_loader (DataLoader): Validation data №1
            - val2_loader (DataLoader): Validation data №2
    """
    train_data = create_dataset(
        "training",
        path_to_processed_data,
        approach_type,
        data_transforms['training_data_transforms'] \
            if approach_type == 'tede' else False
    )

    val1_data = create_dataset(
        "val1",
        path_to_processed_data,
        approach_type,
        data_transforms['val_data_transforms'] \
            if approach_type == 'tede' else False
    )

    val2_data = []
    for i in range(3):
        val2_i_data = create_dataset(
            f"val2_{i+1}",
            path_to_processed_data,
            approach_type,
            data_transforms['val_data_transforms'] \
                if approach_type == 'tede' else False,
            val2_rates=True
        )
        val2_data.append(val2_i_data)
    val2_data = ConcatDataset(val2_data)

    # Create dataloaders with appropriate batch sizes
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=20 if approach_type == 'tede' else 0,
        pin_memory=True
    )

    val1_loader = DataLoader(
        val1_data,
        batch_size=val1_data.__len__() \
            if approach_type == 'tede' else batch_size,
        shuffle=False,
        pin_memory=True
    )

    val2_loader = DataLoader(
        val2_data,
        batch_size=val2_data.__len__() \
            if approach_type == 'tede' else batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val1_loader, val2_loader

def get_nfde_search_space(trial):
    """Define hyperparameter search space for NFDE model.

    Args:
        trial (optuna.Trial): Optuna trial object for suggesting parameters

    Returns:
        dict: Dictionary containing hyperparameter configurations for NFDE, including tunable ones:
            - n_flows: Number of normalizing flows
            - n_units: Number of hidden units in each flow's neural network
            - activation_function: Activation function for the flow's parameter neural networks
            - lr_scheduler: Learning rate scheduler type
    """
    main_hparams = {
        'n_flows': trial.suggest_int(
            'n_flows', 100, 200, step=5),
        'n_units': trial.suggest_int(
            'n_units', 10, 50, step=5),
        'activation_function': trial.suggest_categorical(
            'activation_function', 
            ['relu', 'gelu', 'tanh', 'silu']),
        'lr_scheduler': trial.suggest_categorical(
            'lr_scheduler', 
            ['ReduceLROnPlateau', 'CosineAnnealingLR']),
    }

    main_hparams['batch_size'] = 1
    return main_hparams

def get_tede_search_space(trial):
    """Define hyperparameter search space for TEDE model.

    Args:
        trial (optuna.Trial): Optuna trial object for suggesting parameters

    Returns:
        dict: Dictionary containing hyperparameter configurations for TEDE, including:
            - d_model: Dimension of the transformer model's internal representation
            - nhead: Number of attention heads
            - num_encoder_layers: Number of transformer encoder layers
            - dim_feedforward: Dimension of feedforward network
            - activation_function: Type of activation function
            - learning_rate: Initial learning rate
            - optimizer: Optimizer type
            - lr_scheduler: Learning rate scheduler type
            - dropout: Dropout rate
            - temperature: Temperature parameter for Softmax
            - batch_size: Training batch size
    """
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
            ['ExponentialLR', 'ReduceLROnPlateau', 
            'CosineAnnealingLR', 'None']),
        'dropout': trial.suggest_float(
            'dropout', 0.0, 0.5, step=0.05),
        'temperature': trial.suggest_float(
            'temperature', 0.25, 3.0, step=0.01), # if 1.0, pure nn.Softmax is used
        'batch_size': trial.suggest_categorical(
            'batch_size', [16, 32, 64, 128, 256, 512]),
    }
    return main_hparams

def get_optimizer_params(trial, main_hparams, approach_type):
    """Get optimizer and learning rate scheduler parameters.

    Args:
        trial (optuna.Trial): Optuna trial object for suggesting parameters
        main_hparams (dict): Main hyperparameters dictionary

    Returns:
        tuple: Contains:
            - optimizer (torch.optim.Optimizer): Selected optimizer class
            - lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Selected scheduler class
            - optimizer_hparams (dict): Additional optimizer parameters
    """
    optimizer_hparams = {}
    
    # Learning rate scheduler parameters
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

    # Optimizer parameters
    if approach_type == 'tede':
        main_hparams['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-6, 1e-1, log=True)
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
    else:
        main_hparams['learning_rate'] = 1e-4
        main_hparams['weight_decay'] = 1e-4
        optimizer = optim.AdamW
        optimizer_hparams['beta1'] = trial.suggest_float(
            'beta1', 0.5, 0.95, step=0.01)
        optimizer_hparams['beta2'] = trial.suggest_float(
            'beta2', 0.9, 0.999, step=0.001)

    return optimizer, lr_scheduler, optimizer_hparams

def get_initial_hparams(approach_type):
    """Get initial hyperparameters for the optimization study.

    Args:
        approach_type (str): Type of approach ('nfde' or 'tede')

    Returns:
        dict: Initial hyperparameters for the specified approach
    """
    if approach_type == 'nfde':
        return {
            'n_flows': 125,
            'n_units': 20,
            'activation_function': 'silu',
            'learning_rate': 1e-4,
            'optimizer': 'AdamW',
            'lr_scheduler': 'ReduceLROnPlateau',
            'batch_size': 1,
            'weight_decay': 1e-4,
            'reduction_factor': 0.9,
            'beta1': 0.9,
            'beta2': 0.999,
        }
    else:  # tede
        return {
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

def create_model_and_training(approach_type, main_hparams, optimizer, lr_scheduler, 
                              optimizer_hparams, val_metric_functions, monitor_metric, 
                              extra_config):
    """Create model and lightning training module based on approach type.

    Args:
        approach_type (str): Type of approach ('nfde' or 'tede')
        main_hparams (dict): Main hyperparameters
        optimizer (torch.optim.Optimizer): Optimizer class
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler class
        optimizer_hparams (dict): Additional optimizer parameters
        val_metric_functions (dict): Validation metric functions
        monitor_metric (str): Metric to monitor for model selection
        extra_config (dict): Additional configuration parameters

    Returns:
        tuple: Contains:
            - model (nn.Module): NFDE or TEDE model
            - model_lightning_training (LightningModule): Training module
    """
    if approach_type == 'nfde':
        model = NFDE(
            n_flows=main_hparams['n_flows'],
            n_conditions=data_configs['n_conditions'],
            n_sources=data_configs['n_sources'],
            n_units=main_hparams['n_units'],
            activation=main_hparams['activation_function'],
            flow_type="planar",
        )
        
        model_lightning_training = NFDELightningTraining(
            model=model,
            loss_function='kl-div',
            val_metric_functions=val_metric_functions,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            optimizer_hparams=optimizer_hparams,
            lr=main_hparams['learning_rate'],
            weight_decay=main_hparams['weight_decay'],
            monitor_metric=monitor_metric,
            n_en_values=extra_config['n_en_values'],
            en_limits=extra_config['en_limits']
        )
    else:  # tede
        model = TEDE(
            n_sources=data_configs['n_sources'],
            output_dim=data_configs['n_bins'],
            d_model=main_hparams['d_model'],
            activation=main_hparams['activation_function'],
            nhead=main_hparams['nhead'],
            num_encoder_layers=main_hparams['num_encoder_layers'],
            dim_feedforward=main_hparams['dim_feedforward'],
            dropout=main_hparams['dropout'],
            temperature=main_hparams['temperature'],
            bin_size=extra_config['bin_size']
        )
        
        kl_div = GeneralizedKLDivLoss(
            log_input=False, log_target=False, reduction='batchmean')
        
        model_lightning_training = TEDELightningTraining(
            model=model,
            loss_function=kl_div,
            val_metric_functions=val_metric_functions,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            optimizer_hparams=optimizer_hparams,
            lr=main_hparams['learning_rate'],
            weight_decay=main_hparams['weight_decay'],
            bins_centers=extra_config['kNPE_bins_centers'],
            monitor_metric=monitor_metric
        )
    
    return model, model_lightning_training

def objective(trial, args, path_to_processed_data, 
              path_to_hopt_results, extra_config):
    """Objective function for hyperparameter optimization.

    This function defines the optimization objective for Optuna. It:
    1. Sets up model hyperparameters and training configuration
    2. Creates and trains the model
    3. Evaluates the model performance
    4. Saves results and model state

    Args:
        trial (optuna.Trial): Optuna trial object
        args (argparse.Namespace): Command line arguments
        path_to_processed_data (str): Path to the processed dataset
        path_to_hopt_results (str): Path to save results
        extra_config (dict): Additional configuration parameters

    Returns:
        float: Validation metric value (lower is better)
    """
    # Initialize KL divergence loss for TEDE
    if args.approach_type == 'tede':
        kl_div = GeneralizedKLDivLoss(
            log_input=False, log_target=False, reduction='batchmean')
    
    # Get hyperparameters search space
    main_hparams = (get_nfde_search_space(trial) if args.approach_type == 'nfde' 
                   else get_tede_search_space(trial))
    
    # Get optimizer parameters
    optimizer, lr_scheduler, optimizer_hparams = get_optimizer_params(
        trial, main_hparams, args.approach_type)

    # Set up metrics
    wasserstein_distance = LpNormDistance(p=1)
    cramer_distance = LpNormDistance(p=2)
    ks_distance = LpNormDistance(p=torch.inf)
    val_metric_functions = {
        "wasserstein": wasserstein_distance,
        "cramer": cramer_distance,
        "ks": ks_distance
    }

    # Set up paths and callbacks
    path_to_savings = f"{path_to_hopt_results}/seed_{args.seed}/trial_{trial.number}"
    
    logger = CSVLogger(
        save_dir=path_to_savings,
        name=f"{args.approach_type}_{trial.number}"
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor=args.monitor_metric, mode="min")
    
    early_stopping_callback = EarlyStopping(
        monitor=args.monitor_metric, 
        mode="min", 
        patience=200 if args.approach_type == 'tede' else 50
    )

    model_res_visualizator = res_visualizator_setup(
        data_configs, 
        plot_every_n_train_epochs=50 if args.approach_type == 'tede' else 1
    )
    
    res_visualizer_callback = ModelResultsVisualizerCallback(
        res_visualizer=model_res_visualizator,
        approach_type=args.approach_type,
        base_path_to_savings=path_to_savings,
        plots_dir_name='plots',
        predictions_dir_name='predictions',
        val_metric_names=list(val_metric_functions.keys())
    )

    pruning_callback = PyTorchLightningPruningCallback(
        trial, monitor=args.monitor_metric)

    # Create dataloaders
    train_loader, val1_loader, val2_loader = create_dataloaders(
        args.approach_type,
        path_to_processed_data,
        main_hparams['batch_size'],
        extra_config if args.approach_type == 'tede' else False
    )

    # Create model and training module
    model, model_lightning_training = create_model_and_training(
        args.approach_type, main_hparams, optimizer, lr_scheduler, optimizer_hparams,
        val_metric_functions, args.monitor_metric, extra_config
    )

    # Create trainer
    trainer = Trainer(
        max_epochs=2000 if args.approach_type == 'tede' else 300,
        accelerator=args.accelerator,
        strategy="ddp_spawn" if args.approach_type == 'nfde' else None,
        devices=50 if args.approach_type == 'nfde' else "auto",
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

    print(model)
    print(main_hparams)
    print(optimizer_hparams)

    # Train model
    trainer.fit(
        model_lightning_training,
        train_dataloaders=train_loader,
        val_dataloaders=[val1_loader, val2_loader]
    )

    if args.approach_type == 'nfde':
        pruning_callback.check_pruned()

    # Load the best model and get the score
    callbacks_dict = torch.load(
        checkpoint_callback.best_model_path, 
        map_location="cpu"
    )['callbacks']

    for key in callbacks_dict.keys():
        if "ModelCheckpoint" in key:
            model_checkpoint_key = key
    best_model_score = callbacks_dict[model_checkpoint_key]["best_model_score"]
    trial_value = best_model_score.item()
    print(trial_value)

    # Save the best model state
    best_model = (NFDELightningTraining if args.approach_type == 'nfde' else TEDELightningTraining).load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
        model=model,
        loss_function='kl-div' if args.approach_type == 'nfde' else kl_div,
        val_metric_functions=val_metric_functions,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        optimizer_hparams=optimizer_hparams,
        lr=main_hparams['learning_rate'],
        weight_decay=main_hparams['weight_decay'],
        monitor_metric=args.monitor_metric,
        **({
            'n_en_values': extra_config['n_en_values'],
            'en_limits': extra_config['en_limits']
        } if args.approach_type == 'nfde' else {
            'bins_centers': extra_config['kNPE_bins_centers']
        })
    )

    torch.save(
        best_model.model.state_dict(),
        f"{path_to_savings}/{args.approach_type}_model.pth"
    )

    # Save trial information
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

def main():
    """Main function for running hyperparameter optimization.

    This function:
    1. Parses command line arguments
    2. Sets up the environment and configurations
    3. Creates and configures the Optuna study
    4. Runs the optimization process
    5. Saves the results and best parameters

    Command line arguments:
        --approach_type: Type of approach ('nfde' or 'tede')
        --n_trials: Number of optimization trials
        --accelerator: Device to use for training
        --seed: Random seed for reproducibility
        --monitor_metric: Metric to monitor for optimization
    """
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization')
    parser.add_argument('--approach_type', type=str, choices=['nfde', 'tede'], required=True,
                      help='Choose the approach type: nfde or tede')
    parser.add_argument("--n_trials", type=int, default=100,
                      help='The number of Optuna trials (default=100).')
    parser.add_argument("--accelerator", type=str, default="gpu",
                      help='Device used for training ("cpu" or "gpu", default="gpu").')
    parser.add_argument("--seed", type=int, default=22,
                      help='Seed for reproducibility (default=22).')
    parser.add_argument("--monitor_metric", type=str, default="val_cramer_metric",
                      help='The main validation metric used during the hyperopt search.')
    args = parser.parse_args()

    # Set up environment and configurations
    setup_environment(args.approach_type)
    path_to_processed_data, path_to_hopt_results, extra_config = setup_data_and_paths(
        args, args.approach_type)

    seed_everything(args.seed, workers=True)

    # Create and configure the study
    storage_path = f"{path_to_hopt_results}/seed_{args.seed}/{args.approach_type}_study.db"
    use_initials = not os.path.exists(storage_path)

    study = optuna.create_study(
        study_name=f"{args.approach_type}_hp_optimization",
        storage=f"sqlite:///{storage_path}",
        direction='minimize',
        load_if_exists=True,
        sampler=TPESampler(seed=args.seed),
        pruner=HyperbandPruner(
            min_resource=50 if args.approach_type == 'tede' else 20,
            max_resource="auto",
            reduction_factor=3
        ),
    )

    if use_initials:
        initial_hparams = get_initial_hparams(args.approach_type)
        study.enqueue_trial(initial_hparams)

    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, args, path_to_processed_data, 
            path_to_hopt_results, extra_config
        ),
        n_trials=args.n_trials,
    )

    # Save results
    results_base_path = f'{path_to_hopt_results}/seed_{args.seed}'
    
    with open(f'{results_base_path}/{args.approach_type}_study_output.pkl', "wb") as f:
        pickle.dump(study, f)

    best_params = study.best_params
    best_params['seed'] = args.seed
    best_params['best_value'] = study.best_value

    with open(f'{results_base_path}/{args.approach_type}_best_hparams.pkl', 'wb') as fp:
        pickle.dump(best_params, fp)

    trials_dataframe = study.trials_dataframe()
    trials_dataframe.to_csv(
        f'{results_base_path}/{args.approach_type}_trials_dataframe.csv',
        index=False
    )

    hparam_importances = optuna.importance.get_param_importances(study)
    with open(f'{results_base_path}/{args.approach_type}_hparams_importances.pkl', 'wb') as fp:
        pickle.dump(hparam_importances, fp)

if __name__ == "__main__":
    main() 