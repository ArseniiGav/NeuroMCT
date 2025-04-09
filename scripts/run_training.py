"""
This script provides a unified interface for training both Normalizing Flows Density Estimator (NFDE)
and Transformer Encoder Density Estimator (TEDE) models. It uses PyTorch Lightning for training
and supports different configurations for each approach.

The script handles:
- Data loading and preprocessing
- Model initialization and training
- Model checkpointing and early stopping
- Results visualization and logging

Example usage:
    # For NFDE training
    python run_training.py --approach_type nfde

    # For TEDE training
    python run_training.py --approach_type tede

The script saves trained models, logs, and visualizations in the specified output directories.
"""

import os
import argparse
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

from neuromct.models.ml import NFDE, TEDE
from neuromct.models.ml.callbacks import ModelResultsVisualizerCallback
from neuromct.models.ml.lightning_models import NFDELightningTraining, TEDELightningTraining
from neuromct.models.ml.losses import GeneralizedKLDivLoss
from neuromct.models.ml.metrics import LpNormDistance
from neuromct.configs import data_configs
from neuromct.utils import (
    nfde_argparse,
    tede_argparse,
    create_dataset,
    define_transformations,
    res_visualizator_setup
)

def setup_common_components(args, approach_type, path_to_training_results):
    """Set up components common to both NFDE and TEDE training.

    This function initializes components used by both approaches, including:
    - Directory structure for results
    - Metrics for validation
    - Optimizer and learning rate scheduler
    - Callbacks for model checkpointing and monitoring
    - Result visualization and logging

    Args:
        args (argparse.Namespace): Command line arguments
        approach_type (str): Type of approach ('nfde' or 'tede')
        path_to_training_results (str): Base path for saving training results

    Returns:
        tuple: Contains:
            - optimizer (torch.optim.Optimizer): Selected optimizer class
            - optimizer_hparams (dict): Optimizer parameters
            - lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
            - val_metric_functions (dict): Dictionary of validation metrics
            - checkpoint_callback (ModelCheckpoint): Model checkpointing callback
            - early_stopping_callback (EarlyStopping): Early stopping callback
            - res_visualizer_callback (ModelResultsVisualizerCallback): Results visualization
            - logger (CSVLogger): Training logger
    """
    os.makedirs(f'{path_to_training_results}', exist_ok=True)
    os.makedirs(f'{path_to_training_results}/plots', exist_ok=True)
    os.makedirs(f'{path_to_training_results}/predictions', exist_ok=True)
    os.makedirs(f'{path_to_training_results}/values_to_plot', exist_ok=True)

    # Set up metrics
    wasserstein_distance = LpNormDistance(p=1)  # Wasserstein distance
    cramer_distance = LpNormDistance(p=2)  # Cramér-von Mises distance
    ks_distance = LpNormDistance(p=torch.inf)  # Kolmogorov-Smirnov distance
    val_metric_functions = {
        "wasserstein": wasserstein_distance,
        "cramer": cramer_distance,
        "ks": ks_distance
    }

    # Set up optimizer
    optimizer_hparams = {}
    if args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop
        optimizer_hparams['alpha'] = args.alpha
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW
        optimizer_hparams['beta1'] = args.beta1
        optimizer_hparams['beta2'] = args.beta2

    # Set up learning rate scheduler
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

    # Set up callbacks
    monitor_metric = "val_cramer_metric"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor=monitor_metric, mode="min")
    
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric, 
        mode="min", 
        patience=200 if approach_type == 'tede' else 100
    )

    model_res_visualizator = res_visualizator_setup(
        data_configs, plot_every_n_train_epochs=1)
    
    res_visualizer_callback = ModelResultsVisualizerCallback(
        res_visualizer=model_res_visualizator,
        approach_type=approach_type,
        base_path_to_savings=path_to_training_results,
        plots_dir_name='plots',
        predictions_dir_name='predictions',
        values_to_plot_dir_name='values_to_plot',
        val_metric_names=list(val_metric_functions.keys())
    )

    logger = CSVLogger(
        save_dir=path_to_training_results,
        name="training_logs"
    )

    return (optimizer, optimizer_hparams, lr_scheduler, val_metric_functions,
            checkpoint_callback, early_stopping_callback, res_visualizer_callback, logger)

def create_dataloaders(approach_type, path_to_processed_data, batch_size, bin_size=None):
    """Create data loaders for training and validation.

    Args:
        approach_type (str): Type of approach ('nfde' or 'tede')
        path_to_processed_data (str): Path to the processed dataset
        batch_size (int): Batch size for training
        bin_size (float, optional): Bin size for TEDE approach

    Returns:
        tuple: Contains:
            - train_loader (DataLoader): Training data loader
            - val1_loader (DataLoader): First validation data loader
            - val2_loader (DataLoader): Second validation data loader with rate variations

    Note:
        For TEDE, this function applies additional data transformations (Poisson noise
        and PDF construction for training, PDF construction only for validation).
    """
    if approach_type == 'tede':
        training_data_transforms = define_transformations("training", bin_size)
        val_data_transforms = define_transformations("val", bin_size)
    
    train_data = create_dataset(
        "training",
        path_to_processed_data,
        approach_type,
        training_data_transforms if approach_type == 'tede' else False
    )

    val1_data = create_dataset(
        "val1",
        path_to_processed_data,
        approach_type,
        val_data_transforms if approach_type == 'tede' else False
    )

    val2_data = []
    for i in range(3):
        val2_i_data = create_dataset(
            f"val2_{i+1}",
            path_to_processed_data,
            approach_type,
            val_data_transforms if approach_type == 'tede' else False,
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
        batch_size=val1_data.__len__() if approach_type == 'tede' else batch_size,
        shuffle=False,
        pin_memory=True
    )

    val2_loader = DataLoader(
        val2_data,
        batch_size=val2_data.__len__() if approach_type == 'tede' else batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val1_loader, val2_loader

def main():
    """Main function for model training.

    This function:
    1. Parses command line arguments for the specified approach
    2. Sets up the environment and configurations
    3. Creates data loaders and model components
    4. Trains the model
    5. Saves the best model state

    The function handles both NFDE and TEDE approaches with their specific:
    - Model architectures and configurations
    - Training strategies (DDP for NFDE, automatic device selection for TEDE)
    - Early stopping conditions
    - Data preprocessing steps
    """
    # Get command line arguments for approach type
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach_type', type=str, choices=['nfde', 'tede'], required=True,
                      help='Choose the approach type: nfde or tede')
    approach_args = parser.parse_args()
    approach_type = approach_args.approach_type
    
    # Set up paths and configurations
    base_path_to_models = data_configs['base_path_to_models']
    path_to_processed_data = data_configs['path_to_processed_data']
    path_to_training_results = (data_configs['path_to_tede_training_results'] 
                              if approach_type == 'tede' 
                              else data_configs['path_to_nfde_training_results'])

    # Parse remaining arguments based on approach type
    if approach_type == 'nfde':
        args = nfde_argparse()
        mp.set_start_method('spawn', force=True)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        en_limits = (0.0, 20.0)
    else:  # tede
        args = tede_argparse()
        kNPE_bins_edges = data_configs['kNPE_bins_edges']
        kNPE_bins_centers = torch.tensor(
            (kNPE_bins_edges[:-1] + kNPE_bins_edges[1:]) / 2, 
            dtype=torch.float64
        )
        bin_size = data_configs['bin_size']

    seed_everything(args.seed, workers=True)

    # Set up common components
    (optimizer, optimizer_hparams, lr_scheduler, val_metric_functions,
     checkpoint_callback, early_stopping_callback, res_visualizer_callback,
     logger) = setup_common_components(args, approach_type, path_to_training_results)

    # Create dataloaders
    train_loader, val1_loader, val2_loader = create_dataloaders(
        approach_type, 
        path_to_processed_data, 
        args.batch_size,
        bin_size if approach_type == 'tede' else None
    )

    # Create model based on approach type
    if approach_type == 'nfde':
        model = NFDE(
            n_flows=args.n_flows,
            n_conditions=data_configs['n_conditions'],
            n_sources=args.n_sources,
            n_units=args.n_units,
            activation=args.activation_function,
            flow_type=args.flow_type,
        )
        
        model_lightning_training = NFDELightningTraining(
            model=model,
            loss_function=args.loss_function,
            val_metric_functions=val_metric_functions,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            optimizer_hparams=optimizer_hparams,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            monitor_metric=args.monitor_metric,
            n_en_values=args.n_en_values,
            en_limits=en_limits
        )
        
        trainer = Trainer(
            max_epochs=2000,
            accelerator=args.accelerator,
            strategy="ddp_spawn",
            devices=50,
            precision="64",
            callbacks=[
                checkpoint_callback,
                early_stopping_callback,
                res_visualizer_callback,
                LearningRateMonitor(),
            ],
            logger=logger,
            enable_checkpointing=True,
        )
    else:  # tede
        model = TEDE(
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
        
        kl_div = GeneralizedKLDivLoss(
            log_input=False, log_target=False, reduction='batchmean')
        
        model_lightning_training = TEDELightningTraining(
            model=model,
            loss_function=kl_div,
            val_metric_functions=val_metric_functions,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            optimizer_hparams=optimizer_hparams,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            bins_centers=kNPE_bins_centers,
            monitor_metric=args.monitor_metric
        )
        
        trainer = Trainer(
            max_epochs=2000,
            accelerator=args.accelerator,
            devices="auto",
            precision="64",
            callbacks=[
                checkpoint_callback,
                early_stopping_callback,
                res_visualizer_callback,
                LearningRateMonitor(),
            ],
            logger=logger,
            enable_checkpointing=True,
        )

    # Train the model
    trainer.fit(
        model_lightning_training,
        train_dataloaders=train_loader,
        val_dataloaders=[val1_loader, val2_loader]
    )

    callbacks_dict = torch.load(
        checkpoint_callback.best_model_path,
        map_location="cpu"
    )['callbacks']

    for key in callbacks_dict.keys():
        if "ModelCheckpoint" in key:
            model_checkpoint_key = key
        elif "EarlyStopping" in key:
            early_stopping_key = key
    best_model_score = callbacks_dict[model_checkpoint_key]["best_model_score"].item()
    stopped_epoch = callbacks_dict[early_stopping_key]["stopped_epoch"]

    # Save some results info
    results_info = {
        'best_model_score': best_model_score,
        'stopped_epoch': stopped_epoch
    }

    results_info_filepath = os.path.join(
        path_to_training_results, f'results_info.json')
    with open(results_info_filepath, 'w') as f:
        json.dump(results_info, f, indent=6)

    # Load best model and save it
    if approach_type == 'nfde':
        best_model = NFDELightningTraining.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path,
            model=model,
            loss_function=args.loss_function,
            val_metric_functions=val_metric_functions,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            optimizer_hparams=optimizer_hparams,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            monitor_metric=args.monitor_metric,
            n_en_values=args.n_en_values,
            en_limits=en_limits
        )
    else:  # tede
        best_model = TEDELightningTraining.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path,
            model=model,
            loss_function=kl_div,
            val_metric_functions=val_metric_functions,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            optimizer_hparams=optimizer_hparams,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            bins_centers=kNPE_bins_centers,
            monitor_metric=args.monitor_metric
        )

    # Save the best model
    torch.save(
        best_model.model.state_dict(),
        f"{base_path_to_models}/models/{approach_type}_model.pth"
    )

if __name__ == "__main__":
    main()
