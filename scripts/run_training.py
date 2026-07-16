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
import glob
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

class ResumeFreshEarlyStopping(EarlyStopping):
    """EarlyStopping whose patience window restarts on every resume.

    Plain Lightning restore brings back wait_count (and patience) from the
    checkpoint, so a run interrupted mid-plateau resumes with most of its
    window already consumed and dies ~patience epochs later at a premature
    best. Here only best_score is kept from the checkpoint -- stopping still
    requires beating the all-time best -- while the window restarts and
    patience always stays at the configured study value (never the
    checkpoint's).
    """
    def load_state_dict(self, state_dict):
        self.best_score = state_dict.get("best_score", self.best_score)
        self.wait_count = 0
        self.stopped_epoch = 0


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
    if getattr(args, "resume", False):
        # Fixed dirpath (not the versioned logger dir) so last.ckpt is at a
        # stable path across restarts; save_last=True writes it every epoch.
        # enable_version_counter=False keeps overwriting last.ckpt in place
        # instead of switching to last-v1.ckpt when a resume finds one there.
        mc_kwargs = dict(
            dirpath=os.path.join(path_to_training_results, "checkpoints"),
            save_top_k=1, monitor=monitor_metric, mode="min", save_last=True)
        try:
            checkpoint_callback = ModelCheckpoint(
                enable_version_counter=False, **mc_kwargs)
        except TypeError:  # lightning too old for enable_version_counter
            checkpoint_callback = ModelCheckpoint(**mc_kwargs)
    else:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, monitor=monitor_metric, mode="min")

    early_stopping_callback = ResumeFreshEarlyStopping(
        monitor=monitor_metric,
        mode="min",
        patience=200 if approach_type == 'tede' else 100
    )

    # --plot_every 0 (or negative) disables the visualizer entirely: no
    # intermediate plots/predictions are produced. 
    # Any positive value keeps the usual behaviour.
    if args.plot_every > 0:
        model_res_visualizator = res_visualizator_setup(
            data_configs, plot_every_n_train_epochs=args.plot_every)

        res_visualizer_callback = ModelResultsVisualizerCallback(
            res_visualizer=model_res_visualizator,
            approach_type=approach_type,
            base_path_to_savings=path_to_training_results,
            plots_dir_name='plots',
            predictions_dir_name='predictions',
            values_to_plot_dir_name='values_to_plot',
            val_metric_names=list(val_metric_functions.keys())
        )
    else:
        res_visualizer_callback = None

    logger = CSVLogger(
        save_dir=path_to_training_results,
        name="training_logs"
    )

    return (optimizer, optimizer_hparams, lr_scheduler, val_metric_functions,
            checkpoint_callback, early_stopping_callback, res_visualizer_callback, logger)

def create_dataloaders(approach_type, path_to_processed_data, batch_size, val_batch_size=None, bin_size=None, use_pin_memory=True):
    """Create data loaders for training and validation.

    Args:
        approach_type (str): Type of approach ('nfde' or 'tede')
        path_to_processed_data (str): Path to the processed dataset
        batch_size (int): Batch size for training
        val_batch_size (int, optional): Batch size for validation
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
        pin_memory=use_pin_memory,
    )

    val_batch_size = val_batch_size if val_batch_size is not None else batch_size * 16

    val1_loader = DataLoader(
        val1_data,
        batch_size=val1_data.__len__() if approach_type == 'tede' else val_batch_size,
        shuffle=False,
        num_workers=20 if approach_type == 'tede' else 0,
        pin_memory=use_pin_memory,
    )

    val2_loader = DataLoader(
        val2_data,
        batch_size=val2_data.__len__() if approach_type == 'tede' else val_batch_size,
        shuffle=False,
        num_workers=20 if approach_type == 'tede' else 0,
        pin_memory=use_pin_memory,
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
    parser.add_argument('--processed_data_dir', type=str, default=None,
                      help='Override path to processed data')
    parser.add_argument('--results_dir', type=str, default=None,
                      help='Override path to save training results')
    parser.add_argument('--cpu_devices', type=int, default=50,
                      help='Number of DDP processes for NFDE CPU training '
                           '(must match the cores allocated to the job)')
    approach_args, _ = parser.parse_known_args()
    approach_type = approach_args.approach_type
    
    # Set up paths and configurations
    base_path_to_models = data_configs['base_path_to_models']
    
    if getattr(approach_args, 'processed_data_dir', None):
        path_to_processed_data = approach_args.processed_data_dir
        # Update data_configs so the visualizer uses the correct dataset
        data_configs['path_to_processed_data'] = path_to_processed_data
    else:
        path_to_processed_data = data_configs['path_to_processed_data']
        
    if approach_args.results_dir:
        path_to_training_results = approach_args.results_dir
    else:
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

    if getattr(args, 'test_mode', False):
        path_to_training_results += '_test'

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
        getattr(args, 'val_batch_size', None),
        bin_size if approach_type == 'tede' else None,
        use_pin_memory=(args.accelerator != 'cpu')
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
            #n_spline_bins=getattr(args, 'n_spline_bins', 8),
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
            max_epochs=10000,
            accelerator=args.accelerator,
            strategy="ddp_spawn" if args.accelerator == "cpu" else "auto",
            devices=approach_args.cpu_devices if args.accelerator == "cpu" else "auto",
            precision="64",
            callbacks=[cb for cb in [
                checkpoint_callback,
                early_stopping_callback,
                res_visualizer_callback,
                LearningRateMonitor(),
            ] if cb is not None],
            logger=logger,
            enable_checkpointing=True,
        )
    else:  # tede
        model = TEDE(
            n_sources=args.n_sources,
            params_dim=args.params_dim,
            output_dim=args.output_dim,
            d_model=args.d_model,
            activation=args.activation_function,
            n_tokens_per_param=args.n_tokens_per_param,
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
            max_epochs=10000,
            accelerator=args.accelerator,
            devices="auto",
            precision="64",
            callbacks=[cb for cb in [
                checkpoint_callback,
                early_stopping_callback,
                res_visualizer_callback,
                LearningRateMonitor(),
            ] if cb is not None],
            logger=logger,
            enable_checkpointing=True,
        )

    # Train the model. With --resume, continue from a previous run's last.ckpt
    # if one exists (e.g. after a scheduler eviction); Lightning restores model,
    # optimizer, scheduler, epoch and RNG, so the trajectory matches an
    # uninterrupted run. Fresh runs (no last.ckpt) start normally.
    resume_ckpt = None
    if getattr(args, "resume", False):
        # Newest of last*.ckpt: interrupted runs from before
        # enable_version_counter=False may have left the freshest state in
        # last-v1.ckpt rather than last.ckpt.
        _lasts = glob.glob(
            os.path.join(path_to_training_results, "checkpoints", "last*.ckpt"))
        if _lasts:
            resume_ckpt = max(_lasts, key=os.path.getmtime)
            print(f"[resume] continuing from {resume_ckpt}", flush=True)
    trainer.fit(
        model_lightning_training,
        train_dataloaders=train_loader,
        val_dataloaders=[val1_loader, val2_loader],
        ckpt_path=resume_ckpt
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

    if getattr(args, "model_save_path", ""):
        model_save_path = args.model_save_path
        os.makedirs(os.path.dirname(model_save_path) or '.', exist_ok=True)
    elif "tdata_size_check" in path_to_training_results:
        parts = path_to_training_results.rstrip('/').split('/')
        model_name_prefix = f"{parts[-2]}_{parts[-1]}"
        model_save_dir = f"{base_path_to_models}/models/tdata_size_check"
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f"{model_name_prefix}_model.pth")
    else:
        model_save_path = f"{base_path_to_models}/models/{approach_type}_model.pth"

    torch.save(
        best_model.model.state_dict(),
        model_save_path
    )

if __name__ == "__main__":
    main()
