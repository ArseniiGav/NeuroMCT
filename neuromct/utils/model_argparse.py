import argparse
import json
import pkg_resources

def tede_argparse():
    configs_path = pkg_resources.resource_filename("neuromct", "configs/")

    parser = argparse.ArgumentParser(description='Hyperparameters of the TEDE model')
    parser.add_argument("--config", type=str, default=f'{configs_path}/tede_configs.json',
                         help="The path to the JSON config file")
    parser.add_argument("--n_sources", type=int, default=5,
                         help='The number of calibration sources (default=5).')
    parser.add_argument("--batch_size", type=int, default=32,
                         help='The batch size (default=32).')
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help='The learning rate (default=5e-4)')
    parser.add_argument("--output_dim", type=int, default=800,
                         help='The number of bins that used to describe the spectra (default=800)')
    parser.add_argument("--activation_function", type=str, default="relu",
                         help='Activation function of the hidden layers. Either "relu" or "gelu"; (default="relu")')
    parser.add_argument("--d_model", type=int, default=128,
                         help='The number of expected features in the input of the encoder layers (default=128)')
    parser.add_argument("--nhead", type=int, default=8,
                         help='The number of heads in the multiheadattention models (default=8)')
    parser.add_argument("--num_encoder_layers", type=int,
                         help='The number of sub-encoder-layers in the encoder (default=5)')
    parser.add_argument("--dim_feedforward", type=int, default=32,
                         help='The dimension of the feedforward network model (default=32)')
    parser.add_argument("--dropout", type=float, default=0.1,
                         help='The dropout prob. value in the Transformer encoder layers (default=0.1)')
    parser.add_argument("--temperature", type=float, default=1.0,
                         help='The temperature parameter for the output Softmax layer (default=1.0)')
    parser.add_argument("--monitor_metric", type=str, default="val_cramer_metric",
                        help='''The main validation metric used during the training
                                (i.e., used for the early stopping condition). Can be val_*_metric, 
                                where * is either "cramer" or "wasserstein", or "ks". 
                                Default=val_cramer_metric.''')
    parser.add_argument("--optimizer", type=str, default="AdamW",
                         help='The optimizer to use (default="AdamW")')
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR",
                         help='The learning rate scheduler to use (default="CosineAnnealingLR")')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                         help='Weight decay for optimizer (default=1e-4)')
    parser.add_argument("--accelerator", type=str, default="gpu",
                         help='Device to use for training (default="gpu")')
    parser.add_argument("--seed", type=int, default=22,
                         help='Random seed for reproducibility (default=22)')
    args = parser.parse_args("") # "" is used to avoid errors with Ipython

    if args.config:
        with open(args.config, 'r') as f:
            config_args = json.load(f)
        parser.set_defaults(**config_args)

    final_args = parser.parse_args("") # "" is used to avoid errors with Ipython
    return final_args

def nfde_argparse():
    configs_path = pkg_resources.resource_filename("neuromct", "configs/")

    parser = argparse.ArgumentParser(description='Hyperparameters of the NFDE model')
    parser.add_argument("--config", type=str, default=f'{configs_path}/nfde_configs.json',
                         help="The path to the JSON config file")
    parser.add_argument("--n_flows", type=int, default=25,
                         help='The number of flows (default=25).')
    parser.add_argument("--batch_size", type=int, default=1,
                         help='The batch size (default=1).')
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help='The learning rate (default=5e-4)')
    parser.add_argument("--activation_function", type=str, default="relu",
                        help="""Activation function of the hidden layers.
                                Options: "relu", "gelu", "silu", "tanh"; (default="relu")""")
    parser.add_argument("--loss_function", type=str, default="kl-div",
                         help="""Loss function to train the model.
                                 Options: "kl-div", "wasserstein", or "cramer" (default="kl-div")""")
    parser.add_argument("--n_units", type=int, default=20,
                         help="The number of units in a flow's conditional network (default=20)")
    parser.add_argument("--flow_type", type=str, default="planar",
                         help='The type of the flows (default=planar)')
    parser.add_argument("--dropout", type=float, default=0.0,
                         help="The dropout prob. value in the Flow's networks (default=0.0)")
    parser.add_argument("--monitor_metric", type=str, default="val_cramer_metric",
                        help='''The main validation metric used during the training
                                (i.e., used for the early stopping condition). Can be val_*_metric, 
                                where * is either "cramer" or "wasserstein", or "ks". 
                                Default=val_cramer_metric.''')
    parser.add_argument("--n_en_values", type=int, default=100000,
                         help='''The number of energy values for log_prob estimation 
                                 during training of the NFDE model (default=100000)''')
    parser.add_argument("--optimizer", type=str, default="AdamW",
                         help='The optimizer to use (default="AdamW")')
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR",
                         help='The learning rate scheduler to use (default="CosineAnnealingLR")')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                         help='Weight decay for optimizer (default=1e-4)')
    parser.add_argument("--accelerator", type=str, default="cpu",
                         help='Device to use for training (default="cpu")')
    parser.add_argument("--seed", type=int, default=22,
                         help='Random seed for reproducibility (default=22)')
    args = parser.parse_args("") # "" is used to avoid errors with Ipython

    if args.config:
        with open(args.config, 'r') as f:
            config_args = json.load(f)
        parser.set_defaults(**config_args)

    final_args = parser.parse_args("") # "" is used to avoid errors with Ipython
    return final_args
