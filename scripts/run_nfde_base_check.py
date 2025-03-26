import argparse
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import matplotlib.pyplot as plt
from neuromct.configs import data_configs
from neuromct.dataset.load_data import load_processed_data
from neuromct.models.ml.models_setup import setup
from neuromct.models.ml.metrics import LpNormDistance
from neuromct.plot.matplotlib_setup import matplotlib_setup

matplotlib_setup(tick_labelsize=14, axes_labelsize=14, legend_fontsize=9)

def compute_z_pdf(npes_sample, params_sample, sts_sample, 
                  nfde, n_z_bins, z_min, z_max, z_bin_size):
    no_nan_inds = ~torch.isnan(npes_sample)
    npes_no_nan = npes_sample[no_nan_inds]
    z = nfde.forward(npes_no_nan, params_sample, sts_sample)[0].detach()
    z_counts = torch.histc(z, bins=n_z_bins, min=z_min, max=z_max)
    z_pdf = z_counts / z_counts.sum() / z_bin_size
    return z_pdf

def compute_distance_metrics(z_bin_centers, gauss_x, z_pdf, 
                             gauss_pdf, cramer_dist, wasserstein_dist, 
                             ks_dist):
    cd = cramer_dist(z_bin_centers, gauss_x, z_pdf, gauss_pdf)
    wd = wasserstein_dist(z_bin_centers, gauss_x, z_pdf, gauss_pdf)
    ksd = ks_dist(z_bin_centers, gauss_x, z_pdf, gauss_pdf)
    return cd, wd, ksd

def plot_distribution(ax, gauss_x, gauss_pdf, z_bin_edges, 
                      z_bin_centers, z_median, z_lb=None, z_ub=None, 
                      color="black", label_base="", label_model="", fill=False):
    z_alpha = 0.3 if fill else 1.0
    linestyle = '-' if fill else '--'
    ax.plot(gauss_x, gauss_pdf, color="black", 
            alpha=0.6, linewidth=1.75, label=label_base)
    ax.stairs(z_median, z_bin_edges, color=color, fill=fill,
              alpha=z_alpha, linewidth=1.75, linestyle=linestyle,
              label=label_model)
    if z_lb is not None and z_ub is not None:
        ax.fill_between(z_bin_centers, z_lb, z_ub, fc=color, 
                        alpha=0.25, step='mid')

def add_text_annotations(ax, source_text, source_text_corrs, 
                         metrics_text, metrics_text_corrs):
    ax.text(
        source_text_corrs['x'], 
        source_text_corrs['y'], 
        source_text,
        color="black",
        bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", 
                  facecolor="white"),
        fontsize=12,
        ha="center",
        va="center",
        transform=ax.transAxes
    )
    ax.text(
        metrics_text_corrs['x'],
        metrics_text_corrs['y'],
        metrics_text,
        color="black",
        bbox=dict(boxstyle="round, pad=0.4", edgecolor="grey", 
                  facecolor="white", alpha=0.4),
        fontsize=12,
        ha="center",
        va="center",
        transform=ax.transAxes
    )

def format_metrics_text(median, lb, ub, label):
    return rf"${label}: {median:.4f}_{{{lb - median:.4f}}}^{{+{ub - median:.4f}}}$"

parser = argparse.ArgumentParser(
    description="Compare NFDE's output z with the base distribution.")
parser.add_argument("--dataset_type", type=str, default="", 
                    help='Dataset type: val1/val2')
parser.add_argument("--device", type=str, default="cpu", 
                    help='Device: cpu/cuda. Default: cpu')
parser.add_argument("--base_path_to_plots", type=str, default="./", 
                    help="Base path to save the plots. Default: './'")
args = parser.parse_args()

base_path_to_models = data_configs['base_path_to_models']
path_to_processed_data = data_configs['path_to_processed_data']

n_sources = data_configs['n_sources']
sources_names_to_vis = data_configs['sources_names_to_vis']
sources = data_configs['sources']
colors = data_configs['sources_colors_to_vis']

params_dim = data_configs['params_dim']
approach_type = 'nfde'

nfde = setup(approach_type, args.device, base_path_to_models=base_path_to_models)
cramer_dist = LpNormDistance(p=2, reduction="none")
wasserstein_dist = LpNormDistance(p=1, reduction="none")
ks_dist = LpNormDistance(p=torch.inf, reduction="none")

z_min, z_max = -5, 5
gauss_dist = torch.distributions.Normal(0, 1)  # standard normal
gauss_x = torch.linspace(z_min, z_max, 10000)
gauss_pdf = torch.exp(gauss_dist.log_prob(gauss_x))

n_z_bins = 500
z_bin_edges = torch.linspace(z_min, z_max, n_z_bins + 1)
z_bin_centers = (z_bin_edges[1:] + z_bin_edges[:-1]) / 2
z_bin_size = z_bin_edges[1] - z_bin_edges[0]

if args.dataset_type == 'val1':
    npes, params, sts = load_processed_data(
        args.dataset_type, path_to_processed_data, approach_type)
    max_n_events = npes.shape[1]
    n_points = npes.shape[0] // n_sources

    npes = npes.reshape(n_sources, -1, max_n_events)
    params = params.reshape(n_sources, -1, params_dim)
    sts = sts.reshape(n_sources, -1, 1)

    z_per_source = []
    for i in tqdm(range(n_sources), desc="Processing sources"): 
        z_per_param_set = []
        for j in tqdm(range(npes.shape[1]), desc="Processing param sets", leave=False):
            z_pdf = compute_z_pdf(npes[i][j], params[i][j], sts[i][j],
                                    nfde, n_z_bins, z_min, z_max, z_bin_size)
            z_per_param_set.append(z_pdf)
        z_per_source.append(torch.stack(z_per_param_set))
    z_per_source = torch.stack(z_per_source)

    # Compute quantiles (median, upper and lower bounds) of the modeled PDFs
    z_median_per_source = torch.quantile(z_per_source, q=0.5, dim=1)
    z_ub_per_source = torch.quantile(z_per_source, q=0.975, dim=1)
    z_lb_per_source = torch.quantile(z_per_source, q=0.025, dim=1)

    # Prepare repeated tensors for metric calculation
    z_bin_centers_rep = z_bin_centers.repeat(n_points, 1)
    gauss_x_rep = gauss_x.repeat(n_points, 1)
    gauss_pdf_rep = gauss_pdf.repeat(n_points, 1)

    # Compute metrics per source
    cd_per_source, wd_per_source, ksd_per_source = [], [], []
    for i in range(n_sources): 
        cd, wd, ksd = compute_distance_metrics(z_bin_centers_rep, gauss_x_rep,
                                               z_per_source[i], gauss_pdf_rep,
                                               cramer_dist, wasserstein_dist, ks_dist)
        cd_per_source.append(cd)
        wd_per_source.append(wd)
        ksd_per_source.append(ksd)
    cd_per_source = torch.stack(cd_per_source)
    wd_per_source = torch.stack(wd_per_source)
    ksd_per_source = torch.stack(ksd_per_source)

    # Compute quantiles for metrics
    cd_median = torch.quantile(cd_per_source, q=0.5, dim=1)
    cd_ub = torch.quantile(cd_per_source, q=0.975, dim=1)
    cd_lb = torch.quantile(cd_per_source, q=0.025, dim=1)

    wd_median = torch.quantile(wd_per_source, q=0.5, dim=1)
    wd_ub = torch.quantile(wd_per_source, q=0.975, dim=1)
    wd_lb = torch.quantile(wd_per_source, q=0.025, dim=1)

    ksd_median = torch.quantile(ksd_per_source, q=0.5, dim=1)
    ksd_ub = torch.quantile(ksd_per_source, q=0.975, dim=1)
    ksd_lb = torch.quantile(ksd_per_source, q=0.025, dim=1)

    # Plot per-source distributions
    for i in range(n_sources): 
        metrics_text = "\n".join([
            format_metrics_text(wd_median[i], wd_lb[i], wd_ub[i], r"\overline{d}_{1}^{V_1}"),
            format_metrics_text(cd_median[i], cd_lb[i], cd_ub[i], r"\overline{d}_{2}^{V_1}"),
            format_metrics_text(ksd_median[i], ksd_lb[i], ksd_ub[i], r"\overline{d}_{\infty}^{V_1}")
        ])

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        plot_distribution(
            ax, gauss_x, gauss_pdf, z_bin_edges, z_bin_centers,
            z_median_per_source[i], z_lb_per_source[i], z_ub_per_source[i], colors[i],
            label_base="Base distribution $z$: $\mathcal{N}(0, 1)$",
            label_model="NFDE's modeled $\hat{z}_{V_1}$: median + 95% CI"
        )
        add_text_annotations(ax, 
            f"Source: {sources_names_to_vis[i]}", 
            {"x": 0.175, "y": 0.775},
            metrics_text,
            {"x": 0.84, "y": 0.87},
        )
        ax.set_xlim(z_min, z_max)
        ax.set_ylim(0.0, 0.65)
        ax.legend(fontsize=12, loc="upper left")
        ax.set_xlabel("z", fontsize=16, color='black')
        ax.set_ylabel("Density", fontsize=16, color='black')
        fig.tight_layout()
        fig.savefig(f"{args.base_path_to_plots}/plots/nfde/base_check/{args.dataset_type}_{sources[i]}_nfde_z_vs_true_base.pdf")
        plt.close(fig)

    # Combined sources plot
    n_points_combined = n_points * n_sources
    z_sources_comb = z_per_source.reshape(n_points_combined, n_z_bins)
    z_median_sources_comb = torch.quantile(z_sources_comb, q=0.5, dim=0)
    z_ub_sources_comb = torch.quantile(z_sources_comb, q=0.975, dim=0)
    z_lb_sources_comb = torch.quantile(z_sources_comb, q=0.025, dim=0)

    wd_sources_comb = wd_per_source.flatten()
    cd_sources_comb = cd_per_source.flatten()
    ksd_sources_comb = ksd_per_source.flatten()

    cd_median_sources_comb = torch.quantile(cd_sources_comb, q=0.5, dim=0)
    cd_ub_sources_comb = torch.quantile(cd_sources_comb, q=0.975, dim=0)
    cd_lb_sources_comb = torch.quantile(cd_sources_comb, q=0.025, dim=0)

    wd_median_sources_comb = torch.quantile(wd_sources_comb, q=0.5, dim=0)
    wd_ub_sources_comb = torch.quantile(wd_sources_comb, q=0.975, dim=0)
    wd_lb_sources_comb = torch.quantile(wd_sources_comb, q=0.025, dim=0)

    ksd_median_sources_comb = torch.quantile(ksd_sources_comb, q=0.5, dim=0)
    ksd_ub_sources_comb = torch.quantile(ksd_sources_comb, q=0.975, dim=0)
    ksd_lb_sources_comb = torch.quantile(ksd_sources_comb, q=0.025, dim=0)

    metrics_text_combined = "\n".join([
        format_metrics_text(wd_median_sources_comb, wd_lb_sources_comb, 
                            wd_ub_sources_comb, r"\overline{d}_{1}^{V_1}"),
        format_metrics_text(cd_median_sources_comb, cd_lb_sources_comb, 
                            cd_ub_sources_comb, r"\overline{d}_{2}^{V_1}"),
        format_metrics_text(ksd_median_sources_comb, ksd_lb_sources_comb, 
                            ksd_ub_sources_comb, r"\overline{d}_{\infty}^{V_1}")
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_distribution(
        ax, gauss_x, gauss_pdf, z_bin_edges, z_bin_centers,
        z_median_sources_comb, z_lb_sources_comb, z_ub_sources_comb, '#3971ac',
        label_base="Base distribution $z$: $\mathcal{N}(0, 1)$",
        label_model="NFDE's modeled $\hat{z}_{V_1}$: median + 95% CI"
    )
    add_text_annotations(ax,
        "All sources combined", 
        {"x": 0.2, "y": 0.775},
        metrics_text_combined,
        {"x": 0.84, "y": 0.87},
    )

    ax.set_xlim(z_min, z_max)
    ax.set_ylim(0.0, 0.65)
    ax.legend(fontsize=12, loc="upper left")
    ax.set_xlabel("z", fontsize=16)
    ax.set_ylabel("Density", fontsize=16, color='black')
    fig.tight_layout()
    fig.savefig(f"{args.base_path_to_plots}/plots/nfde/base_check/{args.dataset_type}_all_sources_nfde_z_vs_true_base.pdf")
    plt.close(fig)

elif args.dataset_type == 'val2':
    npes_list, params_list, sts_list = [], [], []
    for i in range(3):
        npes_i, params_i, sts_i = load_processed_data(
            f"{args.dataset_type}_{i+1}", path_to_processed_data, approach_type)
        npes_list.append(npes_i)
        params_list.append(params_i)
        sts_list.append(sts_i)

    # Process each dataset and each source
    z_per_dataset = []
    for i in tqdm(range(3), desc="Processing datasets for val2"):
        z_per_source_i = []
        for j in tqdm(range(n_sources), desc="Processing sources", leave=False):
            z_pdf = compute_z_pdf(npes_list[i][j], params_list[i][j], sts_list[i][j],
                                  nfde, n_z_bins, z_min, z_max, z_bin_size)
            z_per_source_i.append(z_pdf)
        z_per_dataset.append(torch.stack(z_per_source_i))
    z_per_dataset = torch.stack(z_per_dataset)

    # Compute metrics per dataset and per source
    cd_list, wd_list, ksd_list = [], [], []
    for i in range(3):
        cd_source, wd_source, ksd_source = [], [], []
        for j in range(n_sources):
            cd_val = cramer_dist(
                z_bin_centers.unsqueeze(0), gauss_x.unsqueeze(0),
                z_per_dataset[i][j].unsqueeze(0), gauss_pdf.unsqueeze(0)
            )[0]
            wd_val = wasserstein_dist(
                z_bin_centers.unsqueeze(0), gauss_x.unsqueeze(0),
                z_per_dataset[i][j].unsqueeze(0), gauss_pdf.unsqueeze(0)
            )[0]
            ksd_val = ks_dist(
                z_bin_centers.unsqueeze(0), gauss_x.unsqueeze(0),
                z_per_dataset[i][j].unsqueeze(0), gauss_pdf.unsqueeze(0)
            )[0]
            cd_source.append(cd_val)
            wd_source.append(wd_val)
            ksd_source.append(ksd_val)
        cd_list.append(torch.stack(cd_source))
        wd_list.append(torch.stack(wd_source))
        ksd_list.append(torch.stack(ksd_source))
    cd_tensor = torch.stack(cd_list)
    wd_tensor = torch.stack(wd_list)
    ksd_tensor = torch.stack(ksd_list)

    # Compute combined (mean) metrics across datasets
    cd_scomb = torch.mean(cd_tensor, dim=1)
    wd_scomb = torch.mean(wd_tensor, dim=1)
    ksd_scomb = torch.mean(ksd_tensor, dim=1)
    z_per_source_scomb = torch.mean(z_per_dataset, dim=1)

    # Plot distributions
    for i in range(n_sources): 
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        for k in range(3):
            metrics_text = "\n".join([
                rf"${{d}}_{{1}}^{{V_{{2.{k+1}}}}}: {wd_tensor[k][i]:.4f}$",
                rf"${{d}}_{{2}}^{{V_{{2.{k+1}}}}}: {cd_tensor[k][i]:.4f}$",
                rf"${{d}}_{{\infty}}^{{V_{{2.{k+1}}}}}: {ksd_tensor[k][i]:.4f}$"
            ])

            plot_distribution(
                axs[k],
                gauss_x,
                gauss_pdf,
                z_bin_edges,
                z_bin_centers,
                z_per_dataset[k][i],
                None, None,
                colors[i],
                label_base="Base distribution $z$: $\mathcal{N}(0, 1)$",
                label_model=f"NFDE's modeled $\hat{{z}}_{{V_{{2.{k+1}}}}}$",
                fill=True
            )

            add_text_annotations(axs[k], 
                f"Source: {sources_names_to_vis[i]}",
                {"x": 0.225, "y": 0.775},
                metrics_text,
                {"x": 0.85, "y": 0.87},
            )
            axs[k].set_xlim(z_min, z_max)
            axs[k].set_ylim(0.0, 0.65)
            axs[k].legend(fontsize=12, loc="upper left")
            axs[k].set_xlabel("z", fontsize=16)
        axs[0].set_ylabel("Density", fontsize=16, color='black')
        fig.tight_layout()
        fig.savefig(f"{args.base_path_to_plots}/plots/nfde/base_check/{args.dataset_type}_{sources[i]}_nfde_z_vs_true_base.pdf")
        plt.close(fig)

    # Combined sources plot for val2
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for k in range(3):
        metrics_text = "\n".join([
            rf"$\overline{{d}}_{{1}}^{{V_{{2.{k+1}}}}}: {wd_scomb[k]:.4f}$",
            rf"$\overline{{d}}_{{2}}^{{V_{{2.{k+1}}}}}: {cd_scomb[k]:.4f}$",
            rf"$\overline{{d}}_{{\infty}}^{{V_{{2.{k+1}}}}}: {ksd_scomb[k]:.4f}$"
        ])
        plot_distribution(
            axs[k],
            gauss_x,
            gauss_pdf,
            z_bin_edges,
            z_bin_centers,
            z_per_source_scomb[k],
            None, None,
            "#3971ac",
            label_base="Base distribution $z$: $\mathcal{N}(0, 1)$",
            label_model=f"NFDE's modeled $\hat{{z}}_{{V_{{2.{k+1}}}}}$",
            fill=True
        )
        add_text_annotations(axs[k],
            "All sources combined",
            {"x": 0.25, "y": 0.775},
            metrics_text,
            {"x": 0.85, "y": 0.87},
        )
        axs[k].set_xlim(z_min, z_max)
        axs[k].set_ylim(0.0, 0.65)
        axs[k].legend(fontsize=12, loc="upper left")
        axs[k].set_xlabel("z", fontsize=16)
    axs[0].set_ylabel("Density", fontsize=16, color='black')
    fig.tight_layout()
    fig.savefig(f"{args.base_path_to_plots}/plots/nfde/base_check/{args.dataset_type}_all_sources_nfde_z_vs_true_base.pdf")
    plt.close(fig)
