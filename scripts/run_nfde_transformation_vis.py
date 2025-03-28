import argparse
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import matplotlib.pyplot as plt
from neuromct.configs import data_configs
from neuromct.dataset import (
    load_processed_data,
    load_minimax_scaler
)
from neuromct.models.ml.models_setup import setup
from neuromct.plot.matplotlib_setup import matplotlib_setup

matplotlib_setup(tick_labelsize=14, axes_labelsize=14, legend_fontsize=9)

def add_text_annotations_for_transformation(ax, source_text, source_text_corrs):
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

def plot_initial_spectrum(ax, 
        x, x_min=0.0, x_max=20.0, 
        color="black", label="", 
        loc='upper right',
        legend_size=14,
        linestyle='-',
        ylabel_size=18,
        yticks_size=14,
        ylim=(2.5e-4, 25)
    ):
    x_counts, x_bins = torch.histogram(x, bins=800, range=(x_min, x_max))
    x_pdf = x_counts / (x_counts.sum() * (x_bins[1] - x_bins[0]))
    ax.stairs(x_pdf, x_bins, color=color, alpha=0.8, label=label, linestyle=linestyle)
    ax.legend(fontsize=legend_size, loc=loc)
    ax.set_xlabel("Number of photo-electrons: " + r"$N_{p.e.} \ / \ 10^3$", fontsize=16)
    ax.set_ylabel("Prob. density", fontsize=ylabel_size)
    ax.set_yscale('log')
    ax.yaxis.set_tick_params(labelsize=yticks_size)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ylim[0], ylim[1])

def plot_transformation(ax,
        x, z, 
        x_min=0.0, x_max=20.0, 
        z_min=-5.0, z_max=5.0, 
        color="black", 
        label="", 
        linestyle='-', 
        loc='lower right',
        ylabel_size=16,
        yticks_size=14
    ):
    step = x.shape[0] // 10000
    step = 1 if step == 0 else step
    ax.plot(x[::step], z[::step], color=color, label=label, linestyle=linestyle, alpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.legend(fontsize=14, loc=loc)
    ax.set_xlabel("Number of photo-electrons: " + r"$N_{p.e.} \ / \ 10^3$", fontsize=16)
    ax.set_ylabel("z", fontsize=ylabel_size)
    ax.yaxis.set_tick_params(labelsize=yticks_size)

def plot_z_distribution(ax, 
        z, z_min=-5.0, z_max=5.0, ymax=0.55, color="black", 
        ylabel="Prob. density",
        label_z="", label_gauss="Base distribution: $\mathcal{N}(0, 1)$",
        legend_size=14, ylabel_size=18, yticks_size=14, legend=True, yticks=None):
    z_counts, z_bins = torch.histogram(z, bins=800, range=(z_min, z_max))
    z_pdf = z_counts / (z_counts.sum() * (z_bins[1] - z_bins[0]))

    ax.plot(gauss_x, gauss_pdf, color="black", alpha=0.6, linewidth=1.75, label=label_gauss)
    ax.stairs(z_pdf, z_bins, color=color, alpha=0.3, label=label_z, fill=True)
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(0.0, ymax)
    if legend:
        ax.legend(fontsize=legend_size)
    ax.set_xlabel("z", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=ylabel_size)
    ax.yaxis.set_tick_params(labelsize=yticks_size)
    if yticks is not None:
        ax.set_yticks(yticks)

def compute_transformation(npes_sample, params_sample, sts_sample, nfde):
    no_nan_inds = ~torch.isnan(npes_sample)
    npes_no_nan = npes_sample[no_nan_inds]
    npes_no_nan_sorted, _ = torch.sort(npes_no_nan)
    z = nfde.forward(npes_no_nan_sorted, params_sample, sts_sample)[0].detach()
    return npes_no_nan_sorted, z

parser = argparse.ArgumentParser(
    description="Visualize the transformation of NFDE.")
parser.add_argument("--device", type=str, default="cpu", 
                    help='Device: cpu/cuda. Default: cpu')
parser.add_argument("--base_path_to_plots", type=str, default="./", 
                    help="Base path to save the plots. Default: './'")
args = parser.parse_args()

device = args.device
base_path_to_plots = args.base_path_to_plots
base_path_to_models = data_configs['base_path_to_models']
path_to_processed_data = data_configs['path_to_processed_data']
scaler = load_minimax_scaler(base_path_to_models)

n_sources = data_configs['n_sources']
sources_names_to_vis = data_configs['sources_names_to_vis']
sources = data_configs['sources']
colors = data_configs['sources_colors_to_vis']

params_dim = data_configs['params_dim']
approach_type = 'nfde'

nfde = setup(approach_type, device, base_path_to_models=base_path_to_models)

z_min, z_max = -5, 5
gauss_dist = torch.distributions.Normal(0, 1)  # standard normal
gauss_x = torch.linspace(z_min, z_max, 10000)
gauss_pdf = torch.exp(gauss_dist.log_prob(gauss_x))

npes_list, params_list, sts_list = [], [], []
for i in range(3):
    npes_i, params_i, sts_i = load_processed_data(
        f"val2_{i+1}", path_to_processed_data, approach_type)
    npes_list.append(npes_i)
    params_list.append(params_i)
    sts_list.append(sts_i)

x_per_source_val2 = []
z_per_source_val2 = []
for i in tqdm(range(3)):
    x_per_source_val2_i = []
    z_per_source_val2_i = []
    for j in tqdm(range(n_sources), leave=False):
        x, z = compute_transformation(npes_list[i][j], params_list[i][j], sts_list[i][j], nfde)
        x_per_source_val2_i.append(x)
        z_per_source_val2_i.append(z)
    x_per_source_val2.append(x_per_source_val2_i)
    z_per_source_val2.append(z_per_source_val2_i)

for i in range(3):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for j in range(n_sources):
        plot_transformation(ax, 
            x_per_source_val2[i][j], 
            z_per_source_val2[i][j], 
            0.0, 16.0, 
            -5.0, 5.0, 
            colors[j], 
            sources_names_to_vis[j]
        )
    params_transformed = scaler.inverse_transform(params_list[i])
    ax.set_title(r"$k_{B}$"+f": {params_transformed[0][0]:.2f} [g/cm2/GeV], " + 
                r"$f_{C}$"+f": {params_transformed[0][1]:.3f}, " + 
                r"$Y$"+f": {params_transformed[0][2]:.0f} [1/MeV]", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{base_path_to_plots}/plots/nfde/transform_vis/transforms_only/val2_{i+1}_transformations.pdf")
    fig.savefig(f"{base_path_to_plots}/plots/nfde/transform_vis/transforms_only/val2_{i+1}_transformations.png")
    plt.close(fig)

# Create a single plot with all val2 datasets together
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
line_styles = ['-', '--', ':']  # Different line styles for different val2 datasets

for i in range(3):
    for j in range(n_sources):
        plot_transformation(ax, 
            x_per_source_val2[i][j], 
            z_per_source_val2[i][j], 
            0.0, 16.0, 
            -5.0, 5.0, 
            colors[j],
            f"{sources_names_to_vis[j]} (V2.{i+1})",
            linestyle=line_styles[i]
        )

# Clear the existing legend
ax.get_legend().remove()

# Create two separate legends
# First legend for sources (solid lines)
source_lines = []
source_labels = []
for j in range(n_sources):
    line = ax.plot([], [], color=colors[j], linestyle='-', label=sources_names_to_vis[j])
    source_lines.append(line[0])
    source_labels.append(sources_names_to_vis[j])

# Second legend for val2 datasets (dashed lines)
val2_lines = []
val2_labels = []
for i in range(3):
    params_transformed = scaler.inverse_transform(params_list[i])
    add_space = True if i == 0 else False
    if add_space:
        params_names = r"$k_{B}$"+ f":   {params_transformed[0][0]:.2f}, " + \
                    r"$f_{C}$"+f": {params_transformed[0][1]:.3f}, " + \
                    r"$Y$"+f": {params_transformed[0][2]:.0f}"
    else:
        params_names = r"$k_{B}$"+ f": {params_transformed[0][0]:.2f}, " + \
                    r"$f_{C}$"+f": {params_transformed[0][1]:.3f}, " + \
                    r"$Y$"+f": {params_transformed[0][2]:.0f}"
    line = ax.plot([], [], color='black', linestyle=line_styles[i], label=f'V2.{i+1}')
    val2_lines.append(line[0])
    val2_labels.append(f'V2.{i+1}: {params_names}')

# Add both legends
legend1 = ax.legend(handles=source_lines, labels=source_labels, 
                   loc=(0.7, 0.275), fontsize=13)
ax.add_artist(legend1)
ax.legend(handles=val2_lines, labels=val2_labels, 
          loc='lower right', fontsize=13)

fig.tight_layout()
fig.savefig(f"{base_path_to_plots}/plots/nfde/transform_vis/transforms_only/val2_all_transformation.pdf")
fig.savefig(f"{base_path_to_plots}/plots/nfde/transform_vis/transforms_only/val2_all_transformation.png")
plt.close(fig)

x_maxs = [1.5, 3.25, 5.5, 16.0, 16.0]
for i in range(3):
    for j in range(n_sources):
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))

        no_nan_inds = ~torch.isnan(npes_list[i][j])
        npes_no_nan = npes_list[i][j][no_nan_inds]
        params_transformed = scaler.inverse_transform(params_list[i])

        plot_initial_spectrum(ax[0], 
            npes_no_nan, 
            color=colors[j], 
            label=f"{sources_names_to_vis[j]} spectrum",
            x_max=x_maxs[j],
            loc='upper left'
        )

        plot_transformation(ax[1], 
            x_per_source_val2[i][j], 
            z_per_source_val2[i][j], 
            color=colors[j],
            label=f"NFDE's learned transformation",
            linestyle='-',
            x_max=x_maxs[j],
            z_max=6.0,
            loc='upper left'
        )
        
        plot_z_distribution(ax[2], 
            z_per_source_val2[i][j], 
            color=colors[j], 
            label=f"NFDE's modeled $\hat{{z}}_{{V_{{2.{i+1}}}}}$"
        )
        
        fig.suptitle(
            f"Val. dataset №2.{i+1}: " + 
            r"$k_{B}$"+f": {params_transformed[0][0]:.2f} [g/cm2/GeV], " + 
            r"$f_{C}$"+f": {params_transformed[0][1]:.3f}, " + 
            r"$Y$"+f": {params_transformed[0][2]:.0f} [1/MeV]", y=0.95,fontsize=18
        )

        fig.tight_layout()
        fig.savefig(f"{base_path_to_plots}/plots/nfde/transform_vis/norm_dir/{sources[j]}/val2_{i+1}_{sources[j]}_norm_dir.pdf")
        fig.savefig(f"{base_path_to_plots}/plots/nfde/transform_vis/norm_dir/{sources[j]}/val2_{i+1}_{sources[j]}_norm_dir.png")
        plt.close(fig)

# Create combined plots for each source showing all val2 datasets
for j in range(n_sources):
    fig = plt.figure(figsize=(16, 5))
    
    # Create grid for the plots - 2 columns for initial spectrum and transformation
    # 1 column for the 3 z-distributions
    gs = fig.add_gridspec(3, 3, width_ratios=[2, 2, 2], height_ratios=[1.25, 1, 1], hspace=0.00)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0:3, 0])  # Initial spectrum - spans all rows
    ax2 = fig.add_subplot(gs[0:3, 1])  # Transformation - spans all rows
    ax3_1 = fig.add_subplot(gs[0, 2])  # Z distribution for V2.1
    ax3_2 = fig.add_subplot(gs[1, 2], sharex=ax3_1)  # Z distribution for V2.2
    ax3_3 = fig.add_subplot(gs[2, 2], sharex=ax3_1)  # Z distribution for V2.3
    
    # Hide x-axis labels for the top two z distribution plots
    ax3_1.tick_params(labelbottom=False)
    ax3_2.tick_params(labelbottom=False)
    
    # Remove x-axis labels from the first two z distribution plots
    ax3_1.set_xlabel("")
    ax3_2.set_xlabel("")
    
    z_axes = [ax3_1, ax3_2, ax3_3]

    # Plot initial spectrum for each val2 dataset
    for i in range(3):
        no_nan_inds = ~torch.isnan(npes_list[i][j])
        npes_no_nan = npes_list[i][j][no_nan_inds]
        plot_initial_spectrum(ax1, 
            npes_no_nan, 
            color=colors[j], 
            label=f"Val2.{i+1}",
            x_max=x_maxs[j],
            loc=None,  # Don't show the default legend
            linestyle=line_styles[i],
        )

    # Clear the existing legend
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()
        
    # Create separate legends for initial spectrum
    # Legend for val2 datasets (different dash styles)
    val2_lines = []
    val2_labels = []
    for i in range(3):
        line = ax1.plot([], [], color='black', linestyle=line_styles[i], label=f'Val2.{i+1}')
        val2_lines.append(line[0])
        val2_labels.append(f'Val2.{i+1}')
    
    # Source label
    source_line = ax1.plot([], [], color=colors[j], linestyle='-', label=sources_names_to_vis[j] + " spectrum")
    
    # Add both legends to initial spectrum
    loc = (0.025, 0.66) if j < 3 else "upper right"
    legend1 = ax1.legend(handles=val2_lines, labels=val2_labels, 
                       loc=loc, fontsize=13)
    ax1.add_artist(legend1)
    ax1.legend(handles=source_line, loc='upper left', fontsize=13)

    # Plot transformation for each val2 dataset
    for i in range(3):
        plot_transformation(ax2, 
            x_per_source_val2[i][j], 
            z_per_source_val2[i][j], 
            color=colors[j],
            label=f"Val2.{i+1}",
            linestyle=line_styles[i],
            x_max=x_maxs[j],
            z_max=6.0,
            loc=None,
        )
    
    # Clear the existing legend
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()
        
    # Create separate legends for transformation
    # Legend for val2 datasets (different dash styles)
    val2_lines = []
    val2_labels = []
    for i in range(3):
        line = ax2.plot([], [], color='black', linestyle=line_styles[i], label=f'Val2.{i+1}')
        val2_lines.append(line[0])
        val2_labels.append(f'Val2.{i+1}')
    
    # Source label
    source_line = ax2.plot([], [], color=colors[j], linestyle='-', label="NFDE's learned transformation")
    
    # Add both legends to transformation
    legend2 = ax2.legend(handles=val2_lines, labels=val2_labels, 
                       loc=(0.025, 0.66), fontsize=13)
    ax2.add_artist(legend2)
    ax2.legend(handles=source_line, loc='upper left', fontsize=13)
    
    # Plot z distribution for each val2 dataset in separate subplots
    for i in range(3):
        label_gauss = "Base distribution: $\\mathcal{{N}}(0, 1)$" if i == 0 else ""
        label_z = "NFDE's modeled $\hat{z}$" if i == 0 else ""
        ymax = 0.65 * 1.25 if i == 0 else 0.65
        ylabel = "Prob. density" if i == 1 else ""
        legend = True if i == 0 else False
        params_transformed = scaler.inverse_transform(params_list[i])
        plot_z_distribution(z_axes[i], 
            z_per_source_val2[i][j], 
            color=colors[j], 
            ylabel=ylabel,
            label_z=label_z,
            label_gauss=label_gauss,
            ymax=ymax,
            legend_size=12,
            yticks_size=12,
            legend=legend
        )

        z_axes[i].text(
            0.1, 0.8, f'V2.{i+1}', 
            color="black",
            bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),
            fontsize=12,
            ha="center",
            va="center",
            transform=z_axes[i].transAxes
        )

    fig.tight_layout()
    fig.savefig(f"{base_path_to_plots}/plots/nfde/transform_vis/norm_dir/{sources[j]}/val2_combined_{sources[j]}_norm_dir.pdf")
    fig.savefig(f"{base_path_to_plots}/plots/nfde/transform_vis/norm_dir/{sources[j]}/val2_combined_{sources[j]}_norm_dir.png")
    plt.close(fig)

# Create plots showing all sources together for each val2 dataset
for i in range(3):
    fig = plt.figure(figsize=(16, 6))
    
    # Create grid for the plots - 2 columns for initial spectrum and transformation
    # 1 column for the source-specific z-distributions
    gs = fig.add_gridspec(n_sources, 3, width_ratios=[2, 2, 2], height_ratios=[1.5, 1, 1, 1, 1], hspace=0.0)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[:, 0])  # Initial spectrum - spans all rows
    ax2 = fig.add_subplot(gs[:, 1])  # Transformation - spans all rows
    
    # Create z distribution subplots for each source
    z_axes = []
    for j in range(n_sources):
        if j == 0:
            ax_z = fig.add_subplot(gs[j, 2])
        else:
            ax_z = fig.add_subplot(gs[j, 2], sharex=z_axes[0])
        z_axes.append(ax_z)
    
    # Plot initial spectrum for all sources
    for j in range(n_sources):
        no_nan_inds = ~torch.isnan(npes_list[i][j])
        npes_no_nan = npes_list[i][j][no_nan_inds]
        plot_initial_spectrum(ax1, 
            npes_no_nan, 
            color=colors[j], 
            label=sources_names_to_vis[j],
            x_max=16.0,  # Using a common x_max for all sources
            loc=None,  # Don't show the default legend
            linestyle='-',
        )
    
    # Set legend for initial spectrum
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()
    
    source_lines = []
    source_labels = []
    for j in range(n_sources):
        line = ax1.plot([], [], color=colors[j], linestyle='-', label=sources_names_to_vis[j])
        source_lines.append(line[0])
        source_labels.append(sources_names_to_vis[j])
    
    legend = ax1.legend(handles=source_lines, labels=source_labels, 
                     loc='upper right', fontsize=15)

    # Plot transformation for all sources
    for j in range(n_sources):
        plot_transformation(ax2, 
            x_per_source_val2[i][j], 
            z_per_source_val2[i][j], 
            color=colors[j],
            label=sources_names_to_vis[j],
            linestyle='-',
            x_max=16.0,
            z_max=6.0,
            loc=None
        )
    
    # Set legend for transformation
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()
    
    source_lines = []
    source_labels = []
    for j in range(n_sources):
        line = ax2.plot([], [], color=colors[j], linestyle='-', label=sources_names_to_vis[j])
        source_lines.append(line[0])
        source_labels.append(sources_names_to_vis[j])
        
    # Add source legend
    legend2 = ax2.legend(handles=source_lines, labels=source_labels, 
                     loc='lower right', fontsize=15)
    ax2.add_artist(legend2)
    
    # Add second legend for transformation type
    transform_line = ax2.plot([], [], color='black', linestyle='-', label="NFDE's learned transformation")
    ax2.legend(handles=transform_line, loc='upper left', fontsize=14)
    
    # Plot z distribution for each source in its own subplot
    for j in range(n_sources):
        # Only show x-axis label for the bottom plot
        label_gauss = "Base distribution: $\\mathcal{{N}}(0, 1)$" if j == 0 else ""
        label_z = "NFDE's modeled $\hat{z}$" if j == 0 else ""
        ymax = 0.65 * 1.5 if j == 0 else 0.65
        ylabel = "Prob. density" if j == 2 else ""
        legend = True if j == 0 else False
        text_y = 0.75 if j == 0 else 0.65
        yticks = [0.0, 0.4, 0.8] if j == 0 else [0.0, 0.4]

        plot_z_distribution(z_axes[j], 
            z_per_source_val2[i][j], 
            color=colors[j], 
            label_z=label_z,
            label_gauss=label_gauss,
            legend=legend,
            ylabel=ylabel,
            ymax=ymax,
            legend_size=12.5,
            yticks_size=12,
            yticks=yticks
        )
        
        # Add source name to each z distribution subplot
        z_axes[j].text(
            0.15, text_y, sources_names_to_vis[j], 
            color="black",
            bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),
            fontsize=13,
            ha="center",
            va="center",
            transform=z_axes[j].transAxes
        )
       
    # Add parameter information as overall title
    params_transformed = scaler.inverse_transform(params_list[i])
    fig.suptitle(
        f"Val. dataset №2.{i+1}: " + 
        r"$k_{B}$"+f": {params_transformed[0][0]:.2f} [g/cm2/GeV], " + 
        r"$f_{C}$"+f": {params_transformed[0][1]:.3f}, " + 
        r"$Y$"+f": {params_transformed[0][2]:.0f} [1/MeV]", y=0.95, fontsize=20
    )

    fig.tight_layout()
    fig.savefig(f"{base_path_to_plots}/plots/nfde/transform_vis/norm_dir/val2_{i+1}_all_sources.pdf")
    fig.savefig(f"{base_path_to_plots}/plots/nfde/transform_vis/norm_dir/val2_{i+1}_all_sources.png")
    plt.close(fig)
