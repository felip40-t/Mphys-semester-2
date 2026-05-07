"""Heatmap and contour rendering for the Bell-operator and concurrence grids."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


def plot_contour_heatmap(save_dir, cos_psi_grid, inv_mass_grid, bell_value_grid, label, process_label, concurrence=False):
    """
    Plots contour and heatmap of the Bell operator values or Concurrence values.

    Parameters:
        save_dir (str): Directory where plots will be saved.
        cos_psi_grid (ndarray): 2D mesh grid of cos(theta) centers.
        inv_mass_grid (ndarray): 2D mesh grid of M_VV centers.
        bell_value_grid (ndarray): 2D array of Bell operator values or concurrence values.
        label (str): Label appended to the output filenames.
        process_label (str): Process tag used in filenames and the y-axis label, e.g. "ZZ" or "WW".
        concurrence (bool): If True, indicates that the values are concurrence values.
    """
    colors = ['darkblue', 'blue', 'purple', 'red']
    concurrence_cmap = LinearSegmentedColormap.from_list('concurrence_cmap', colors, N=256)

    bell_grid_smoothed = gaussian_filter(bell_value_grid, sigma=1.0)
    plt.figure(figsize=(12, 10))
    custom_levels = np.arange(np.round(np.min(bell_grid_smoothed), 1) - 0.1, np.round(np.max(bell_grid_smoothed), 1) + 0.2, step=0.1)
    contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, bell_grid_smoothed,
                                  levels=custom_levels, cmap=concurrence_cmap if concurrence else 'plasma')
    contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, bell_grid_smoothed,
                                levels=custom_levels, colors='black', linewidths=0.7)
    plt.clabel(contour_lines, inline=True, fontsize=12, fmt="%.2f")
    if concurrence:
        colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{C}_LB$', orientation='vertical')
    else:
        colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{I}_3$', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(16)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
    plt.ylabel(rf'$M_{{{process_label}}} (GeV)$', fontsize=16)
    plt.yticks(np.arange(300, 1200, 100), fontsize=14)
    plt.xticks(np.arange(0.1, 0.9, 0.1), fontsize=14)
    plt.tight_layout()
    if concurrence:
        name = f"concurrence_contour_{process_label}_{label}.pdf"
    else:
        name = f"bell_operator_contour_{process_label}_{label}.pdf"
    plot_filename = os.path.join(save_dir, name)
    plt.savefig(plot_filename)
    plot_filename = os.path.join(save_dir, name.replace('.pdf', '.png'))
    plt.savefig(plot_filename)
    plt.close()

    plt.figure(figsize=(12, 10))
    plt.imshow(bell_value_grid, origin='lower', extent=[0, 0.9, 200, 1200],
               aspect='auto', cmap=concurrence_cmap if concurrence else 'plasma')
    if concurrence:
        colorbar = plt.colorbar(label=r'$\mathcal{C}_{LB}$', orientation='vertical')
    else:
        colorbar = plt.colorbar(label=r'$\mathcal{I}_3$', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(16)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
    plt.ylabel(rf'$M_{{{process_label}}} (GeV)$', fontsize=16)
    plt.yticks(np.arange(200, 1201, 100), fontsize=14)
    plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=14)

    num_rows, num_cols = bell_value_grid.shape
    x_centers = np.linspace(0.05, 0.85, num_cols)
    y_centers = np.linspace(225.0, 1175.0, num_rows)
    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            plt.text(x, y, f"{bell_value_grid[i, j]:.2f}", color="white",
                     ha="center", va="center", fontsize=12)
    plt.tight_layout()

    if concurrence:
        name = f"concurrence_heatmap_{process_label}_{label}.pdf"
    else:
        name = f"bell_operator_heatmap_{process_label}_{label}.pdf"

    heatmap_filename = os.path.join(save_dir, name)
    plt.savefig(heatmap_filename)
    heatmap_filename = os.path.join(save_dir, name.replace('.pdf', '.png'))
    plt.savefig(heatmap_filename)
    plt.close()
