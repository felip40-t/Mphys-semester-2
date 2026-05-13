"""Heatmap and contour rendering for the Bell-operator and concurrence grids."""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from diboson.plotting.style import (
    FIGSIZE_HEATMAP,
    FONTSIZE_LABEL,
    FONTSIZE_TICK,
    FONTSIZE_ANNOTATION,
)
from diboson.analysis.region_analysis import ProcessSpec


def plot_contour_heatmap(save_dir, spec, bell_value_grid, label, concurrence=False):
    """
    Plots contour and heatmap of the Bell operator values or Concurrence values.

    Parameters:
        save_dir (str): Directory where plots will be saved.
        spec (ProcessSpec): Process specification providing grid geometry.
        bell_value_grid (ndarray): 2D array of shape (n_cos, n_mass) — the natural
            storage convention used by main.py.
        label (str): Label appended to the output filenames.
        concurrence (bool): If True, indicates that the values are concurrence values.
    """
    bell_value_grid = bell_value_grid.T  # (n_cos, n_mass) -> (n_mass, n_cos) for meshgrid alignment
    cos_psi_grid, inv_mass_grid = spec.grid_centers()

    bell_grid_smoothed = gaussian_filter(bell_value_grid, sigma=1.0)
    plt.figure(figsize=FIGSIZE_HEATMAP)
    custom_levels = np.arange(np.round(np.min(bell_grid_smoothed), 1) - 0.1, np.round(np.max(bell_grid_smoothed), 1) + 0.2, step=0.1)
    contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, bell_grid_smoothed,
                                  levels=custom_levels, cmap='plasma')
    contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, bell_grid_smoothed,
                                levels=custom_levels, colors='black', linewidths=0.7)
    plt.clabel(contour_lines, inline=True, fontsize=FONTSIZE_ANNOTATION, fmt="%.2f")
    if concurrence:
        colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{C}_{LB}$', orientation='vertical')
    else:
        colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{I}_3$', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(FONTSIZE_LABEL)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=FONTSIZE_LABEL)
    plt.ylabel(rf'$M_{{{spec.name}}} (GeV)$', fontsize=FONTSIZE_LABEL)
    plt.yticks(np.arange(spec.mass_min + spec.mass_width, spec.mass_max, spec.mass_width), fontsize=FONTSIZE_TICK)
    plt.xticks(np.arange(spec.cos_min + spec.cos_width, spec.cos_max, spec.cos_width), fontsize=FONTSIZE_TICK)
    plt.tight_layout()
    if concurrence:
        name = f"concurrence_contour_{spec.name}_{label}.pdf"
    else:
        name = f"bell_operator_contour_{spec.name}_{label}.pdf"
    plot_filename = os.path.join(save_dir, name)
    plt.savefig(plot_filename)
    plot_filename = os.path.join(save_dir, name.replace('.pdf', '.png'))
    plt.savefig(plot_filename)
    plt.close()

    plt.figure(figsize=FIGSIZE_HEATMAP)
    plt.imshow(bell_value_grid, origin='lower',
               extent=[spec.cos_min, spec.cos_max, spec.mass_min, spec.mass_max],
               aspect='auto', cmap='plasma')
    if concurrence:
        colorbar = plt.colorbar(label=r'$\mathcal{C}_{LB}$', orientation='vertical')
    else:
        colorbar = plt.colorbar(label=r'$\mathcal{I}_3$', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(FONTSIZE_LABEL)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=FONTSIZE_LABEL)
    plt.ylabel(rf'$M_{{{spec.name}}} (GeV)$', fontsize=FONTSIZE_LABEL)
    plt.yticks(np.arange(spec.mass_min, spec.mass_max + 1, spec.mass_width), fontsize=FONTSIZE_TICK)
    plt.xticks(np.arange(spec.cos_min, spec.cos_max + 0.001, spec.cos_width), fontsize=FONTSIZE_TICK)

    x_centers = cos_psi_grid[0, :]
    y_centers = inv_mass_grid[:, 0]
    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            plt.text(x, y, f"{bell_value_grid[i, j]:.2f}", color="white",
                     ha="center", va="center", fontsize=FONTSIZE_ANNOTATION)
    plt.tight_layout()

    if concurrence:
        name = f"concurrence_heatmap_{spec.name}_{label}.pdf"
    else:
        name = f"bell_operator_heatmap_{spec.name}_{label}.pdf"

    heatmap_filename = os.path.join(save_dir, name)
    plt.savefig(heatmap_filename)
    heatmap_filename = os.path.join(save_dir, name.replace('.pdf', '.png'))
    plt.savefig(heatmap_filename)
    plt.close()

def generate_unphysicality_heatmap(spec, save_dir, data=None):
    cos_psi_grid, inv_mass_grid = spec.grid_centers()
    x_centers = cos_psi_grid[0, :]
    y_centers = inv_mass_grid[:, 0]

    unphysicality_grid = data if data is not None else np.zeros((spec.n_cos_bins, spec.n_mass_bins))
    unphysicality_grid = unphysicality_grid.T  # (n_cos, n_mass) -> (n_mass, n_cos) for imshow alignment

    plt.figure(figsize=FIGSIZE_HEATMAP)
    plt.imshow(unphysicality_grid, origin='lower',
                extent=[spec.cos_min, spec.cos_max, spec.mass_min, spec.mass_max],
                aspect='auto', cmap='plasma')
    colorbar = plt.colorbar(label='Unphysicality', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(FONTSIZE_LABEL)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=FONTSIZE_LABEL)
    plt.ylabel(rf'$M_{{{spec.name}}} (GeV)$', fontsize=FONTSIZE_LABEL)
    plt.yticks(np.arange(spec.mass_min, spec.mass_max + 1, spec.mass_width), fontsize=FONTSIZE_TICK)
    plt.xticks(np.arange(spec.cos_min, spec.cos_max + 0.001, spec.cos_width), fontsize=FONTSIZE_TICK)

    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            plt.text(x, y, f"{unphysicality_grid[i, j]:.2f}", color="white",
                     ha="center", va="center", fontsize=FONTSIZE_ANNOTATION)

    plt.tight_layout()
    base = os.path.join(save_dir, f"unphysicality_heatmap_{spec.name}")
    plt.savefig(base + ".pdf")
    plt.savefig(base + ".png")
    plt.close()

    return unphysicality_grid