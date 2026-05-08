"""Per-region analysis and heatmap generation shared across ZZ and WW tracks.

The four functions here (`process_region`, `generate_event_count_heatmap`,
`generate_uniformity_heatmap`, `generate_unphysicality_heatmap`) are
parametrised by a `ProcessSpec` that captures the only differences between
the two tracks: process name, mass-bin grid shape, raw-data filenames, and
the per-process density-matrix / variance callables.
"""

import os
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import matplotlib.pyplot as plt

from diboson.plotting.style import FIGSIZE_HEATMAP, FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_ANNOTATION
from config import COS_BIN_MIN, COS_BIN_WIDTH, MASS_BIN_MIN, MASS_BIN_WIDTH
from core.density_matrix_calculator import (
    O_bell_prime1,
    project_to_psd,
    unphysicality_score,
)
from core.Bell_inequality_optimizer import (
    bell_inequality_optimization,
    optimal_bell_operator,
)
from core.concurrence_bound import concurrence_lower, check_density_matrix


@dataclass
class ProcessSpec:
    name: str
    n_mass_bins: int
    n_cos_bins: int
    mass_max: float
    eta: float
    psi_filename: str
    inv_mass_filename: str
    theta_filenames: Dict[int, str]
    phi_filenames: Dict[int, str]
    get_density_matrix: Callable
    get_variance: Callable

    def build_regions(self):
        """Return the canonical {(i,j): [cos_range, mass_range]} dict for this process."""
        return {
            (i, j): [
                (COS_BIN_MIN + COS_BIN_WIDTH * i, COS_BIN_MIN + COS_BIN_WIDTH * (i + 1)),
                (MASS_BIN_MIN + MASS_BIN_WIDTH * j, MASS_BIN_MIN + MASS_BIN_WIDTH * (j + 1)),
            ]
            for i in range(self.n_cos_bins)
            for j in range(self.n_mass_bins)
        }

    def grid_centers(self):
        """Return (cos_psi_grid, inv_mass_grid) meshgrids of bin centres."""
        cos_centers = np.arange(
            COS_BIN_MIN + COS_BIN_WIDTH / 2,
            COS_BIN_MIN + self.n_cos_bins * COS_BIN_WIDTH,
            COS_BIN_WIDTH,
        )
        mass_centers = np.arange(
            MASS_BIN_MIN + MASS_BIN_WIDTH / 2,
            MASS_BIN_MIN + self.n_mass_bins * MASS_BIN_WIDTH + MASS_BIN_WIDTH / 2,
            MASS_BIN_WIDTH,
        )
        return np.meshgrid(cos_centers, mass_centers)


def _region_dir(raw_dir, region):
    return os.path.join(
        raw_dir,
        f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}",
    )


def _theta_phi_paths(spec, save_dir):
    theta_paths = {k: os.path.join(save_dir, v) for k, v in spec.theta_filenames.items()}
    phi_paths = {k: os.path.join(save_dir, v) for k, v in spec.phi_filenames.items()}
    return theta_paths, phi_paths


def process_region(
    region_key,
    spec,
    raw_dir,
    regions,
    calc_bell=True,
    calc_concurrence=True,
    check_density=False,
    raw=False,
):
    region = regions[region_key]
    print(f"\n\nCalculating for region: {region}...\n")
    save_dir = _region_dir(raw_dir, region)
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} not found. Skipping region.")
        return None

    theta_paths, phi_paths = _theta_phi_paths(spec, save_dir)

    density_matrix = spec.get_density_matrix(theta_paths, phi_paths)
    if check_density:
        check_density_matrix(density_matrix)

    unphysicality = unphysicality_score(density_matrix)
    print(f"\nUnphysicality score for region: {region} = {unphysicality:.4g}\n")

    if not raw:
        density_matrix = project_to_psd(density_matrix, const=unphysicality, normalize_trace=True)
        if check_density:
            check_density_matrix(density_matrix)

    concurrence_val = 0.0
    bell_value = 0.0
    optimal_params = np.zeros(12)
    uncertainty_bell = 0.0

    if calc_concurrence:
        concurrence_val = concurrence_lower(density_matrix)
        print(f"\nConcurrence bound for region: {region} = {concurrence_val:.4g}\n")

    if calc_bell:
        bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
        optimal_O_bell = optimal_bell_operator(O_bell_prime1, optimal_params)
        print(f"Bell operator value for region: {region} = {bell_value:.4g}\n")

        variance = spec.get_variance(theta_paths, phi_paths, optimal_O_bell).real
        print(f"Variance of Bell operator for region: {region} = {variance:.6g}\n")
        uncertainty_bell = np.sqrt(variance)
        print(f"Uncertainty of Bell operator for region: {region} = {uncertainty_bell:.6g}\n")

    return {
        'region': region,
        'concurrence_val': concurrence_val,
        'bell_value': bell_value,
        'optimal_params': optimal_params,
        'uncertainty_bell': uncertainty_bell,
        'unphysicality': unphysicality,
    }


def generate_event_count_heatmap(spec, raw_dir, save_dir, regions, num_x_bins=180, num_y_bins=200):
    event_count_grid = np.zeros((num_y_bins, num_x_bins))

    cos_psi_edges = np.linspace(0, 0.9, num_x_bins + 1)
    inv_mass_edges = np.linspace(200, 1200, num_y_bins + 1)

    for key, region in regions.items():
        print("Calculating event count for region:", region)
        region_dir = _region_dir(raw_dir, region)

        if os.path.exists(region_dir):
            print(f"Directory {region_dir} found.")
            cos_psi_path = os.path.join(region_dir, spec.psi_filename)
            inv_mass_path = os.path.join(region_dir, spec.inv_mass_filename)

            cos_psi_data = np.loadtxt(cos_psi_path)
            inv_mass_data = np.loadtxt(inv_mass_path)

            hist2d, _, _ = np.histogram2d(
                inv_mass_data, cos_psi_data, bins=[inv_mass_edges, cos_psi_edges]
            )
            event_count_grid += hist2d

    plt.figure(figsize=FIGSIZE_HEATMAP)
    plt.imshow(event_count_grid, origin='lower', extent=[0, 0.9, 200, 1200],
               aspect='auto', cmap='inferno', vmin=0, vmax=2500)
    colorbar = plt.colorbar(label=r'Event Count', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(FONTSIZE_LABEL)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=FONTSIZE_LABEL)
    plt.ylabel(rf'$M_{{{spec.name}}} (GeV)$', fontsize=FONTSIZE_LABEL)

    base = os.path.join(save_dir, f"event_count_heatmap_{spec.name}_{num_x_bins}x{num_y_bins}")
    plt.savefig(base + ".pdf")
    plt.savefig(base + ".png")
    plt.close()

    return event_count_grid


def generate_uniformity_heatmap(spec, raw_dir, save_dir, regions):
    uniformity_grid = np.zeros((9, 20))

    for (i, j), region in regions.items():
        print(f"Calculating uniformity for region: {region}...")
        region_dir = _region_dir(raw_dir, region)

        if os.path.exists(region_dir):
            cos_psi_path = os.path.join(region_dir, spec.psi_filename)
            inv_mass_path = os.path.join(region_dir, spec.inv_mass_filename)

            cos_psi_data = np.loadtxt(cos_psi_path)
            inv_mass_data = np.loadtxt(inv_mass_path)

            h, _, _ = np.histogram2d(inv_mass_data, cos_psi_data, bins=[10, 10])
            mean = np.mean(h)
            std = np.std(h)

            uniformity = 1.0 - (std / mean) if mean > 0 else 0.0
            uniformity_grid[i, j] = uniformity

    uniformity_grid = uniformity_grid.T
    np.save(os.path.join(save_dir, "uniformity_scores.npy"), uniformity_grid)

    plt.figure(figsize=FIGSIZE_HEATMAP)
    plt.imshow(uniformity_grid, origin='lower', extent=[0, 0.9, 200, 1200],
               aspect='auto', cmap='plasma_r', vmin=0.7, vmax=1)
    colorbar = plt.colorbar(label='Uniformity Score', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(FONTSIZE_LABEL)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=FONTSIZE_LABEL)
    plt.ylabel(rf'$M_{{{spec.name}}} (GeV)$', fontsize=FONTSIZE_LABEL)
    plt.yticks(np.arange(200, 1201, 100), fontsize=FONTSIZE_TICK)
    plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=FONTSIZE_TICK)

    num_rows, num_cols = uniformity_grid.shape
    x_centers = np.linspace(0.05, 0.85, num_cols)
    y_centers = np.linspace(225.0, 1175.0, num_rows)
    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            plt.text(x, y, f"{uniformity_grid[i, j]:.2f}", color="white",
                     ha="center", va="center", fontsize=FONTSIZE_ANNOTATION)

    base = os.path.join(save_dir, f"uniformity_heatmap_{spec.name}")
    plt.savefig(base + ".pdf")
    plt.savefig(base + ".png")
    plt.close()

    return uniformity_grid


def generate_unphysicality_heatmap(spec, raw_dir, save_dir, regions, data=None):
    if data is None:
        unphysicality_grid = np.zeros((9, spec.n_mass_bins))

        for (i, j), region in regions.items():
            print(f"Calculating unphysicality for region: {region}...")
            region_dir = _region_dir(raw_dir, region)

            if os.path.exists(region_dir):
                theta_paths, phi_paths = _theta_phi_paths(spec, region_dir)
                density_matrix = spec.get_density_matrix(theta_paths, phi_paths)
                unphysicality = unphysicality_score(density_matrix)
                print(f"\nUnphysicality score for region: {region} = {unphysicality:.4g}\n")
                unphysicality_grid[i, j] = unphysicality
            else:
                print(f"Directory {region_dir} not found. Skipping region.")

        unphysicality_grid = unphysicality_grid.T
        np.save(os.path.join(save_dir, f"unphysicality_scores_{spec.name}.npy"), unphysicality_grid)
    else:
        unphysicality_grid = data

    plt.figure(figsize=FIGSIZE_HEATMAP)
    plt.imshow(unphysicality_grid, origin='lower', extent=[0, 0.9, 200, spec.mass_max],
               aspect='auto', cmap='plasma')
    colorbar = plt.colorbar(label='Unphysicality', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(FONTSIZE_LABEL)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=FONTSIZE_LABEL)
    plt.ylabel(rf'$M_{{{spec.name}}} (GeV)$', fontsize=FONTSIZE_LABEL)
    plt.yticks(np.arange(200, int(spec.mass_max) + 1, 100), fontsize=FONTSIZE_TICK)
    plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=FONTSIZE_TICK)

    num_rows, num_cols = unphysicality_grid.shape
    x_centers = np.linspace(0.05, 0.85, num_cols)
    y_centers = np.linspace(225.0, spec.mass_max - 25.0, num_rows)
    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            plt.text(x, y, f"{unphysicality_grid[i, j]:.2f}", color="white",
                     ha="center", va="center", fontsize=FONTSIZE_ANNOTATION)

    base = os.path.join(save_dir, f"unphysicality_heatmap_{spec.name}")
    plt.savefig(base + ".pdf")
    plt.savefig(base + ".png")
    plt.close()

    return unphysicality_grid
