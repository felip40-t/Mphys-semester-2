import numpy as np
import os
from utils.histo_plotter import read_data
from coefficient_calculator_ZZ import calculate_coefficients_AC, read_masked_data, calculate_coefficients_fgh, calculate_variance_AC
from core.density_matrix_calculator import calculate_density_matrix_AC, O_bell_prime1, calculate_density_matrix_fgh, project_to_psd, unphysicality_score
from core.Bell_inequality_optimizer import bell_inequality_optimization, inequality_function, optimal_bell_operator
from core.Unitary_Matrix import euler_unitary_matrix
from core.concurrence_bound import concurrence_lower, check_density_matrix
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from config import ZZ_RAW_DIR, ZZ_PROCESSED_DIR, ZZ_PLOTS_DIR  # also bootstraps src/ onto sys.path
from diboson.plotting.contour import plot_contour_heatmap as _plot_contour_heatmap
from diboson.analysis.region_analysis import (
    ProcessSpec,
    process_region as _process_region,
    generate_event_count_heatmap as _generate_event_count_heatmap,
    generate_uniformity_heatmap as _generate_uniformity_heatmap,
    generate_unphysicality_heatmap as _generate_unphysicality_heatmap,
)

ZZ_path = ZZ_RAW_DIR        # per-bin angular data subdirs
ZZ_processed = ZZ_PROCESSED_DIR  # bell/concurrence grids and coefficient files
ZZ_save = ZZ_PLOTS_DIR

regions = {
        (i, j): [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for i in range(9)
        for j in range(16)
        for cos_min in [0.0 + 0.1 * i]
        for mass_min in [200.0 + 50.0 * j]
    }


def _zz_get_density_matrix(theta_paths, phi_paths):
    A, C = calculate_coefficients_AC(theta_paths, phi_paths)
    return calculate_density_matrix_AC(A, C)


ZZ_SPEC = ProcessSpec(
    name="ZZ",
    n_mass_bins=16,
    mass_max=1000.0,
    psi_filename="psi_data_combined_new.txt",
    inv_mass_filename="ZZ_inv_mass_combined_new.txt",
    theta_filenames={1: "e+_theta_data_combined_new.txt", 3: "mu+_theta_data_combined_new.txt"},
    phi_filenames={1: "e+_phi_data_combined_new.txt", 3: "mu+_phi_data_combined_new.txt"},
    get_density_matrix=_zz_get_density_matrix,
    get_variance=calculate_variance_AC,
)

def generate_event_count_heatmap(ZZ_path, ZZ_save, regions, num_x_bins=180, num_y_bins=200):
    return _generate_event_count_heatmap(ZZ_SPEC, ZZ_path, ZZ_save, regions, num_x_bins=num_x_bins, num_y_bins=num_y_bins)


def generate_uniformity_heatmap(ZZ_path, ZZ_save, regions):
    return _generate_uniformity_heatmap(ZZ_SPEC, ZZ_path, ZZ_save, regions)


def generate_unphysicality_heatmap(ZZ_path=ZZ_path, ZZ_save=ZZ_save, regions=regions, unphysicality_grid=False):
    data = unphysicality_grid if isinstance(unphysicality_grid, np.ndarray) else None
    return _generate_unphysicality_heatmap(ZZ_SPEC, ZZ_path, ZZ_save, regions, data=data)


def process_region(region_key, ZZ_path=ZZ_path, regions=regions, calc_bell=True, calc_concurrence=True, check_density=False, raw=False):
    return _process_region(region_key, ZZ_SPEC, ZZ_path, regions, calc_bell=calc_bell, calc_concurrence=calc_concurrence, check_density=check_density, raw=raw)

def plot_contour_heatmap(ZZ_save, cos_psi_grid, inv_mass_grid, bell_value_grid, label, concurrence=False):
    _plot_contour_heatmap(ZZ_save, cos_psi_grid, inv_mass_grid, bell_value_grid, label, "ZZ", concurrence=concurrence)

process_region((8, 4), ZZ_path=ZZ_path, regions=regions, calc_bell=True, calc_concurrence=True, check_density=True, raw=True)

cos_psi_centers = np.arange(0.05, 0.9, 0.1)         
inv_mass_centers = np.arange(225.0, 1001.0, 50.0) 

cos_psi_grid, inv_mass_grid = np.meshgrid(cos_psi_centers, inv_mass_centers)

label = "raw"

bell_value_grid = np.loadtxt(os.path.join(ZZ_processed, f"bell_operator_grid_ZZ_AC_{label}.txt"), delimiter=',')
uncertainty_grid = np.loadtxt(os.path.join(ZZ_processed, f"uncertainty_grid_ZZ_AC_{label}.txt"), delimiter=',')
concurrence_grid = np.loadtxt(os.path.join(ZZ_processed, f"concurrence_grid_ZZ_AC_{label}.txt"), delimiter=',')
optimal_params_grid = np.load(os.path.join(ZZ_processed, f"optimal_params_grid_ZZ_AC_{label}.npy"))


for key, region in regions.items():
    quantities = process_region(key, ZZ_path=ZZ_path, regions=regions, calc_bell=True, calc_concurrence=True, check_density=False, raw=True)
    if quantities is not None:
        i, j = key
        bell_value_grid[i, j] = quantities['bell_value']
        uncertainty_grid[i, j] = quantities['uncertainty_bell']
        concurrence_grid[i, j] = quantities['concurrence_val']
        optimal_params_grid[:, i, j] = quantities['optimal_params']

        # Save the Bell operator value grid to a file
        np.savetxt(os.path.join(ZZ_processed, f"bell_operator_grid_ZZ_AC_{label}.txt"), bell_value_grid, delimiter=',')

        # Save the concurrence value grid to a file
        np.savetxt(os.path.join(ZZ_processed, f"concurrence_grid_ZZ_AC_{label}.txt"), concurrence_grid, delimiter=',')

        # Save the optimal parameters grid to a npy file
        np.save(os.path.join(ZZ_processed, f"optimal_params_grid_ZZ_AC_{label}.npy"), optimal_params_grid)

        # Save the uncertainty grid to a file
        np.savetxt(os.path.join(ZZ_processed, f"uncertainty_grid_ZZ_AC_{label}.txt"), uncertainty_grid, delimiter=',')



# bell_matrix = np.array([
#     [2.15, 2.15, 2.1,  2,    1.9,  1.8,  1.75, 1.65, 1.55],
#     [2.15, 2.15, 2.05, 2,    1.9,  1.8,  1.75, 1.65, 1.55],
#     [2.15, 2.15, 2.05, 2,    1.9,  1.8,  1.75, 1.65, 1.55],
#     [2.15, 2.15, 2.05, 2,    1.9,  1.8,  1.75, 1.65, 1.55],
#     [2.15, 2.15, 2.05, 2,    1.9,  1.8,  1.75, 1.65, 1.55],
#     [2.05, 2.05, 2.05, 2,    1.9,  1.8,  1.75, 1.65, 1.55],
#     [2.05, 2.05, 2.05, 2,    1.9,  1.8,  1.75, 1.65, 1.55],
#     [2.05, 2.05, 2.05, 1.95,    1.9,  1.8,  1.75, 1.65, 1.55],
#     [1.95, 1.95, 1.95, 1.95, 1.9,  1.75,  1.75, 1.65, 1.55],
#     [1.9,  1.9,  1.9,  1.85,  1.85,  1.8,  1.75, 1.65, 1.55],
#     [1.85, 1.85, 1.85, 1.85, 1.8,  1.8,  1.75, 1.65, 1.55],
#     [1.8,  1.8,  1.8,  1.8,  1.8,  1.8,  1.75, 1.7, 1.55],
#     [1.6,  1.6,  1.6,  1.6,  1.6,  1.6,  1.75,  1.7, 1.55],
#     [1.4,  1.4,  1.4,  1.4,  1.4,  1.5,  1.6,  1.65,  1.55],
# ])[::-1]

# bell_matrix = gaussian_filter(bell_matrix, sigma=0.5)  # Apply Gaussian filter for smoothing

# cos_psi_centers = np.arange(0.05, 0.9, 0.1)        
# inv_mass_centers = np.arange(225.0, 900.0, 50.0)    

# cos_psi_grid, inv_mass_grid = np.meshgrid(cos_psi_centers, inv_mass_centers)


# # Plot the contour of Bell operator values
# plt.figure(figsize=(12, 10))
# # Define custom contour levels
# custom_levels = np.arange(1.3, 2.3, step=0.1)
# contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, bell_matrix, levels=custom_levels, cmap='plasma')
# contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, bell_matrix, levels=custom_levels, colors='black', linewidths=0.7)
# plt.clabel(contour_lines, inline=True, fontsize=12, fmt="%.2f")
# colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{I}_3$', orientation='vertical')
# colorbar.ax.yaxis.label.set_fontsize(16)
# plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
# plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)
# plt.yticks(np.arange(300, 900, 100), fontsize=12)
# plt.xticks(np.arange(0.1, 0.9, 0.1), fontsize=12)
# plot_filename = os.path.join(ZZ_save, "bell_operator_contour_ZZ_paper.pdf")
# plt.savefig(plot_filename)
