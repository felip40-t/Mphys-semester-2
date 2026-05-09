import numpy as np
import os

from diboson.physics.coefficients import calculate_coefficients_fgh, calculate_variance_fgh
from diboson.physics.density_matrix import calculate_density_matrix_fgh
from diboson.physics.bell_optimiser import bell_inequality_optimization

from diboson.physics.concurrence import concurrence_lower

from config import WW_RAW_DIR, WW_PROCESSED_DIR, WW_PLOTS_DIR, WW_ETA, N_COS_BINS  # also bootstraps src/ onto sys.path

from diboson.plotting.contour import plot_contour_heatmap as _plot_contour_heatmap
from diboson.analysis.region_analysis import (
    ProcessSpec,
    process_region as _process_region,
    generate_event_count_heatmap as _generate_event_count_heatmap,
    generate_uniformity_heatmap as _generate_uniformity_heatmap,
    generate_unphysicality_heatmap as _generate_unphysicality_heatmap,
)


WW_path = WW_RAW_DIR          # per-bin angular data subdirs
WW_processed = WW_PROCESSED_DIR  # bell/concurrence grids and coefficient files
WW_save = WW_PLOTS_DIR

RAW_ANALYSIS = False # Whether to perform the positive semi definite projection of the density matrix


def _ww_get_density_matrix(theta_paths, phi_paths):
    f, g, h = calculate_coefficients_fgh(theta_paths, phi_paths)
    return calculate_density_matrix_fgh(f, g, h)


WW_SPEC = ProcessSpec(
    name="WW",
    n_mass_bins=20,
    n_cos_bins=N_COS_BINS,
    mass_max=1200.0,
    eta=WW_ETA,
    psi_filename="psi_data.txt",
    inv_mass_filename="WW_inv_mass.txt",
    theta_filenames={1: "e+_theta_data.txt", 3: "mu-_theta_data.txt"},
    phi_filenames={1: "e+_phi_data.txt", 3: "mu-_phi_data.txt"},
    get_density_matrix=_ww_get_density_matrix,
    get_variance=calculate_variance_fgh,
)

regions = WW_SPEC.build_regions()

label = "raw" if RAW_ANALYSIS else "projected"

def generate_event_count_heatmap(WW_path, WW_save, regions, num_x_bins=180, num_y_bins=200):
    return _generate_event_count_heatmap(WW_SPEC, WW_path, WW_save, regions, num_x_bins=num_x_bins, num_y_bins=num_y_bins)


def generate_uniformity_heatmap(WW_path, WW_save, regions):
    return _generate_uniformity_heatmap(WW_SPEC, WW_path, WW_save, regions)


def generate_unphysicality_heatmap(WW_path=WW_path, WW_save=WW_save, regions=regions, data=None):
    return _generate_unphysicality_heatmap(WW_SPEC, WW_path, WW_save, regions, data=data)


def process_region(region_key, WW_path=WW_path, regions=regions, calc_bell=True, calc_concurrence=True, check_density=False, raw=False):
    return _process_region(region_key, WW_SPEC, WW_path, regions, calc_bell=calc_bell, calc_concurrence=calc_concurrence, check_density=check_density, raw=raw)

def plot_contour_heatmap(WW_save, cos_psi_grid, inv_mass_grid, bell_value_grid, label, concurrence=False):
    _plot_contour_heatmap(WW_save, cos_psi_grid, inv_mass_grid, bell_value_grid, label, "WW", concurrence=concurrence)

cos_psi_grid, inv_mass_grid = WW_SPEC.grid_centers()

# Initialize the Bell operator, uncertainty, and concurrence grids for regions
bell_value_grid = np.loadtxt(os.path.join(WW_processed, f"bell_operator_grid_WW_fgh_{label}.txt"), delimiter=',')
uncertainty_grid = np.loadtxt(os.path.join(WW_processed, f"uncertainty_grid_WW_fgh_{label}.txt"), delimiter=',')
concurrence_grid = np.loadtxt(os.path.join(WW_processed, f"concurrence_grid_WW_fgh_{label}.txt"), delimiter=',')
optimal_params_grid = np.load(os.path.join(WW_processed, f"optimal_params_grid_WW_fgh_{label}.npy"))


for key, region in regions.items():
    quantities = process_region(key, WW_path=WW_path, regions=regions, calc_bell=True, calc_concurrence=True, check_density=False, raw=RAW_ANALYSIS)
    if quantities is not None:
        i, j = key
        bell_value_grid[i, j] = quantities['bell_value']
        uncertainty_grid[i, j] = quantities['uncertainty_bell']
        concurrence_grid[i, j] = quantities['concurrence_val']
        optimal_params_grid[:, i, j] = quantities['optimal_params']
            
        # Save the Bell operator value grid to a file
        np.savetxt(os.path.join(WW_processed, f"bell_operator_grid_WW_fgh_{label}.txt"), bell_value_grid, delimiter=',')

        # Save the concurrence value grid to a file
        np.savetxt(os.path.join(WW_processed, f"concurrence_grid_WW_fgh_{label}.txt"), concurrence_grid, delimiter=',')

        # Save the optimal parameters grid to a npy file
        np.save(os.path.join(WW_processed, f"optimal_params_grid_WW_fgh_{label}.npy"), optimal_params_grid)

        # Save the uncertainty grid to a file
        np.savetxt(os.path.join(WW_processed, f"uncertainty_grid_WW_fgh_{label}.txt"), uncertainty_grid, delimiter=',')

# Read the Bell operator value grid from the file
bell_value_grid = np.loadtxt(os.path.join(WW_processed, f"bell_operator_grid_WW_fgh_{label}.txt"), delimiter=',').T

plot_contour_heatmap(WW_save, cos_psi_grid, inv_mass_grid, bell_value_grid, label=label, concurrence=False)

# Read the concurrence value grid from the file
concurrence_grid = np.loadtxt(os.path.join(WW_processed, f"concurrence_grid_WW_fgh_{label}.txt"), delimiter=',').T

plot_contour_heatmap(WW_save, cos_psi_grid, inv_mass_grid, concurrence_grid, label=label, concurrence=True)



