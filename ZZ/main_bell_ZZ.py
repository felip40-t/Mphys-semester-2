import numpy as np
import os

from coefficient_calculator_ZZ import calculate_coefficients_AC, calculate_variance_AC
from core.density_matrix_calculator import calculate_density_matrix_AC
from core.Bell_inequality_optimizer import bell_inequality_optimization

from core.concurrence_bound import concurrence_lower

from config import ZZ_RAW_DIR, ZZ_PROCESSED_DIR, ZZ_PLOTS_DIR, ZZ_ETA, N_COS_BINS  # also bootstraps src/ onto sys.path

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

RAW_ANALYSIS = False # Whether to perform the positive semi definite projection of the density matrix

def _zz_get_density_matrix(theta_paths, phi_paths):
    A, C = calculate_coefficients_AC(theta_paths, phi_paths)
    return calculate_density_matrix_AC(A, C)


ZZ_SPEC = ProcessSpec(
    name="ZZ",
    n_mass_bins=16,
    n_cos_bins=N_COS_BINS,
    mass_max=1000.0,
    eta=ZZ_ETA,
    psi_filename="psi_data.txt",
    inv_mass_filename="ZZ_inv_mass.txt",
    theta_filenames={1: "e+_theta_data.txt", 3: "mu+_theta_data.txt"},
    phi_filenames={1: "e+_phi_data.txt", 3: "mu+_phi_data.txt"},
    get_density_matrix=_zz_get_density_matrix,
    get_variance=calculate_variance_AC,
)

regions = ZZ_SPEC.build_regions()

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

cos_psi_grid, inv_mass_grid = ZZ_SPEC.grid_centers()

label = "raw" if RAW_ANALYSIS else "projected"

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

# Plot the Bell operator heatmap
bell_value_grid = np.loadtxt(os.path.join(ZZ_processed, f"bell_operator_grid_ZZ_AC_{label}.txt"), delimiter=',')
plot_contour_heatmap(ZZ_save, cos_psi_grid, inv_mass_grid, bell_value_grid, label, concurrence=False)
# Plot the concurrence heatmap
concurrence_grid = np.loadtxt(os.path.join(ZZ_processed, f"concurrence_grid_ZZ_AC_{label}.txt"), delimiter=',')
plot_contour_heatmap(ZZ_save, cos_psi_grid, inv_mass_grid, concurrence_grid, label, concurrence=True)