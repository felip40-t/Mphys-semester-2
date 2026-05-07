
from matplotlib.colors import ListedColormap
import numpy as np
import os
from utils.histo_plotter import read_data
from coefficient_calculator_WW import calculate_coefficients, read_masked_data, calculate_coefficients_fgh, calculate_variance_fgh
from core.density_matrix_calculator import calculate_density_matrix_AC, O_bell_prime1, calculate_density_matrix_fgh, project_to_psd, unphysicality_score
from core.Bell_inequality_optimizer import bell_inequality_optimization, inequality_function, optimal_bell_operator
from core.Unitary_Matrix import euler_unitary_matrix
from core.concurrence_bound import concurrence_lower, check_density_matrix, concurrence_MB
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from config import WW_RAW_DIR, WW_PROCESSED_DIR, WW_PLOTS_DIR  # also bootstraps src/ onto sys.path
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

regions = {
        (i, j): [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for i in range(9)
        for j in range(20)
        for cos_min in [0.0 + 0.1 * i]
        for mass_min in [200.0 + 50.0 * j]
    }


def _ww_get_density_matrix(theta_paths, phi_paths):
    f, g, h = calculate_coefficients_fgh(theta_paths, phi_paths)
    return calculate_density_matrix_fgh(f, g, h)


WW_SPEC = ProcessSpec(
    name="WW",
    n_mass_bins=20,
    mass_max=1200.0,
    psi_filename="psi_data.txt",
    inv_mass_filename="WW_inv_mass.txt",
    theta_filenames={1: "e+_theta_data.txt", 3: "mu-_theta_data.txt"},
    phi_filenames={1: "e+_phi_data.txt", 3: "mu-_phi_data.txt"},
    get_density_matrix=_ww_get_density_matrix,
    get_variance=calculate_variance_fgh,
)

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

# Generate mesh grid for cos_psi and WW_inv_mass based on regions (9 columns and 20 rows)
# Each cos_psi region is [cos_min, cos_min+0.1] so we take the center as cos_min + 0.05 for 0 <= cos_min < 0.9
# Each WW_inv_mass region is [mass_min, mass_min+50.0] so we take the center as mass_min + 25.0 for 200 <= mass_min < 1200
cos_psi_centers = np.arange(0.05, 0.9, 0.1)         
inv_mass_centers = np.arange(225.0, 1200.0, 50.0)   

cos_psi_grid, inv_mass_grid = np.meshgrid(cos_psi_centers, inv_mass_centers)

# Initialize the Bell operator, uncertainty, and concurrence grids for 20x9 regions
bell_value_grid = np.loadtxt(os.path.join(WW_processed, "bell_operator_grid_WW_fgh_smooth_clip.txt"), delimiter=',')
uncertainty_grid = np.loadtxt(os.path.join(WW_processed, "uncertainty_grid_WW_fgh_smooth_clip.txt"), delimiter=',')
concurrence_grid = np.loadtxt(os.path.join(WW_processed, "concurrence_grid_WW_fgh_smooth_clip.txt"), delimiter=',')
optimal_params_grid = np.load(os.path.join(WW_processed, "optimal_params_grid_WW_fgh_smooth_clip.npy"))


for key, region in regions.items():
    quantities = process_region(key, WW_path=WW_path, regions=regions, calc_bell=True, calc_concurrence=True, check_density=False, raw=False)
    if quantities is not None:
        i, j = key
        bell_value_grid[i, j] = quantities['bell_value']
        uncertainty_grid[i, j] = quantities['uncertainty_bell']
        concurrence_grid[i, j] = quantities['concurrence_val']
        optimal_params_grid[:, i, j] = quantities['optimal_params']

        # Save the Bell operator value grid to a file
        np.savetxt(os.path.join(WW_processed, "bell_operator_grid_WW_fgh_smooth_clip.txt"), bell_value_grid, delimiter=',')

        # Save the concurrence value grid to a file
        np.savetxt(os.path.join(WW_processed, "concurrence_grid_WW_fgh_smooth_clip.txt"), concurrence_grid, delimiter=',')

        # Save the optimal parameters grid to a npy file
        np.save(os.path.join(WW_processed, "optimal_params_grid_WW_fgh_smooth_clip.npy"), optimal_params_grid)

        # Save the uncertainty grid to a file
        np.savetxt(os.path.join(WW_processed, "uncertainty_grid_WW_fgh_smooth_clip.txt"), uncertainty_grid, delimiter=',')

type = "smooth_clip"

# Read the Bell operator value grid from the file
bell_value_grid = np.loadtxt(os.path.join(WW_path, f"bell_operator_grid_WW_fgh_{type}.txt"), delimiter=',').T

plot_contour_heatmap(WW_save, cos_psi_grid, inv_mass_grid, bell_value_grid, label=type, concurrence=False)

# Read the concurrence value grid from the file
concurrence_grid = np.loadtxt(os.path.join(WW_path, f"concurrence_grid_WW_fgh_{type}.txt"), delimiter=',').T

plot_contour_heatmap(WW_save, cos_psi_grid, inv_mass_grid, concurrence_grid, label=type, concurrence=True)


bell_matrix = np.array([
    [2.18, 2.15, 2.05, 2.05, 1.95, 1.8, 1.75, 1.7, 1.65],
    [2.15, 2.15, 2.05, 2.05, 1.95, 1.8, 1.75, 1.7, 1.65],
    [2.15, 2.15, 2.05, 2.05, 1.95, 1.8, 1.75, 1.7, 1.65],
    [2.15, 2.15, 2.05, 2.05, 1.95, 1.8, 1.75, 1.7, 1.65],
    [2.15, 2.05, 2.05, 2.05, 1.95, 1.8, 1.75, 1.7, 1.65],
    [2.05, 2.05, 2.05, 2.05, 1.95, 1.75, 1.75, 1.7, 1.65],
    [2.05, 2.05, 2.05, 1.99, 1.95, 1.75, 1.75, 1.7, 1.6],
    [2.05, 2.05, 2.0, 1.99, 1.95, 1.75, 1.75, 1.7, 1.5],
    [2.0, 2.0, 1.99, 1.95, 1.95, 1.8, 1.75, 1.7, 1.5],
    [1.95, 1.95, 1.95, 1.9, 1.9, 1.8, 1.75, 1.7, 1.4],
    [1.95, 1.95, 1.95, 1.85, 1.85, 1.75, 1.75, 1.7, 1.4],
    [1.85, 1.85, 1.85, 1.8, 1.8, 1.65, 1.75, 1.7, 1.5],
    [1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.75, 1.75, 1.5],
    [1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.75, 1.75, 1.65]
])

bell_matrix = bell_matrix[::-1]
bell_matrix = gaussian_filter(bell_matrix, sigma=0.5)  # Apply Gaussian filter for smoothing

bell_value_grid = bell_value_grid[:14, :]

# Compute absolute discrepancy
discrepancy_grid = bell_value_grid - bell_matrix

# Load unphysicality grid
unphysicality_grid = np.load(os.path.join(WW_save, "unphysicality_scores_WW.npy"))
unphysicality_grid = unphysicality_grid[:14, :]

# Test correlation between unphysicality and discrepancy
from scipy.stats import spearmanr
correlation = spearmanr(unphysicality_grid.flatten(), discrepancy_grid.flatten()).correlation
print(f"Correlation between unphysicality and discrepancy: {correlation:.4f}")

# Plot the heatmap
plt.figure(figsize=(12, 10))
plt.imshow(discrepancy_grid, origin='lower', cmap='magma', aspect='auto', 
           extent=[0, 0.9, 200, 900])
cbar = plt.colorbar(label=r'$\Delta \mathcal{I}_3$', orientation='vertical')
cbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
plt.ylabel(r'$M_{WW} \, [\mathrm{GeV}]$', fontsize=16)

# Tick labels
plt.yticks(np.arange(200, 901, 100), fontsize=14)
plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=14)

# Add text annotations to each grid cell
num_rows, num_cols = discrepancy_grid.shape

x_centers = np.linspace(0.05, 0.85, num_cols)
y_centers = np.linspace(225.0, 875.0, num_rows)
for i, y in enumerate(y_centers):
    for j, x in enumerate(x_centers):
        plt.text(x, y, f"{discrepancy_grid[i, j]:.2f}", color="white", 
                    ha="center", va="center", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(WW_save, "bell_operator_discrepancy_heatmap_WW_fgh_raw.pdf"))
plt.savefig(os.path.join(WW_save, "bell_operator_discrepancy_heatmap_WW_fgh_raw.png"))
plt.close()

