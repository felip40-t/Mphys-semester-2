import numpy as np
import os
from histo_plotter import read_data
from coefficient_calculator_ZZ import calculate_coefficients_AC, read_masked_data
from density_matrix_calculator import calculate_density_matrix_AC, O_bell_prime1
from Bell_inequality_optimizer import bell_inequality_optimization, inequality_function
from Unitary_Matrix import euler_unitary_matrix
from concurrence_bound import concurrence_lower, check_density_matrix, concurrence_upper
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

ZZ_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_ZZ_SM/Plots and data/reorganised_data"
ZZ_save = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_ZZ_SM/Plots and data/reorganised_data/Entanglement plots"

regions = { 
        (i, j): [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for i in range(9)
        for j in range(16)
        for cos_min in [0.0 + 0.1 * i]
        for mass_min in [200.0 + 50.0 * j]
    }

# num_x_bins, num_y_bins = 100, 100
# event_count_grid = np.zeros((num_y_bins, num_x_bins))

# # Define bin edges for high-resolution mapping
# cos_psi_edges = np.linspace(0, 0.9, num_x_bins + 1)  # Bin edges in cos_psi
# inv_mass_edges = np.linspace(200, 1000, num_y_bins + 1)  # Bin edges in M_ZZ

# # Loop through each defined region and bin events
# for key, region in regions.items():
#     print("Calculating event count for region:", region)
#     save_dir = os.path.join(ZZ_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
    
#     if os.path.exists(save_dir):
#         print(f"Directory {save_dir} found.")
        
#         # Load event data
#         cos_psi_path = os.path.join(save_dir, "psi_data_combined_new.txt")
#         ZZ_inv_path = os.path.join(save_dir, "ZZ_inv_mass_combined_new.txt")

#         cos_psi_data = np.loadtxt(cos_psi_path)
#         ZZ_inv_mass = np.loadtxt(ZZ_inv_path)

#         # 2D histogram binning into 100x100 grid
#         hist2d, _, _ = np.histogram2d(ZZ_inv_mass, cos_psi_data, bins=[inv_mass_edges, cos_psi_edges])
        
#         # Accumulate event counts into the main grid
#         event_count_grid += hist2d

# # Plot the heatmap of event counts
# plt.figure(figsize=(8, 6))
# plt.imshow(event_count_grid, origin='lower', extent=[0, 0.9, 200, 1000], aspect='auto', cmap='inferno', vmin=0, vmax=4000)
# colorbar = plt.colorbar(label=r'Event Count', orientation='vertical')
# colorbar.ax.yaxis.label.set_fontsize(16)
# plt.xlabel(r'$\cos{\Theta}$', fontsize=16)   
# plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)

# # Save the heatmap
# heatmap_filename = os.path.join(ZZ_path, "Entanglement plots/event_count_heatmap_ZZ_100x100.pdf")
# plt.savefig(heatmap_filename)
# heatmap_filename = os.path.join(ZZ_path, "Entanglement plots/event_count_heatmap_ZZ_100x100.png")
# plt.savefig(heatmap_filename)

# regions = { 
#     (i, j): [(cos_min, cos_min + 0.1), (mass_min, mass_min + 100.0)]
#     for i in range(9)
#     for j in range(8)  # Reduce to 8 mass bins since we double the range
#     for cos_min in [0.0 + 0.1 * i]
#     for mass_min in [200.0 + 100.0 * j]  # Increase step size to 100 GeV
# }

# uniformity_grid = np.zeros((9, 16))  # To store uniformity scores

# for (i, j), region in regions.items():
#     print(f"Calculating uniformity for region: {region}...")

#     save_dir = os.path.join(ZZ_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
    
#     if os.path.exists(save_dir):
#         cos_psi_path = os.path.join(save_dir, "psi_data_combined_new.txt")
#         ZZ_inv_path = os.path.join(save_dir, "ZZ_inv_mass_combined_new.txt")

#         cos_psi_data = np.loadtxt(cos_psi_path)
#         ZZ_inv_mass = np.loadtxt(ZZ_inv_path)


#         # Histogram within region: use 10x10 binning
#         h, _, _ = np.histogram2d(ZZ_inv_mass, cos_psi_data, bins=[10, 10])
#         mean = np.mean(h)
#         std = np.std(h)

#         # Avoid divide by zero
#         if mean > 0:
#             uniformity = 1.0 - (std / mean)
#         else:
#             uniformity = 0.0

#         # Store the moment-based gradient in the uniformity grid
#         uniformity_grid[i, j] = uniformity


# # Save the uniformity scores as a .npy file
# uniformity_grid = uniformity_grid.T  # Transpose to match the grid shape
# np.save(os.path.join(ZZ_save, "uniformity_scores.npy"), uniformity_grid)

# # Read uniformity scores
# uniformity_grid = np.load(os.path.join(ZZ_save, "uniformity_scores.npy"))

# # Plot the heatmap of uniformity scores
# plt.figure(figsize=(12, 10))
# plt.imshow(uniformity_grid, origin='lower', extent=[0, 0.9, 200, 1000], aspect='auto', cmap='plasma_r', vmin=0.7, vmax=1)
# colorbar = plt.colorbar(label='Uniformity Score', orientation='vertical')
# colorbar.ax.yaxis.label.set_fontsize(16)
# plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
# plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)
# plt.yticks(np.arange(200, 1001, 100), fontsize=14)
# plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=14)

# # Add the value of the uniformity score to each square
# num_rows, num_cols = uniformity_grid.shape
# x_centers = np.linspace(0.05, 0.85, num_cols)
# y_centers = np.linspace(225.0, 976.0, num_rows)
# for i, y in enumerate(y_centers):
#     for j, x in enumerate(x_centers):
#         score = uniformity_grid[i, j]
#         plt.text(x, y, f"{score:.2f}", color="white", ha="center", va="center", fontsize=12)

# # Save the plot
# heatmap_filename = os.path.join(ZZ_save, "uniformity_heatmap_ZZ.pdf")
# plt.savefig(heatmap_filename)
# heatmap_filename = os.path.join(ZZ_save, "uniformity_heatmap_ZZ.png")
# plt.savefig(heatmap_filename)



# bell_matrix = np.array([
#     [2.15, 2.15, 2.15, 2,    1.9,  1.8,  1.75, 1.65, 1.55],
#     [2.15, 2.15, 2.1,  2,    1.9,  1.8,  1.75, 1.65, 1.55],
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
# ])[::-1][:14, :]

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


regions = { 
        (i, j): [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for i in range(9)
        for j in range(16)
        for cos_min in [0.0 + 0.1 * i]
        for mass_min in [200.0 + 50.0 * j]
    }

bell_value_grid = np.zeros((9, 16, 4))
concurrence_grid = np.zeros((9, 16, 4))
optimal_params_grid = np.zeros((16, 9, 16, 4))

bell_value_avg_grid = np.zeros((9, 16))
concurrence_avg_grid = np.zeros((9, 16))

for key, region in regions.items():
    print("Calculating for region:", region)
    save_dir = os.path.join(ZZ_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
    if not os.path.exists(save_dir):
        print(f"Directory not found.")
    
    cos_theta_paths = {
    1: os.path.join(save_dir, "e+_theta_data_combined_new.txt"),
    3: os.path.join(save_dir, "mu+_theta_data_combined_new.txt")
    }
    phi_paths = {
        1: os.path.join(save_dir, "e+_phi_data_combined_new.txt"),
        3: os.path.join(save_dir, "mu+_phi_data_combined_new.txt")
    }

    cos_psi_path = os.path.join(save_dir, "psi_data_combined_new.txt")
    ZZ_inv_path = os.path.join(save_dir, "ZZ_inv_mass_combined_new.txt")
    cos_psi_data = np.loadtxt(cos_psi_path)
    ZZ_inv_mass = np.loadtxt(ZZ_inv_path)

    cos_range = region[0]
    mass_range = region[1]

    # Create subregions by halving both axes
    subregions = [
        [(cos_range[0], (cos_range[0] + cos_range[1]) / 2), (mass_range[0], (mass_range[0] + mass_range[1]) / 2)],  # bottom-left
        [(cos_range[0], (cos_range[0] + cos_range[1]) / 2), ((mass_range[0] + mass_range[1]) / 2, mass_range[1])],  # top-left
        [((cos_range[0] + cos_range[1]) / 2, cos_range[1]), (mass_range[0], (mass_range[0] + mass_range[1]) / 2)],  # bottom-right
        [((cos_range[0] + cos_range[1]) / 2, cos_range[1]), ((mass_range[0] + mass_range[1]) / 2, mass_range[1])]   # top-right
    ]

    weighted_bell_sum = 0.0
    weighted_conc_sum = 0.0
    total_events = 0

    # Loop over subregions and compute Bell operator
    for idx, subregion in enumerate(subregions):
        cos_sub, mass_sub = subregion
        mask = read_masked_data(cos_psi_data, ZZ_inv_mass, cos_sub, mass_sub)
        event_count = np.sum(mask)

        A_coefficients, C_coefficients, A_uncertainties, C_uncertainties = calculate_coefficients_AC(cos_theta_paths, phi_paths, mask)
        density_matrix, uncertainty_matrix_real, uncertainty_matrix_imag = calculate_density_matrix_AC(A_coefficients, C_coefficients, A_uncertainties, C_uncertainties)
        bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
        concurrence_val = concurrence_lower(density_matrix)

        print(f"Subregion {subregion}:\n Bell operator = {bell_value:.4f}\n Concurrence = {concurrence_val:.4f}\n")
        
        # Store the Bell operator value and concurrence in the grid
        bell_value_grid[key[0], key[1], idx] = bell_value
        concurrence_grid[key[0], key[1], idx] = concurrence_val
        # Store the optimal parameters in the grid
        optimal_params_grid[:, key[0], key[1], idx] = optimal_params

        # Weighted accumulation
        weighted_bell_sum += bell_value * event_count
        weighted_conc_sum += concurrence_val * event_count
        total_events += event_count

    bell_value_avg_grid[key[0], key[1]] = weighted_bell_sum / total_events
    concurrence_avg_grid[key[0], key[1]] = weighted_conc_sum / total_events

# Save the weighted average Bell operator and concurrence grids
np.savetxt(os.path.join(ZZ_save, "bell_operator_weighted_avg_grid_ZZ_9x16.txt"), bell_value_avg_grid, delimiter=',')
np.savetxt(os.path.join(ZZ_save, "concurrence_weighted_avg_grid_ZZ_9x16.txt"), concurrence_avg_grid, delimiter=',')

# Save the full grids with all subregion data
np.save(os.path.join(ZZ_save, "bell_operator_subregions_ZZ_9x16x4.npy"), bell_value_grid)
np.save(os.path.join(ZZ_save, "concurrence_subregions_ZZ_9x16x4.npy"), concurrence_grid)
np.save(os.path.join(ZZ_save, "optimal_params_grid_ZZ_16x9x16x4.npy"), optimal_params_grid)

# print("Bell operator value grid:")
# for row in bell_value_grid:
#     print(" ".join(f"{val:.4f}" for val in row))

# # Save the Bell operator value grid to a file
# np.savetxt(os.path.join(ZZ_path, "Entanglement plots/bell_operator_grid_ZZ_small_24M.txt"), bell_value_grid, delimiter=',')

# # Save the optimal parameters grid to a file
# np.save(os.path.join(ZZ_path, "Entanglement plots/optimal_params_grid_ZZ_small_24M.npy"), optimal_params_grid)

# # Save the concurrence grid to a file
# np.savetxt(os.path.join(ZZ_path, "Entanglement plots/concurrence_grid_ZZ_test_24M.txt"), merged_concurrence_grid, delimiter=',')

# print("Bell operator value grid:")
# for row in bell_value_grid:
#     print(" ".join(f"{val:.4f}" for val in row))
# print("Average Bell operator value:", np.mean(bell_value_grid))


# # Read the Bell operator value grid from the file
# bell_value_grid = np.loadtxt(os.path.join(ZZ_path, "Entanglement plots/bell_operator_grid_ZZ_small_24M.txt"), delimiter=',')
# bell_value_grid = bell_value_grid.T  # Transpose the matrix instead of reshaping


# # Define cos_psi and mass grid
# cos_psi_grid = np.linspace(0.05, 0.85, 9)  # Midpoints of cos_psi bins
# inv_mass_grid = np.linspace(225, 975, 16)  # Midpoints of mass bins

# # Apply gaussian smoothing
# smoothed_bell_value_grid = gaussian_filter(bell_value_grid, sigma=1.0)

# # Plot the contour of Bell operator values
# plt.figure(figsize=(10, 8))
# # Define custom contour levels
# custom_levels = np.arange(1.0, 2.4, step=0.1)
# contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, smoothed_bell_value_grid, levels=custom_levels, cmap='Blues')
# contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, smoothed_bell_value_grid, levels=custom_levels, colors='black', linewidths=0.7)
# plt.clabel(contour_lines, inline=True, fontsize=12, fmt="%.2f")
# colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{I}_3$', orientation='vertical')
# colorbar.ax.yaxis.label.set_fontsize(16)
# plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
# plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)
# plot_filename = os.path.join(ZZ_path, "Entanglement plots/smoothed_bell_operator_contour_ZZ_small_24M.pdf")
# plt.savefig(plot_filename)
# plot_filename_png = os.path.join(ZZ_path, "Entanglement plots/smoothed_bell_operator_contour_ZZ_small_24M.png")
# plt.savefig(plot_filename_png)

# # Plot the 2D heatmap of Bell operator values
# plt.figure(figsize=(8, 6))
# plt.imshow(smoothed_bell_value_grid, origin='lower', extent=[0, 0.9, 200, 1000], aspect='auto', cmap='plasma')
# colorbar = plt.colorbar(label=r'$\mathcal{I}_3$', orientation='vertical')
# colorbar.ax.yaxis.label.set_fontsize(16)
# plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
# plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)

# # Add the value of the Bell operator as a label to each square
# num_rows, num_cols = smoothed_bell_value_grid.shape
# x_centers = np.linspace(0.05, 0.85, num_cols)  # Center of bins for cos_psi
# y_centers = np.linspace(225.0, 975.0, num_rows)  # Center of bins for M_ZZ
# for i, y in enumerate(y_centers):
#     for j, x in enumerate(x_centers):
#         plt.text(x, y, f"{smoothed_bell_value_grid[i, j]:.2f}", color="white", ha="center", va="center", fontsize=9)

# heatmap_filename = os.path.join(ZZ_path, "Entanglement plots/smoothed_bell_operator_heatmap_ZZ_small_24M.pdf")
# plt.savefig(heatmap_filename)
# plot_filename_png = os.path.join(ZZ_path, "Entanglement plots/smoothed_bell_operator_heatmap_ZZ_small_24M.png")
# plt.savefig(plot_filename_png)


# # Plot the 2D heatmap of Bell operator uncertainties
# plt.figure(figsize=(12, 10))
# plt.imshow(uncertainty_grid, origin='lower', extent=[0, 1, 200, 1000], aspect='auto', cmap='plasma')
# colorbar = plt.colorbar(label=r'$\sigma_{B}$', orientation='vertical')
# colorbar.ax.yaxis.label.set_fontsize(16)
# plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
# plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)

# # Add the value of the Bell operator uncertainty as a label to each square
# num_rows, num_cols = uncertainty_grid.shape
# x_centers = np.linspace(0.05, 0.85, num_cols)  # Center of bins for cos_psi
# y_centers = np.linspace(225.0, 975.0, num_rows)  # Center of bins for M_ZZ
# for i, y in enumerate(y_centers):
#     for j, x in enumerate(x_centers):
#         plt.text(x, y, f"{uncertainty_grid[i, j]:.2f}", color="white", ha="center", va="center", fontsize=9)

# heatmap_filename = os.path.join(ZZ_path, "Entanglement plots/bell_uncertainty_heatmap_ZZ_small_8M.pdf")
# plt.savefig(heatmap_filename)
# plot_filename_png = os.path.join(ZZ_path, "Entanglement plots/bell_uncertainty_heatmap_ZZ_small_8M.png")
# plt.savefig(plot_filename_png)

# Read bell operator value grid from the file
bell_value_grid = np.loadtxt(os.path.join(ZZ_path, "Entanglement plots/bell_operator_grid_ZZ_small_24M.txt"), delimiter=',').T
bell_value_grid = gaussian_filter(bell_value_grid, sigma=1.0)[:14, :]  # Apply Gaussian filter for smoothing


cos_psi_centers = np.arange(0.05, 0.9, 0.1)        
inv_mass_centers = np.arange(225.0, 900.0, 50.0)    

cos_psi_grid, inv_mass_grid = np.meshgrid(cos_psi_centers, inv_mass_centers)


# Plot the contour of Bell operator values
plt.figure(figsize=(12, 10))
# Define custom contour levels
custom_levels = np.arange(1.3, 2.3, step=0.1)
contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, bell_value_grid, levels=custom_levels, cmap='plasma')
contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, bell_value_grid, levels=custom_levels, colors='black', linewidths=0.7)
plt.clabel(contour_lines, inline=True, fontsize=12, fmt="%.2f")
colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{I}_3$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)
plt.yticks(np.arange(300, 900, 100), fontsize=12)
plt.xticks(np.arange(0.1, 0.9, 0.1), fontsize=12)
plot_filename = os.path.join(ZZ_save, "bell_operator_contour_ZZ_1604.pdf")
plt.savefig(plot_filename)

# Compute absolute discrepancy
discrepancy_grid = np.abs(bell_value_grid - bell_matrix)

# Plot the heatmap
plt.figure(figsize=(12, 10))
plt.imshow(discrepancy_grid, origin='lower', cmap='magma', aspect='auto')
cbar = plt.colorbar(label=r'$|\Delta \mathcal{I}_3|$', orientation='vertical')
cbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
plt.ylabel(r'$M_{ZZ} \, [\mathrm{GeV}]$', fontsize=16)
plt.title("Discrepancy between Grids", fontsize=18)

# Tick labels
plt.xticks(np.arange(9), labels=[f"{0.05 + 0.1*i:.2f}" for i in range(9)], fontsize=12)
plt.yticks(np.arange(14), labels=[f"{225 + 50*i:.0f}" for i in range(14)], fontsize=12)

# Add text annotations to each grid cell
num_rows, num_cols = discrepancy_grid.shape
x_centers = np.arange(num_cols)
y_centers = np.arange(num_rows)

for i in y_centers:
    for j in x_centers:
        plt.text(j, i, f"{discrepancy_grid[i, j]:.2f}", ha='center', va='center', color='white', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(ZZ_save, "bell_operator_discrepancy_heatmap_ZZ.pdf"))







# Load concurrence data
concurrence_grid = np.loadtxt(os.path.join(ZZ_path, "Entanglement plots/concurrence_grid_ZZ_test_24M.txt"), delimiter=',').T
smoothed_concurrence_grid = gaussian_filter(concurrence_grid, sigma=0.5)

cos_psi_grid = np.linspace(0.1, 0.7, 4)  # Midpoints of cos_psi bins
inv_mass_grid = np.linspace(250, 950, 8)  # Midpoints of mass bins

# Plot the contour of concurrence values
plt.figure(figsize=(10, 8))
contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, smoothed_concurrence_grid, cmap='viridis')
contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, smoothed_concurrence_grid, colors='black', linewidths=0.7)
plt.clabel(contour_lines, inline=True, fontsize=12, fmt="%.2f")
colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{C}_{LB}$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)
plot_filename = os.path.join(ZZ_path, "Entanglement plots/smooothed_concurrence_contour_ZZ_test_24M.pdf")
plt.savefig(plot_filename)
plot_filename = os.path.join(ZZ_path, "Entanglement plots/smooothed_concurrence_contour_ZZ_test_24M.png")
plt.savefig(plot_filename)

# Plot the 2D heatmap of concurrence values
plt.figure(figsize=(8, 6))
plt.imshow(concurrence_grid, origin='lower', extent=[0, 0.8, 200, 1000], aspect='auto', cmap='viridis')
colorbar = plt.colorbar(label=r'$\mathcal{C}_{LB}$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)   
plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)

# Add the value of the concurrence bound as a label to each square
num_rows, num_cols = concurrence_grid.shape
x_centers = np.linspace(0.1, 0.8, num_cols)  # Center of bins for cos_psi
y_centers = np.linspace(250.0, 950.0, num_rows)  # Center of bins for M_ZZ (updated for new range)
for i, y in enumerate(y_centers):
    for j, x in enumerate(x_centers):
        plt.text(x, y, f"{concurrence_grid[i, j]:.2f}", color="white", ha="center", va="center", fontsize=9)

heatmap_filename = os.path.join(ZZ_path, "Entanglement plots/concurrence_heatmap_ZZ_test_24M.pdf")
plt.savefig(heatmap_filename)
heatmap_filename = os.path.join(ZZ_path, "Entanglement plots/concurrence_heatmap_ZZ_test_24M.png")
plt.savefig(heatmap_filename)