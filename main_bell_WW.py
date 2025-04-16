import numpy as np
import os
from histo_plotter import read_data
from coefficient_calculator_WW import calculate_coefficients, read_masked_data, calculate_coefficients_fgh
from density_matrix_calculator import calculate_density_matrix_AC, O_bell_prime1, calculate_density_matrix_fgh, calculate_uncertainty_matrix_fgh
from Bell_inequality_optimizer import bell_inequality_optimization, inequality_function
from Unitary_Matrix import euler_unitary_matrix
from concurrence_bound import concurrence_lower, check_density_matrix
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

WW_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_WW_4l_final_process/Plots and data/organised_data"
WW_save = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_WW_4l_final_process/Plots and data/Plots"

regions = { 
        (i, j): [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for i in range(9)
        for j in range(20)
        for cos_min in [0.0 + 0.1 * i]
        for mass_min in [200.0 + 50.0 * j]
    }

# num_x_bins, num_y_bins = 180, 200
# event_count_grid = np.zeros((num_y_bins, num_x_bins))

# # Define bin edges for high-resolution mapping
# cos_psi_edges = np.linspace(0, 0.9, num_x_bins + 1)  # Bin edges in cos_psi
# inv_mass_edges = np.linspace(200, 1200, num_y_bins + 1)  # Bin edges in M_WW

# # Loop through each defined region and bin events
# for key, region in regions.items():
#     print("Calculating event count for region:", region)
#     save_dir = os.path.join(WW_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
    
#     if os.path.exists(save_dir):
#         print(f"Directory {save_dir} found.")
        
#         # Load event data
#         cos_psi_path = os.path.join(save_dir, "psi_data.txt")
#         WW_inv_path = os.path.join(save_dir, "WW_inv_mass.txt")

#         cos_psi_data = np.loadtxt(cos_psi_path)
#         WW_inv_mass = np.loadtxt(WW_inv_path)

#         # 2D histogram binning into 100x100 grid
#         hist2d, _, _ = np.histogram2d(WW_inv_mass, cos_psi_data, bins=[inv_mass_edges, cos_psi_edges])
        
#         # Accumulate event counts into the main grid
#         event_count_grid += hist2d

# # Plot the heatmap of event counts
# plt.figure(figsize=(12, 10))
# plt.imshow(event_count_grid, origin='lower', extent=[0, 0.9, 200, 1200], aspect='auto', cmap='inferno', vmin=0, vmax=2500)
# colorbar = plt.colorbar(label=r'Event Count', orientation='vertical')
# colorbar.ax.yaxis.label.set_fontsize(16)
# plt.xlabel(r'$\cos{\Theta}$', fontsize=16)   
# plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)

# # Save the heatmap
# heatmap_filename = os.path.join(WW_save, "event_count_heatmap_WW_180x200.pdf")
# plt.savefig(heatmap_filename)
# heatmap_filename = os.path.join(WW_save, "event_count_heatmap_WW_180x200.png")
# plt.savefig(heatmap_filename)


uniformity_grid = np.zeros((9, 20))  # To store uniformity scores

for (i, j), region in regions.items():
    print(f"Calculating uniformity for region: {region}...")

    save_dir = os.path.join(WW_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
    
    if os.path.exists(save_dir):
        cos_psi_path = os.path.join(save_dir, "psi_data.txt")
        WW_inv_path = os.path.join(save_dir, "WW_inv_mass.txt")

        cos_psi_data = np.loadtxt(cos_psi_path)
        WW_inv_mass = np.loadtxt(WW_inv_path)


        # Histogram within region: use 10x10 binning
        h, _, _ = np.histogram2d(WW_inv_mass, cos_psi_data, bins=[10, 10])
        mean = np.mean(h)
        std = np.std(h)

        # Avoid divide by zero
        if mean > 0:
            uniformity = 1.0 - (std / mean)
        else:
            uniformity = 0.0

        # Store the moment-based gradient in the uniformity grid
        uniformity_grid[i, j] = uniformity


# Save the uniformity scores as a .npy file
uniformity_grid = uniformity_grid.T  # Transpose to match the grid shape
np.save(os.path.join(WW_save, "uniformity_scores.npy"), uniformity_grid)

# # Read the uniformity scores from the .npy file
# uniformity_grid = np.load(os.path.join(WW_save, "uniformity_scores.npy"))


# Plot the heatmap of uniformity scores
plt.figure(figsize=(12, 10))
plt.imshow(uniformity_grid, origin='lower', extent=[0, 0.9, 200, 1200], aspect='auto', cmap='plasma')
colorbar = plt.colorbar(label='Uniformity Score', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
plt.yticks(np.arange(200, 1201, 100), fontsize=12)
plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=12)

# # Add the value of the uniformity score to each square
# num_rows, num_cols = uniformity_grid.shape
# x_centers = np.linspace(0.05, 0.85, num_cols)
# y_centers = np.linspace(225.0, 1175.0, num_rows)
# for i, y in enumerate(y_centers):
#     for j, x in enumerate(x_centers):
#         score = uniformity_grid[i, j]
#         plt.text(x, y, f"{score:.2f}", color="white", ha="center", va="center", fontsize=9)

# Save the plot
heatmap_filename = os.path.join(WW_save, "uniformity_heatmap_WW_type2.pdf")
plt.savefig(heatmap_filename)
heatmap_filename = os.path.join(WW_save, "uniformity_heatmap_WW_type2.png")
plt.savefig(heatmap_filename)




# # Calculate bell operator for single region
# region = [(0.0, 0.1), (550.0, 600.0)]
# save_dir = os.path.join(WW_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
# if os.path.exists(save_dir):
#     print(f"Directory {save_dir} found.")

# # Read theta and phi values for both datasets
# cos_theta_paths = {
#     1: os.path.join(save_dir, "e+_theta_data.txt"),
#     3: os.path.join(save_dir, "mu-_theta_data.txt")
# }
# phi_paths = {
#     1: os.path.join(save_dir, "e+_phi_data.txt"),
#     3: os.path.join(save_dir, "mu-_phi_data.txt")
# }
# cos_psi_path = os.path.join(save_dir, "psi_data.txt")
# WW_inv_path = os.path.join(save_dir, "WW_inv_mass.txt")
# cos_psi_data = np.loadtxt(cos_psi_path)
# WW_inv_mass = np.loadtxt(WW_inv_path)

# A_coefficients, C_coefficients, A_uncertainties, C_uncertainties = calculate_coefficients(cos_theta_paths, phi_paths)
# density_matrix, uncertainty_matrix_real, uncertainty_matrix_imag = calculate_density_matrix_AC(A_coefficients, C_coefficients, A_uncertainties, C_uncertainties)

# f_coefficients, g_coefficients, h_coefficients, f_uncs, g_uncs, h_uncs = calculate_coefficients_fgh(cos_theta_paths, phi_paths)
# density_matrix = calculate_density_matrix_fgh(f_coefficients, g_coefficients, h_coefficients)
# uncertainty_matrix = calculate_uncertainty_matrix_fgh(f_uncs, g_uncs, h_uncs)
# check_density_matrix(density_matrix)

# # Calculate the concurrence value for the region
# concurrence_val = concurrence_lower(density_matrix)
# print(f"Concurrence bound for region: {region} = {concurrence_val:.4g}")
# # Calculate the Bell operator value for the region
# bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
# print(f"Bell operator value for region: {region} = {bell_value:.4g}")

# bell_matrix = np.array([
#     [2.18, 2.15, 2.05, 2.05, 1.95, 1.8, 1.75, 1.7, 1.65],
#     [2.15, 2.15, 2.05, 2.05, 1.95, 1.8, 1.75, 1.7, 1.65],
#     [2.15, 2.15, 2.05, 2.05, 1.95, 1.8, 1.75, 1.7, 1.65],
#     [2.15, 2.15, 2.05, 2.05, 1.95, 1.8, 1.75, 1.7, 1.65],
#     [2.15, 2.05, 2.05, 2.05, 1.95, 1.8, 1.75, 1.7, 1.65],
#     [2.05, 2.05, 2.05, 2.05, 1.95, 1.75, 1.75, 1.7, 1.65],
#     [2.05, 2.05, 2.05, 1.99, 1.95, 1.75, 1.75, 1.7, 1.6],
#     [2.05, 2.05, 2.0, 1.99, 1.95, 1.75, 1.75, 1.7, 1.5],
#     [2.0, 2.0, 1.99, 1.95, 1.95, 1.8, 1.75, 1.7, 1.5],
#     [1.95, 1.95, 1.95, 1.9, 1.9, 1.8, 1.75, 1.7, 1.4],
#     [1.95, 1.95, 1.95, 1.85, 1.85, 1.75, 1.75, 1.7, 1.4],
#     [1.85, 1.85, 1.85, 1.8, 1.8, 1.65, 1.75, 1.7, 1.5],
#     [1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.75, 1.75, 1.5],
#     [1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.75, 1.75, 1.65]
# ])

# bell_matrix = bell_matrix[::-1]
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
# plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
# plt.yticks(np.arange(300, 900, 100), fontsize=12)
# plt.xticks(np.arange(0.1, 0.9, 0.1), fontsize=12)
# plot_filename = os.path.join(WW_save, "bell_operator_contour_WW_paper.pdf")
# plt.savefig(plot_filename)

# # Step 9: Plot the 2D heatmap of Bell operator values
# plt.figure(figsize=(12, 10))
# plt.imshow(bell_matrix, origin='lower', extent=[0, 0.9, 200, 900], aspect='auto', cmap='plasma')
# colorbar = plt.colorbar(label=r'$\mathcal{I}_3$', orientation='vertical')
# colorbar.ax.yaxis.label.set_fontsize(16)
# plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
# plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
# plt.yticks(np.arange(200, 901, 100), fontsize=12)
# plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=12)


# # Add the value of the Bell operator as a label to each square
# num_rows, num_cols = bell_matrix.shape
# x_centers = np.linspace(0.05, 0.85, num_cols)
# y_centers = np.linspace(225.0, 875.0, num_rows)
# for i, y in enumerate(y_centers):
#     for j, x in enumerate(x_centers):
#         plt.text(x, y, f"{bell_matrix[i, j]:.2f}", color="white", ha="center", va="center", fontsize=9)

# heatmap_filename = os.path.join(WW_save, "bell_operator_heatmap_WW_paper.pdf")
# plt.savefig(heatmap_filename)



# Generate mesh grid for cos_psi and WW_inv_mass based on regions (9 columns and 20 rows)
# Each cos_psi region is [cos_min, cos_min+0.1] so we take the center as cos_min + 0.05 for 0 <= cos_min < 0.9
# Each WW_inv_mass region is [mass_min, mass_min+50.0] so we take the center as mass_min + 25.0 for 200 <= mass_min < 1200
cos_psi_centers = np.arange(0.05, 0.9, 0.1)         # 9 centers: 0.05, 0.15, ..., 0.85
inv_mass_centers = np.arange(225.0, 1200.0, 50.0)     # 20 centers: 225, 275, ..., 1175

cos_psi_grid, inv_mass_grid = np.meshgrid(cos_psi_centers, inv_mass_centers)

# # Initialize the Bell operator, uncertainty, and concurrence grids for 20x9 regions
# bell_value_grid = np.zeros((9, 20))
# uncertainty_grid = np.zeros((9, 20))
# concurrence_grid = np.zeros((9, 20))
# optimal_params_grid = np.zeros((16, 9, 20))  # 3 parameters for each region

# for key, region in regions.items():
#     print("Calculating for region:", region)
#     save_dir = os.path.join(WW_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
#     if os.path.exists(save_dir):
#         print(f"Directory {save_dir} found.")
    
#     # Read theta and phi values for both datasets
#     theta_data = {
#         1: np.loadtxt(os.path.join(save_dir, "e+_theta_data.txt")),
#         3: np.loadtxt(os.path.join(save_dir, "mu-_theta_data.txt"))
#     }
#     phi_data = {
#         1: np.loadtxt(os.path.join(save_dir, "e+_phi_data.txt")),
#         3: np.loadtxt(os.path.join(save_dir, "mu-_phi_data.txt"))
#     }

#     cos_psi_path = os.path.join(save_dir, "psi_data.txt")
#     WW_inv_path = os.path.join(save_dir, "WW_inv_mass.txt")
#     cos_psi_data = np.loadtxt(cos_psi_path)
#     WW_inv_mass = np.loadtxt(WW_inv_path)

#     # A_coefficients, C_coefficients, A_uncertainties, C_uncertainties = calculate_coefficients(theta_data, phi_data)
#     # density_matrix, uncertainty_matrix_real, uncertainty_matrix_imag = calculate_density_matrix_AC(A_coefficients, C_coefficients, A_uncertainties, C_uncertainties)
#     f_coefficients, g_coefficients, h_coefficients, f_uncs, g_uncs, h_uncs = calculate_coefficients_fgh(theta_data, phi_data)
#     density_matrix = calculate_density_matrix_fgh(f_coefficients, g_coefficients, h_coefficients)
#     bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
#     concurrence_val = concurrence_lower(density_matrix)
#     print(f"Concurrence bound : {concurrence_val:.3g}")
#     concurrence_grid[key[0], key[1]] = concurrence_val
#     bell_value_grid[key] = bell_value
#     optimal_params_grid[:, key[0], key[1]] = optimal_params
#     print(f"Bell operator value : {bell_value:.3g}")


# # Save the Bell operator value grid to a file
# np.savetxt(os.path.join(WW_path, "bell_operator_grid_WW2.txt"), bell_value_grid, delimiter=',')

# # Save the concurrence value grid to a file
# np.savetxt(os.path.join(WW_path, "concurrence_grid_WW2.txt"), concurrence_grid, delimiter=',')

# # Save the optimal parameters grid to a npy file
# np.save(os.path.join(WW_path, "optimal_params_grid_WW2.npy"), optimal_params_grid)


# Read the Bell operator value grid from the file
bell_value_grid = np.loadtxt(os.path.join(WW_path, "bell_operator_grid_WW2.txt"), delimiter=',').T
bell_value_grid = gaussian_filter(bell_value_grid, sigma=0.5)[:14, :]  # Apply Gaussian filter for smoothing


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

# Compute absolute discrepancy
discrepancy_grid = np.abs(bell_value_grid - bell_matrix)

# Plot the heatmap
plt.figure(figsize=(12, 10))
plt.imshow(discrepancy_grid, origin='lower', cmap='magma', aspect='auto')
cbar = plt.colorbar(label=r'$|\Delta \mathcal{I}_3|$', orientation='vertical')
cbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
plt.ylabel(r'$M_{WW} \, [\mathrm{GeV}]$', fontsize=16)
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
plt.savefig(os.path.join(WW_save, "bell_operator_discrepancy_heatmap_annotated2.pdf"))


# Read the concurrence value grid from the file
concurrence_grid = np.loadtxt(os.path.join(WW_path, "concurrence_grid_WW2.txt"), delimiter=',').T

bell_value_grid = np.loadtxt(os.path.join(WW_path, "bell_operator_grid_WW2.txt"), delimiter=',').T
bell_value_grid = gaussian_filter(bell_value_grid, sigma=0.5)  # Apply Gaussian filter for smoothing

# Plot the contour of Bell operator values
plt.figure(figsize=(10, 8))
# Define custom contour levels
custom_levels = np.arange(1.2, 2.6, step=0.1)
contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, bell_value_grid, levels=custom_levels, cmap='Blues')
contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, bell_value_grid, levels=custom_levels, colors='black', linewidths=0.7)
plt.clabel(contour_lines, inline=True, fontsize=12, fmt="%.2f")
colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{I}_3$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
plt.yticks(np.arange(300, 1200, 100), fontsize=12)
plt.xticks(np.arange(0.1, 0.9, 0.1), fontsize=12)
plot_filename = os.path.join(WW_save, "bell_operator_contour_WW_smooth2.pdf")
plt.savefig(plot_filename)

# Step 9: Plot the 2D heatmap of Bell operator values
plt.figure(figsize=(8, 6))
plt.imshow(bell_value_grid, origin='lower', extent=[0, 0.9, 200, 1200], aspect='auto', cmap='plasma')
colorbar = plt.colorbar(label=r'$\mathcal{I}_3$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
plt.yticks(np.arange(200, 1201, 100), fontsize=12)
plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=12)


# Add the value of the Bell operator as a label to each square
num_rows, num_cols = bell_value_grid.shape
x_centers = np.linspace(0.05, 0.85, num_cols)
y_centers = np.linspace(225.0, 1175.0, num_rows)
for i, y in enumerate(y_centers):
    for j, x in enumerate(x_centers):
        plt.text(x, y, f"{bell_value_grid[i, j]:.2f}", color="white", ha="center", va="center", fontsize=9)

heatmap_filename = os.path.join(WW_save, "bell_operator_heatmap_WW_smooth2.pdf")
plt.savefig(heatmap_filename)

# Plot the contour of concurrence values
plt.figure(figsize=(10, 8))
custom_levels = np.arange(0.0, 0.7, step=0.1)
contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, concurrence_grid, levels=custom_levels, cmap='viridis')
contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, concurrence_grid, levels=custom_levels, colors='black', linewidths=0.7)
plt.clabel(contour_lines, inline=True, fontsize=12, fmt="%.2f")
colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{C}_{LB}$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
plt.yticks(np.arange(300, 1200, 100), fontsize=12)
plt.xticks(np.arange(0.1, 0.9, 0.1), fontsize=12)

plot_filename = os.path.join(WW_save, "concurrence_contour_WW2.pdf")
plt.savefig(plot_filename)

# Plot the 2D heatmap of concurrence values
plt.figure(figsize=(8, 6))
plt.imshow(concurrence_grid, origin='lower', extent=[0, 0.9, 200, 1200], aspect='auto', cmap='viridis')
colorbar = plt.colorbar(label=r'$\mathcal{C}_{LB}$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)   
plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
plt.yticks(np.arange(200, 1201, 100), fontsize=12)
plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=12)


# Add the value of the concurrence bound as a label to each square
num_rows, num_cols = concurrence_grid.shape
x_centers = np.linspace(0.05, 0.85, num_cols)
y_centers = np.linspace(225.0, 1175.0, num_rows)
for i, y in enumerate(y_centers):
    for j, x in enumerate(x_centers):
        plt.text(x, y, f"{concurrence_grid[i, j]:.2f}", color="white", ha="center", va="center", fontsize=9)

heatmap_filename = os.path.join(WW_save, "concurrence_heatmap_WW2.pdf")
plt.savefig(heatmap_filename)