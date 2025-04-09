import numpy as np
import os
from histo_plotter import read_data
from coefficient_calculator_ZZ import calculate_coefficients_AC, read_masked_data
from density_matrix_calculator import calculate_density_matrix_AC, O_bell_prime1, error_propagation_bell
from Bell_inequality_optimizer import bell_inequality_optimization, inequality_function
from Unitary_Matrix import euler_unitary_matrix
from concurrence_bound import concurrence_lower, check_density_matrix, concurrence_upper
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

ZZ_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_ZZ_SM/Plots and data/reorganised_data"

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

# # Calculate bell operator for single region
# region = regions[(7, 13)]
# save_dir = os.path.join(ZZ_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
# if os.path.exists(save_dir):
#     print(f"Directory {save_dir} found.")

# # Read theta and phi values for both datasets
# cos_theta_paths = {
#     1: os.path.join(save_dir, "e+_theta_data_combined_new.txt"),
#     3: os.path.join(save_dir, "mu+_theta_data_combined_new.txt")
# }
# phi_paths = {
#     1: os.path.join(save_dir, "e+_phi_data_combined_new.txt"),
#     3: os.path.join(save_dir, "mu+_phi_data_combined_new.txt")
# }
# cos_psi_path = os.path.join(save_dir, "psi_data_combined_new.txt")
# ZZ_inv_path = os.path.join(save_dir, "ZZ_inv_mass_combined_new.txt")
# cos_psi_data = np.loadtxt(cos_psi_path)
# ZZ_inv_mass = np.loadtxt(ZZ_inv_path)
# # Apply the mask to the data
# mask_region = [(0.7, 0.75), (875, 900)]
# mask = read_masked_data(cos_psi_data, ZZ_inv_mass, mask_region[0], mask_region[1])

# A_coefficients, C_coefficients, A_uncertainties, C_uncertainties = calculate_coefficients_AC(cos_theta_paths, phi_paths, mask)
# density_matrix, uncertainty_matrix_real, uncertainty_matrix_imag = calculate_density_matrix_AC(A_coefficients, C_coefficients, A_uncertainties, C_uncertainties)
# bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
# print(f"Bell operator value for region: {mask_region} = {bell_value}")


# Initialize the Bell operator value grid
# bell_value_grid = np.zeros((9, 16))
# uncertainty_grid = np.zeros((9, 16))
# optimal_params_grid = np.zeros((16, 9, 16))
# concurrence_grid = np.zeros((5, 8))


# Process merged regions for cos_ψ from 0.0 to 0.8 (each merged block is 0.2 in cos_ψ and 100 GeV in mass)
# Original directories are defined in 50 GeV (mass) × 0.1 (cos_ψ) bins.
# To form a merged region of 0.2 × 100 GeV, we merge four subregions.
# merged_concurrence_grid = np.zeros((4, 8))
# for i in range(4):  # cos_ψ bins: [0.0,0.2], [0.2,0.4], [0.4,0.6], [0.6,0.8]
#     cos_min = 0.0 + 0.2 * i
#     for j in range(8):  # mass bins: [200,300], [300,400], ..., [900,1000]
#         mass_min = 200.0 + 100.0 * j
#         print("Calculating concurrence bound for merged region:",
#               f"cos_ψ in [{cos_min}, {cos_min+0.2}], mass in [{mass_min}, {mass_min+100}]")
        
#         # Initialize lists to collect data from the 4 subregions in cos_ψ and mass
#         psi_data_list = []
#         ZZ_inv_list = []
#         theta_data_dict = {1: [], 3: []}
#         phi_data_dict = {1: [], 3: []}
        
#         # Loop over the two cos_ψ sub-bins and two mass sub-bins
#         for dcos in [0, 0.1]:
#             for dmass in [0, 50]:
#                 sub_cos_min = cos_min + dcos
#                 sub_mass_min = mass_min + dmass
#                 sub_dir = os.path.join(ZZ_path, f"cos_psi_{sub_cos_min}_{sub_cos_min+0.1}_inv_mass_{sub_mass_min}_{sub_mass_min+50.0}")
#                 if not os.path.exists(sub_dir):
#                     print(f"Skipping missing subregion: {sub_dir}")
#                     continue
#                 psi_data_list.append(np.loadtxt(os.path.join(sub_dir, "psi_data_combined_new.txt")))
#                 ZZ_inv_list.append(np.loadtxt(os.path.join(sub_dir, "ZZ_inv_mass_combined_new.txt")))
#                 theta_data_dict[1].append(np.loadtxt(os.path.join(sub_dir, "e+_theta_data_combined_new.txt")))
#                 theta_data_dict[3].append(np.loadtxt(os.path.join(sub_dir, "mu+_theta_data_combined_new.txt")))
#                 phi_data_dict[1].append(np.loadtxt(os.path.join(sub_dir, "e+_phi_data_combined_new.txt")))
#                 phi_data_dict[3].append(np.loadtxt(os.path.join(sub_dir, "mu+_phi_data_combined_new.txt")))
        
#         if len(psi_data_list) == 0:
#             print(f"No data found for merged region cos_ψ [{cos_min}, {cos_min+0.2}], mass [{mass_min}, {mass_min+100}]")
#             continue
        
#         # Merge the data from all available subregions
#         psi_data_merged = np.concatenate(psi_data_list)
#         ZZ_inv_merged = np.concatenate(ZZ_inv_list)
#         merged_theta_data = {key: np.concatenate(theta_data_dict[key]) for key in theta_data_dict}
#         merged_phi_data = {key: np.concatenate(phi_data_dict[key]) for key in phi_data_dict}
        
#         # Compute coefficients, density matrix and concurrence bound
#         A_coefficients, C_coefficients, A_uncertainties, C_uncertainties = calculate_coefficients_AC(merged_theta_data, merged_phi_data)
#         density_matrix, uncertainty_matrix_real, uncertainty_matrix_imag = calculate_density_matrix_AC(
#             A_coefficients, C_coefficients, A_uncertainties, C_uncertainties)
#         concurrence_val = concurrence_lower(density_matrix, uncertainty_matrix_real, uncertainty_matrix_imag)
#         print(f"Concurrence bound: {concurrence_val}")
        
#         merged_concurrence_grid[i, j] = concurrence_val

# # Process two extra regions for cos_ψ in (0.8,0.9)
# def merge_for_extra_region(cos_range, mass_range):
#     cos_min, cos_max = cos_range
#     mass_low, mass_high = mass_range
#     psi_data_list = []
#     ZZ_inv_list = []
#     theta_data_dict = {1: [], 3: []}
#     phi_data_dict = {1: [], 3: []}
#     # Iterate in steps of 50 GeV over the mass range
#     for m_low in np.arange(mass_low, mass_high, 50.0):
#         save_dir = os.path.join(ZZ_path, f"cos_psi_{cos_min}_{cos_max}_inv_mass_{m_low}_{m_low+50.0}")
#         if not os.path.exists(save_dir):
#             print(f"Directory {save_dir} missing; skipping current file group")
#             continue
#         psi_data_list.append(np.loadtxt(os.path.join(save_dir, "psi_data_combined_new.txt")))
#         ZZ_inv_list.append(np.loadtxt(os.path.join(save_dir, "ZZ_inv_mass_combined_new.txt")))
#         theta_data_dict[1].append(np.loadtxt(os.path.join(save_dir, "e+_theta_data_combined_new.txt")))
#         theta_data_dict[3].append(np.loadtxt(os.path.join(save_dir, "mu+_theta_data_combined_new.txt")))
#         phi_data_dict[1].append(np.loadtxt(os.path.join(save_dir, "e+_phi_data_combined_new.txt")))
#         phi_data_dict[3].append(np.loadtxt(os.path.join(save_dir, "mu+_phi_data_combined_new.txt")))
#     if len(psi_data_list) == 0:
#         print(f"No data found for extra region: cos_ψ {cos_range}, mass {mass_range}")
#         return None
#     psi_data_merged = np.concatenate(psi_data_list)
#     ZZ_inv_merged = np.concatenate(ZZ_inv_list)
#     merged_theta_data = {key: np.concatenate(theta_data_dict[key]) for key in theta_data_dict}
#     merged_phi_data = {key: np.concatenate(phi_data_dict[key]) for key in phi_data_dict}
    
#     A_coefficients, C_coefficients, A_uncertainties, C_uncertainties = calculate_coefficients_AC(merged_theta_data, merged_phi_data)
#     density_matrix, uncertainty_matrix_real, uncertainty_matrix_imag = calculate_density_matrix_AC(
#         A_coefficients, C_coefficients, A_uncertainties, C_uncertainties)
#     return concurrence_lower(density_matrix, uncertainty_matrix_real, uncertainty_matrix_imag)

# print("Calculating concurrence bound for extra region: cos_ψ in [0.8,0.9], mass in [300,1000]")
# extra1 = merge_for_extra_region((0.8, 0.9), (300.0, 1000.0))
# if extra1 is not None:
#     print(f"Concurrence bound for extra region [0.8,0.9] & [300,1000]: {extra1}")

# print("Calculating concurrence bound for extra region: cos_ψ in [0.8,0.9], mass in [200,300]")
# extra2 = merge_for_extra_region((0.8, 0.9), (200.0, 300.0))
# if extra2 is not None:
#     print(f"Concurrence bound for extra region [0.8,0.9] & [200,300]: {extra2}")

# print("Concurrence grid:")
# for row in merged_concurrence_grid:
#     print(" ".join(f"{val:.4f}" for val in row))

# for key, region in regions.items():
#     print("Calculating concurrence bound for region:", region)
#     save_dir = os.path.join(ZZ_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
#     if os.path.exists(save_dir):
#         print(f"Directory {save_dir} found.")
    
#     # Read theta and phi values for both datasets
#     theta_data = {
#         1: np.loadtxt(os.path.join(save_dir, "e+_theta_data_combined_new.txt")),
#         3: np.loadtxt(os.path.join(save_dir, "mu+_theta_data_combined_new.txt"))
#     }
#     phi_data = {
#         1: np.loadtxt(os.path.join(save_dir, "e+_phi_data_combined_new.txt")),
#         3: np.loadtxt(os.path.join(save_dir, "mu+_phi_data_combined_new.txt"))
#     }

#     cos_psi_path = os.path.join(save_dir, "psi_data_combined_new.txt")
#     ZZ_inv_path = os.path.join(save_dir, "ZZ_inv_mass_combined_new.txt")
#     cos_psi_data = np.loadtxt(cos_psi_path)
#     ZZ_inv_mass = np.loadtxt(ZZ_inv_path)

#     A_coefficients, C_coefficients, A_uncertainties, C_uncertainties = calculate_coefficients_AC(theta_data, phi_data)
#     density_matrix, uncertainty_matrix_real, uncertainty_matrix_imag = calculate_density_matrix_AC(A_coefficients, C_coefficients, A_uncertainties, C_uncertainties)
#     # bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
#     concurrence_val = concurrence_lower(density_matrix, uncertainty_matrix_real, uncertainty_matrix_imag)
#     print(f"Concurrence bound : {concurrence_val:.3g}")
#     concurrence_grid[key[0], key[1]] = concurrence_val
#     # bell_value_grid[key] = bell_value
#     # optimal_params_grid[:, key[0], key[1]] = optimal_params
#     # print(f"Bell operator value for region: {region} = {bell_value}")

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