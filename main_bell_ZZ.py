import numpy as np
import os
from histo_plotter import read_data
from coefficient_calculator_ZZ import calculate_coefficients_AC, read_masked_data
from density_matrix_calculator import calculate_density_matrix_AC, O_bell_prime1, error_propagation_bell
from Bell_inequality_optimizer import bell_inequality_optimization, inequality_function
from Unitary_Matrix import euler_unitary_matrix
from concurrence_bound import concurrence_lower, check_density_matrix, concurrence_upper
import matplotlib.pyplot as plt

ZZ_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_ZZ_SM/Plots and data"
# Read theta and phi values for both datasets
cos_theta_paths = {
    1: os.path.join(ZZ_path, "mu+/theta_data_combined_new.txt"),
    3: os.path.join(ZZ_path, "e+/theta_data_combined_new.txt")
}
phi_paths = {
    1: os.path.join(ZZ_path, "mu+/phi_data_combined_new.txt"),
    3: os.path.join(ZZ_path, "e+/phi_data_combined_new.txt")
}

cos_psi_data = read_data(os.path.join(ZZ_path, "psi_data_combined_new.txt"))
ZZ_inv_mass = read_data(os.path.join(ZZ_path, "ZZ_inv_mass_combined_new.txt"))

cos_psi_region = (0.1, 0.2)
ZZ_inv_mass_region = (250, 300)

# # Use np.histogram2d to count the number of events in each bin
# num_bins_cos_psi = 100
# num_bins_inv_mass = 100
# event_counts, x_edges, y_edges = np.histogram2d(cos_psi_data, ZZ_inv_mass, 
#                                                 bins=[num_bins_cos_psi, num_bins_inv_mass], 
#                                                 range=[[0, 1], [200, 1000]])
# plt.figure(figsize=(8, 6))
# plt.imshow(event_counts.T, origin='lower', extent=[0, 1, 200, 1000], aspect='auto', cmap='hot', vmin=0, vmax=2500)
# plt.colorbar(label='Number of events')
# plt.xlabel(r'$\cos{\Psi}$')
# plt.ylabel(r'$M_{ZZ} (GeV) $')

# # Save the event count heatmap to the ZZ_path directory
# event_plot_filename = os.path.join(ZZ_path, "Entanglement plots/event_count_heatmap_combined_new_filtered.pdf")
# plt.savefig(event_plot_filename)

# # Find coefficients for masked phase space
# mask = read_masked_data(cos_psi_data, ZZ_inv_mass, cos_psi_region, ZZ_inv_mass_region)
# A_coefficients, C_coefficients, A_uncertainties, C_uncertainties = calculate_coefficients_AC(cos_theta_paths, phi_paths, mask)
# density_matrix, uncertainty_matrix = calculate_density_matrix_AC(A_coefficients, C_coefficients, A_uncertainties, C_uncertainties)

# # Calculate Bell operator for masked space
# masked_bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
# U_params = optimal_params[:8]
# V_params = optimal_params[8:]
# U = euler_unitary_matrix(*U_params)
# V = euler_unitary_matrix(*V_params) 
# U_cross_V = np.kron(U, V)
# O_bell = U_cross_V.conj().T @ O_bell_prime1 @ U_cross_V
# bell_value_uncertainty = np.sqrt(np.real(np.trace((np.real(uncertainty_matrix)**2 @ O_bell**2))))
# print(f"Maximized Bell inequality value for region:\n Cos(Psi) = {cos_psi_region}, M_ZZ = {ZZ_inv_mass_region}\n I_3 = {masked_bell_value} +- {bell_value_uncertainty}")

# # Calculate concurrence for masked space
# concurrence_value, concurrence_uncertainty = concurrence_lower(density_matrix, uncertainty_matrix)
# print(f"Concurrence bound for region:\n Cos(Psi) = {cos_psi_region}, M_ZZ = {ZZ_inv_mass_region}\n Concurrence = {concurrence_value} +- {concurrence_uncertainty}")


# Generate mesh grid for cos_psi and ZZ_inv_mass
cos_psi_range = np.arange(0.0, 1.2, step=0.2) 
inv_mass_range = np.arange(200.0, 1100.0, step=100.0)

cos_psi_contour = np.arange(0.1, 1.1, step=0.2)
inv_mass_contour = np.arange(250.0, 1050.0, step=100.0) 

# cos_psi_range = np.arange(0.0, 1.1, step=0.1) 
# inv_mass_range = np.arange(200.0, 1050.0, step=50.0)

# cos_psi_contour = np.arange(0.05, 1.05, step=0.1)
# inv_mass_contour = np.arange(225.0, 1025.0, step=50.0) 

cos_psi_grid, inv_mass_grid = np.meshgrid(cos_psi_contour, inv_mass_contour)

# # Initialize the Bell operator value grid
# bell_value_grid = np.zeros((8, 5))
# uncertainty_grid = np.zeros((8, 5))
# optimal_params_grid = np.zeros((16, 8, 5))

# concurrence_grid = np.zeros((8, 5))
# concurrence_unc_grid = np.zeros((8, 5))

# # Calculate Bell operator value for each point in the mesh grid
# for i in range(len(inv_mass_range)-1):
#     for j in range(len(cos_psi_range)-1):
#         # Apply a mask for the current region in the mesh grid
#         mask = read_masked_data(cos_psi_data, ZZ_inv_mass, 
#                                 (cos_psi_range[j], cos_psi_range[j+1]), 
#                                 (inv_mass_range[i], inv_mass_range[i+1]))
#         # Calculate coefficients for this masked region
#         print(f"Calculating coefficients for region: cos_psi = ({cos_psi_range[j]:.1f}, {cos_psi_range[j+1]:.1f}), M_ZZ = ({inv_mass_range[i]:.1f}, {inv_mass_range[i+1]:.1f})")
#         A_coefficients, C_coefficients, A_uncertainties, C_uncertainties = calculate_coefficients_AC(cos_theta_paths, phi_paths, mask)
#         # Calculate the density matrix for this region
#         density_matrix, uncertainty_matrix = calculate_density_matrix_AC(A_coefficients, C_coefficients, A_uncertainties, C_uncertainties)
#         # Calculate the Bell operator value for this region
#         bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
#         bell_value_grid[i, j] = bell_value
#         # Reconstruct to do uncertainty calculation
#         U_params = optimal_params[:8]
#         V_params = optimal_params[8:]
#         U = euler_unitary_matrix(*U_params)
#         V = euler_unitary_matrix(*V_params)
#         U_cross_V = np.kron(U, V)
#         O_bell = U_cross_V.conj().T @ O_bell_prime1 @ U_cross_V
#         bell_value_uncertainty = np.sqrt(np.real(np.trace((np.real(uncertainty_matrix)**2 @ O_bell**2))))
#         uncertainty_grid[i, j] = bell_value_uncertainty
#         # Calculate concurrence for this region
#         concurrence_value, concurrence_uncertainty = concurrence_lower(density_matrix, uncertainty_matrix)
#         concurrence_grid[i, j] = concurrence_value
#         concurrence_unc_grid[i, j] = concurrence_uncertainty
#         # Save the optimal parameters for this region
#         optimal_params_grid[:, i, j] = optimal_params

# # Save the Bell operator value grid to a file
# np.savetxt(os.path.join(ZZ_path, "Entanglement plots/bell_operator_grid_ZZ_big.txt"), bell_value_grid, delimiter=',')

# # Save the uncertainty grid to a file
# np.savetxt(os.path.join(ZZ_path, "Entanglement plots/uncertainty_grid_ZZ_big.txt"), uncertainty_grid, delimiter=',')

# # Save the concurrence value grid to a file
# np.savetxt(os.path.join(ZZ_path, "Entanglement plots/concurrence_grid_ZZ_big.txt"), concurrence_grid, delimiter=',')

# # Save the concurrence uncertainty grid to a file
# np.savetxt(os.path.join(ZZ_path, "Entanglement plots/concurrence_unc_grid_ZZ_big.txt"), concurrence_unc_grid, delimiter=',')

# # Save the optimal parameters grid to a file
# np.save(os.path.join(ZZ_path, "Entanglement plots/optimal_params_grid_ZZ_big.npy"), optimal_params_grid)


# Read the Bell operator value grid from the file
bell_value_grid = np.loadtxt(os.path.join(ZZ_path, "Entanglement plots/bell_operator_grid_ZZ_big.txt"), delimiter=',')
# print("Average Bell operator value:", np.mean(bell_value_grid))

# Read the concurrence value grid from the file
concurrence_grid = np.loadtxt(os.path.join(ZZ_path, "Entanglement plots/concurrence_grid_ZZ_big.txt"), delimiter=',')

# # Read the concurrence uncertainty grid from the file
# concurrence_unc_grid = np.loadtxt(os.path.join(ZZ_path, "Entanglement plots/concurrence_unc_grid_ZZ_big.txt"), delimiter=',')

# Read the uncertainty grid from the file
uncertainty_grid = np.loadtxt(os.path.join(ZZ_path, "Entanglement plots/uncertainty_grid_ZZ_big.txt"), delimiter=',')

# Read the optimal parameters grid from the file
optimal_params_grid = np.load(os.path.join(ZZ_path, "Entanglement plots/optimal_params_grid_ZZ_big.npy"))

# # Calculate Bell operator value for each point in the mesh grid
# for i in range(len(inv_mass_range)-1):
#     for j in range(len(cos_psi_range)-1):
#         print(f"Calculating uncertainty for region: cos_psi = ({cos_psi_range[j]:.1f}, {cos_psi_range[j+1]:.1f}), M_ZZ = ({inv_mass_range[i]:.1f}, {inv_mass_range[i+1]:.1f})")
#         # Apply a mask for the current region in the mesh grid
#         mask = read_masked_data(cos_psi_data, ZZ_inv_mass, 
#                                 (cos_psi_range[j], cos_psi_range[j+1]), 
#                                 (inv_mass_range[i], inv_mass_range[i+1]))
#         # Calculate coefficients for this masked region
#         A_coefficients, C_coefficients, A_uncertainties, C_uncertainties = calculate_coefficients_AC(cos_theta_paths, phi_paths, mask)
#         # Calculate the density matrix for this region
#         density_matrix, uncertainty_matrix_real, uncertainty_matrix_imag = calculate_density_matrix_AC(A_coefficients, C_coefficients, A_uncertainties, C_uncertainties)
#         # Reconstruct to do uncertainty calculation
#         optimal_params = optimal_params_grid[:, i, j]
#         U_params = optimal_params[:8]
#         V_params = optimal_params[8:]
#         U = euler_unitary_matrix(*U_params)
#         V = euler_unitary_matrix(*V_params)
#         U_cross_V = np.kron(U, V)
#         O_bell = U_cross_V.conj().T @ O_bell_prime1 @ U_cross_V
#         bell_value_uncertainty = error_propagation_bell(O_bell, uncertainty_matrix_real, uncertainty_matrix_imag)
#         print(bell_value_uncertainty)
#         uncertainty_grid[i, j] = bell_value_uncertainty
#         print(inequality_function(density_matrix, O_bell_prime1, optimal_params))

# Calculate averages of combined subregions for Bell values, concurrence values, and uncertainties

# print("Uncertainty grid:")
# for row in uncertainty_grid:
#     print(" ".join(f"{val:.4f}" for val in row))

# # Save the uncertainty grid to a file
# np.savetxt(os.path.join(ZZ_path, "Entanglement plots/uncertainty_grid_ZZ_big.txt"), uncertainty_grid, delimiter=',')


# bell_value_grid_avgs = np.empty((8, 5))
# concurrence_grid_avgs = np.empty((8, 5))
# uncertainty_grid_avgs = np.empty((8, 5))

# for i in range(8):
#     for j in range(5):
#         # Take the average of a 2x2 block for Bell values
#         bell_value_grid_avgs[i, j] = np.mean(bell_value_grid[i*2:(i+1)*2, j*2:(j+1)*2])
#         # Take the average of a 2x2 block for concurrence values
#         concurrence_grid_avgs[i, j] = np.mean(concurrence_grid[i*2:(i+1)*2, j*2:(j+1)*2])
#         # Take the average of a 2x2 block for uncertainties
#         uncertainty_grid_avgs[i, j] = np.sqrt(np.sum(uncertainty_grid[i*2:(i+1)*2, j*2:(j+1)*2]**2) / 4) 


# Plot the contour of Bell operator values
plt.figure(figsize=(10, 8))
# Define custom contour levels
custom_levels = np.arange(0.8, 2.5, step=0.1)
contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, bell_value_grid, levels=custom_levels, cmap='Blues')
contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, bell_value_grid, levels=custom_levels, colors='black', linewidths=0.7)
plt.clabel(contour_lines, inline=True, fontsize=12, fmt="%.2f")
colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{I}_3$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)
plot_filename = os.path.join(ZZ_path, "Entanglement plots/bell_operator_contour_ZZ_big.pdf")
plt.savefig(plot_filename)
plot_filename_png = os.path.join(ZZ_path, "Entanglement plots/bell_operator_contour_ZZ_big.png")
plt.savefig(plot_filename_png)

# Plot the 2D heatmap of Bell operator values
plt.figure(figsize=(8, 6))
plt.imshow(bell_value_grid, origin='lower', extent=[0, 1, 200, 1000], aspect='auto', cmap='plasma')
colorbar = plt.colorbar(label=r'$\mathcal{I}_3$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)

# Add the value of the Bell operator as a label to each square
num_rows, num_cols = bell_value_grid.shape
x_centers = np.linspace(0.1, 0.9, num_cols)  # Center of bins for cos_psi
y_centers = np.linspace(250.0, 950.0, num_rows)  # Center of bins for M_ZZ
for i, y in enumerate(y_centers):
    for j, x in enumerate(x_centers):
        plt.text(x, y, f"{bell_value_grid[i, j]:.2f}", color="white", ha="center", va="center", fontsize=9)

heatmap_filename = os.path.join(ZZ_path, "Entanglement plots/bell_operator_heatmap_ZZ_big.pdf")
plt.savefig(heatmap_filename)
plot_filename_png = os.path.join(ZZ_path, "Entanglement plots/bell_operator_heatmap_ZZ_big.png")
plt.savefig(plot_filename_png)

# Plot the 2D heatmap of Bell operator uncertainties
plt.figure(figsize=(12, 10))
plt.imshow(uncertainty_grid, origin='lower', extent=[0, 1, 200, 1000], aspect='auto', cmap='plasma')
colorbar = plt.colorbar(label=r'$\sigma_{B}$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)

# Add the value of the Bell operator uncertainty as a label to each square
num_rows, num_cols = uncertainty_grid.shape
x_centers = np.linspace(0.1, 0.9, num_cols)  # Center of bins for cos_psi
y_centers = np.linspace(250.0, 950.0, num_rows)  # Center of bins for M_ZZ
for i, y in enumerate(y_centers):
    for j, x in enumerate(x_centers):
        plt.text(x, y, f"{uncertainty_grid[i, j]:.2f}", color="white", ha="center", va="center", fontsize=9)

heatmap_filename = os.path.join(ZZ_path, "Entanglement plots/bell_uncertainty_heatmap_ZZ_big.pdf")
plt.savefig(heatmap_filename)
plot_filename_png = os.path.join(ZZ_path, "Entanglement plots/bell_uncertainty_heatmap_ZZ_big.png")
plt.savefig(plot_filename_png)


# Plot the contour of concurrence values
plt.figure(figsize=(10, 8))
contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, concurrence_grid, cmap='viridis')
contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, concurrence_grid, colors='black', linewidths=0.7)
plt.clabel(contour_lines, inline=True, fontsize=12, fmt="%.2f")
colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{C}_{LB}$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)
plot_filename = os.path.join(ZZ_path, "Entanglement plots/concurrence_contour_ZZ_big.pdf")
plt.savefig(plot_filename)
plot_filename = os.path.join(ZZ_path, "Entanglement plots/concurrence_contour_ZZ_big.png")
plt.savefig(plot_filename)

# Plot the 2D heatmap of concurrence values
plt.figure(figsize=(8, 6))
plt.imshow(concurrence_grid, origin='lower', extent=[0, 1, 200, 1000], aspect='auto', cmap='viridis')
colorbar = plt.colorbar(label=r'$\mathcal{C}_{LB}$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)   
plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)

# Add the value of the concurrence bound as a label to each square
num_rows, num_cols = concurrence_grid.shape
x_centers = np.linspace(0.1, 0.9, num_cols)  # Center of bins for cos_psi
y_centers = np.linspace(250.0, 950.0, num_rows)  # Center of bins for M_ZZ
for i, y in enumerate(y_centers):
    for j, x in enumerate(x_centers):
        plt.text(x, y, f"{concurrence_grid[i, j]:.2f}", color="white", ha="center", va="center", fontsize=9)

heatmap_filename = os.path.join(ZZ_path, "Entanglement plots/concurrence_heatmap_ZZ_big.pdf")
plt.savefig(heatmap_filename)
heatmap_filename = os.path.join(ZZ_path, "Entanglement plots/concurrence_heatmap_ZZ_big.png")
plt.savefig(heatmap_filename)