import numpy as np
import os
from histo_plotter import read_data
from coefficient_calculator_WW import calculate_coefficients, read_masked_data
from density_matrix_calculator import calculate_density_matrix_AC, O_bell_prime1
from Bell_inequality_optimizer import bell_inequality_optimization, inequality_function
from Unitary_Matrix import euler_unitary_matrix
from concurrence_bound import concurrence_lower, concurrence_upper 
import matplotlib.pyplot as plt

WW_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_WW_SM/Plots and data"
# Read theta and phi values for both datasets
cos_theta_paths = {
    1: os.path.join(WW_path, "mu-/theta_data_combined.txt"),
    3: os.path.join(WW_path, "e+/theta_data_combined.txt")
}
phi_paths = {
    1: os.path.join(WW_path, "mu-/phi_data_combined.txt"),
    3: os.path.join(WW_path, "e+/phi_data_combined.txt")
}

cos_psi_data = read_data(os.path.join(WW_path, "psi_data_combined.txt"))
WW_inv_mass = read_data(os.path.join(WW_path, "WW_inv_mass_combined.txt"))

cos_psi_region = (0.0, 0.2)
WW_inv_mass_region = (250, 300)

# Use np.histogram2d to count the number of events in each bin
num_bins_cos_psi = 100
num_bins_inv_mass = 100
event_counts, x_edges, y_edges = np.histogram2d(cos_psi_data, WW_inv_mass, 
                                                bins=[num_bins_cos_psi, num_bins_inv_mass], 
                                                range=[[0, 1], [200, 1000]])
plt.figure(figsize=(8, 6))
plt.imshow(event_counts.T, origin='lower', extent=[0, 1, 200, 1000], aspect='auto', cmap='hot', vmin=0, vmax=2500)
plt.colorbar(label='Number of events')
plt.xlabel(r'$\cos{\Psi}$')
plt.ylabel(r'$M_{WW} (GeV) $')

# Save the event count heatmap to the WW_path directory
event_plot_filename = os.path.join(WW_path, "event_count_heatmap_combined.pdf")
plt.savefig(event_plot_filename)

# Find coefficients for masked phase space
mask = read_masked_data(cos_psi_data, WW_inv_mass, cos_psi_region, WW_inv_mass_region)
A_coefficients, C_coefficients, A_uncertainties, C_uncertainties = calculate_coefficients(cos_theta_paths, phi_paths, mask)
density_matrix, uncertainty_matrix = calculate_density_matrix_AC(A_coefficients, C_coefficients, A_uncertainties, C_uncertainties)

# Calculate concurrence bound for masked space
concurrence_value = concurrence_lower(density_matrix)
print(f"Concurrence bound for region:\n Cos(Psi) = {cos_psi_region}, M_WW = {WW_inv_mass_region}\n Concurrence = {concurrence_value}")

# Calculate Bell operator for masked space
masked_bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
U_params = optimal_params[:8]
V_params = optimal_params[8:]
U = euler_unitary_matrix(*U_params)
V = euler_unitary_matrix(*V_params) 
U_cross_V = np.kron(U, V)
uncertainty_matrix = U_cross_V.conj().T**2 @ uncertainty_matrix**2 @ U_cross_V**2
uncertainty_matrix = np.sqrt(np.abs(np.real(uncertainty_matrix)))
bell_value_uncertainty = np.sqrt(np.real(np.trace((uncertainty_matrix**2 @ O_bell_prime1**2))))
print(f"Maximized Bell inequality value for region:\n Cos(Psi) = {cos_psi_region}, M_WW = {WW_inv_mass_region}\n I_3 = {masked_bell_value} +- {bell_value_uncertainty}")


# Generate mesh grid for cos_psi and WW_inv_mass
cos_psi_range = np.arange(1.0, 1.2, step=0.2) 
inv_mass_range = np.arange(200.0, 1100.0, step=100.0) 

cos_psi_contour = np.arange(0.1, 1.1, step=0.2)   
inv_mass_contour = np.arange(250.0, 1050.0, step=100.0) 

cos_psi_grid, inv_mass_grid = np.meshgrid(cos_psi_contour, inv_mass_contour)

# Step 8: Initialize the Bell operator value grid
bell_value_grid = np.zeros((8,5))
uncertainty_grid = np.zeros((8,5))
concurrence_grid = np.zeros((8,5))


# Calculate Bell operator value for each point in the mesh grid
for i in range(len(inv_mass_range)-1):
    for j in range(len(cos_psi_range)-1):
        # Apply a mask for the current region in the mesh grid
        mask = read_masked_data(cos_psi_data, WW_inv_mass, 
                                (cos_psi_range[j], cos_psi_range[j+1]), 
                                (inv_mass_range[i], inv_mass_range[i+1]))
        # Calculate coefficients for this masked region
        print(f"Calculating coefficients for region: cos_psi = ({cos_psi_range[j]:.1f}, {cos_psi_range[j+1]:.1f}), M_WW = ({inv_mass_range[i]:.1f}, {inv_mass_range[i+1]:.1f})")
        A_coefficients, C_coefficients, A_uncertainties, C_uncertainties = calculate_coefficients(cos_theta_paths, phi_paths, mask)
        density_matrix, uncertainty_matrix = calculate_density_matrix_AC(A_coefficients, C_coefficients, A_uncertainties, C_uncertainties)
        bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
        bell_value_grid[i, j] = bell_value
        concurrence_value = concurrence_lower(density_matrix)
        concurrence_grid[i, j] = concurrence_value


# Save the Bell operator value grid to a file
np.savetxt(os.path.join(WW_path, "bell_operator_grid_WW.txt"), bell_value_grid, delimiter=',')

# Save the concurrence value grid to a file
np.savetxt(os.path.join(WW_path, "concurrence_grid_WW.txt"), concurrence_grid, delimiter=',')

# # Read the Bell operator value grid from the file
# bell_value_grid = np.loadtxt(os.path.join(WW_path, "bell_operator_grid_WW.txt"), delimiter=',')
# print("Average Bell operator value:", np.mean(bell_value_grid))

# # Read the concurrence value grid from the file
# concurrence_grid = np.loadtxt(os.path.join(WW_path, "concurrence_grid_WW.txt"), delimiter=',')

# Plot the contour of Bell operator values
plt.figure(figsize=(10, 8))
# Define custom contour levels
custom_levels = np.arange(0.8, 2.6, step=0.2)
contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, bell_value_grid, levels=custom_levels, cmap='Blues')
contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, bell_value_grid, levels=custom_levels, colors='black', linewidths=0.7)
plt.clabel(contour_lines, inline=True, fontsize=12, fmt="%.2f")
colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{I}_3$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
plot_filename = os.path.join(WW_path, "bell_operator_contour_WW.pdf")
plt.savefig(plot_filename)

# Step 9: Plot the 2D heatmap of Bell operator values
plt.figure(figsize=(8, 6))
plt.imshow(bell_value_grid, origin='lower', extent=[0, 1, 200, 1000], aspect='auto', cmap='plasma')
colorbar = plt.colorbar(label=r'$\mathcal{I}_3$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)

# Add the value of the Bell operator as a label to each square
num_rows, num_cols = bell_value_grid.shape
x_centers = np.linspace(0.1, 0.9, num_cols)
y_centers = np.linspace(250.0, 1050.0, num_rows)
for i, y in enumerate(y_centers):
    for j, x in enumerate(x_centers):
        plt.text(x, y, f"{bell_value_grid[i, j]:.2f}", color="white", ha="center", va="center", fontsize=9)

heatmap_filename = os.path.join(WW_path, "bell_operator_heatmap_WW.pdf")
plt.savefig(heatmap_filename)

# Plot the contour of concurrence values
plt.figure(figsize=(10, 8))
contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, concurrence_grid, cmap='viridis')
contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, concurrence_grid, colors='black', linewidths=0.7)
plt.clabel(contour_lines, inline=True, fontsize=12, fmt="%.2f")
colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{C}_{LB}$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
plot_filename = os.path.join(WW_path, "concurrence_contour_WW.pdf")
plt.savefig(plot_filename)

# Plot the 2D heatmap of concurrence values
plt.figure(figsize=(8, 6))
plt.imshow(concurrence_grid, origin='lower', extent=[0, 1, 200, 1000], aspect='auto', cmap='viridis')
colorbar = plt.colorbar(label=r'$\mathcal{C}_{LB}$', orientation='vertical')
colorbar.ax.yaxis.label.set_fontsize(16)
plt.xlabel(r'$\cos{\Theta}$', fontsize=16)   
plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)

# Add the value of the concurrence bound as a label to each square
num_rows, num_cols = concurrence_grid.shape
x_centers = np.linspace(0.1, 0.9, num_cols)  
y_centers = np.linspace(250.0, 1050.0, num_rows) 
for i, y in enumerate(y_centers):
    for j, x in enumerate(x_centers):
        plt.text(x, y, f"{concurrence_grid[i, j]:.2f}", color="white", ha="center", va="center", fontsize=9)

heatmap_filename = os.path.join(WW_path, "concurrence_heatmap_WW.pdf")
plt.savefig(heatmap_filename)