import numpy as np
import os
from histo_plotter import read_data
from coefficient_calculator_WW import calculate_coefficients, save_coefficients, read_masked_data
from density_matrix_calculator import read_coefficients, calculate_density_matrix, O_bell_prime
from Bell_inequality_optimizer import bell_inequality_optimization, inequality_function
import matplotlib.pyplot as plt

WW_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_WW_SM/Plots and data"
# Read theta and phi values for both datasets
cos_theta_paths = {
    1: os.path.join(WW_path, "mu-/theta_data_1.txt"),
    3: os.path.join(WW_path, "e+/theta_data_1.txt")
}
phi_paths = {
    1: os.path.join(WW_path, "mu-/phi_data_1.txt"),
    3: os.path.join(WW_path, "e+/phi_data_1.txt")
}

cos_psi_data = read_data(os.path.join(WW_path, "psi_data_1.txt"))
WW_inv_mass = read_data(os.path.join(WW_path, "WW_inv_mass_1.txt"))

# Step 1: Calculate coefficients for whole dataset
A_coefficients, C_coefficients = calculate_coefficients(cos_theta_paths, phi_paths, mask=None)
save_coefficients(A_coefficients, C_coefficients, WW_path)

# Step 2: Read coefficients and calculate the density matrix
A_coefficients_file = os.path.join(WW_path, "A_coefficients_run_1.csv")
C_coefficients_file = os.path.join(WW_path, "C_coefficients_run_1.csv")
A_coefficients = read_coefficients(A_coefficients_file)
C_coefficients = read_coefficients(C_coefficients_file)
density_matrix = calculate_density_matrix(A_coefficients, C_coefficients)

# Step 3: Perform Bell operator optimization
total_bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime)
while total_bell_value < 1.0:
    total_bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime)
print(f"Maximized Bell inequality value for whole phase space: {total_bell_value}")

# Step 4: Find coefficients for masked phase space
mask = read_masked_data(cos_psi_data, WW_inv_mass, (0, 0.25), (500, 600))
A_coefficients, C_coefficients = calculate_coefficients(cos_theta_paths, phi_paths, mask)
save_coefficients(A_coefficients, C_coefficients, WW_path)
A_coefficients_file = os.path.join(WW_path, "A_coefficients_run_1.csv")
C_coefficients_file = os.path.join(WW_path, "C_coefficients_run_1.csv")
A_coefficients = read_coefficients(A_coefficients_file)
C_coefficients = read_coefficients(C_coefficients_file)

# Step 5: Calculate density matrix for masked coefficients
density_matrix = calculate_density_matrix(A_coefficients, C_coefficients)

# Step 6: Calculate Bell operator for masked space
masked_bell_value = inequality_function(density_matrix, O_bell_prime, optimal_params)
print(f"Maximized Bell inequality value for region: {masked_bell_value}")


# Step 7: Generate mesh grid for cos_psi and WW_inv_mass
cos_psi_range = np.arange(-1, 1.2, step=0.2) 
inv_mass_range = np.arange(200, 1080, step=80) 


cos_psi_contour = np.arange(-0.9, 1.1, step=0.2)   
inv_mass_contour = np.arange(240, 1040, step=80) 

cos_psi_grid, inv_mass_grid = np.meshgrid(cos_psi_contour, inv_mass_contour)

# Step 8: Initialize the Bell operator value grid
bell_value_grid = np.zeros((10,10))

num_bins_cos_psi = 10
num_bins_inv_mass = 10

# Use np.histogram2d to count the number of events in each bin
event_counts, x_edges, y_edges = np.histogram2d(cos_psi_data, WW_inv_mass, 
                                                bins=[num_bins_cos_psi, num_bins_inv_mass], 
                                                range=[[-1, 1], [200, 1000]])
plt.figure(figsize=(8, 6))
plt.imshow(event_counts.T, origin='lower', extent=[-1, 1, 200, 1000], aspect='auto', cmap='hot')
plt.colorbar(label='Number of events')
plt.xlabel(r'$\cos{\Psi}$')
plt.ylabel(r'$M_{WW}$')

# Save the event count heatmap to the WW_path directory
event_plot_filename = os.path.join(WW_path, "event_count_heatmap.png")
plt.savefig(event_plot_filename)

# Step 9: Calculate Bell operator value for each point in the mesh grid
for i in range(len(inv_mass_range)-1):
    for j in range(len(cos_psi_range)-1):
        # Apply a mask for the current region in the mesh grid
        mask = read_masked_data(cos_psi_data, WW_inv_mass, 
                                (cos_psi_range[j], cos_psi_range[j+1]), 
                                (inv_mass_range[i], inv_mass_range[i+1]))
        # Calculate coefficients for this masked region
        A_coefficients, C_coefficients = calculate_coefficients(cos_theta_paths, phi_paths, mask)
        save_coefficients(A_coefficients, C_coefficients, WW_path)
        A_coefficients_file = os.path.join(WW_path, "A_coefficients_run_1.csv")
        C_coefficients_file = os.path.join(WW_path, "C_coefficients_run_1.csv")
        A_coefficients = read_coefficients(A_coefficients_file)
        C_coefficients = read_coefficients(C_coefficients_file)
        density_matrix = calculate_density_matrix(A_coefficients, C_coefficients)
        bell_value = inequality_function(density_matrix, O_bell_prime, optimal_params)
        bell_value_grid[i, j] = bell_value

# Step 10: Plot the contour of Bell operator values
plt.figure(figsize=(8, 6))
contour = plt.contourf(cos_psi_grid, inv_mass_grid, bell_value_grid, levels=5, cmap='viridis')
plt.colorbar(contour)
plt.xlabel(r'$\cos{\Psi}$') 
plt.ylabel(r'$M_{WW}$')
plot_filename = os.path.join(WW_path, "bell_operator_contour_WW.png")
plt.savefig(plot_filename)

# Step 11: Plot the 2D heatmap of Bell operator values
plt.figure(figsize=(8, 6))
plt.imshow(bell_value_grid, origin='lower', extent=[-1, 1, 200, 1000], aspect='auto', cmap='plasma')
plt.colorbar(label='Bell Operator Value')
plt.xlabel(r'$\cos{\Psi}$')  
plt.ylabel(r'$M_{WW}$')

# Save the 2D heatmap of Bell operator values
heatmap_filename = os.path.join(WW_path, "bell_operator_heatmap.png")
plt.savefig(heatmap_filename)