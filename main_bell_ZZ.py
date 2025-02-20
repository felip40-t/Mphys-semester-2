import numpy as np
import os
from histo_plotter import read_data
from coefficient_calculator_ZZ import calculate_coefficients, read_masked_data
from density_matrix_calculator import calculate_density_matrix, O_bell_prime1
from Bell_inequality_optimizer import bell_inequality_optimization, inequality_function
import matplotlib.pyplot as plt

ZZ_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_ZZ_SM/Plots and data"
# Read theta and phi values for both datasets
cos_theta_paths = {
    1: os.path.join(ZZ_path, "mu+/theta_data_4.txt"),
    3: os.path.join(ZZ_path, "e+/theta_data_4.txt")
}
phi_paths = {
    1: os.path.join(ZZ_path, "mu+/phi_data_4.txt"),
    3: os.path.join(ZZ_path, "e+/phi_data_4.txt")
}

cos_psi_data = read_data(os.path.join(ZZ_path, "psi_data_4.txt"))
ZZ_inv_mass = read_data(os.path.join(ZZ_path, "ZZ_inv_mass_4.txt"))

# Step 1: Calculate coefficients for whole dataset
A_coefficients, C_coefficients = calculate_coefficients(cos_theta_paths, phi_paths, mask=None)
density_matrix = calculate_density_matrix(A_coefficients, C_coefficients)

# Step 2: Perform Bell operator optimization
total_bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
print(f"Maximized Bell inequality value for whole phase space: {total_bell_value}")

# Step 3: Find coefficients for masked phase space
mask = read_masked_data(cos_psi_data, ZZ_inv_mass, (0, 0.2), (300, 325))
A_coefficients, C_coefficients = calculate_coefficients(cos_theta_paths, phi_paths, mask)
density_matrix = calculate_density_matrix(A_coefficients, C_coefficients)

# Step 4: Calculate Bell operator for masked space
masked_bell_value, _ = bell_inequality_optimization(density_matrix, O_bell_prime1)
print(f"Maximized Bell inequality value for region: {masked_bell_value}")

# Step 5: Generate mesh grid for cos_psi and ZZ_inv_mass
cos_psi_range = np.arange(-1, 1.1, step=0.1) 
inv_mass_range = np.arange(200, 1040, step=40) 

cos_psi_contour = np.arange(-0.95, 1.05, step=0.1)   
inv_mass_contour = np.arange(220, 1020, step=40) 

cos_psi_grid, inv_mass_grid = np.meshgrid(cos_psi_contour, inv_mass_contour)

# Step 6: Initialize the Bell operator value grid
bell_value_grid = np.zeros((20,20))

# Step 7: Calculate Bell operator value for each point in the mesh grid
for i in range(len(inv_mass_range)-1):
    for j in range(len(cos_psi_range)-1):
        # Apply a mask for the current region in the mesh grid
        mask = read_masked_data(cos_psi_data, ZZ_inv_mass, 
                                (cos_psi_range[j], cos_psi_range[j+1]), 
                                (inv_mass_range[i], inv_mass_range[i+1]))
        # Calculate coefficients for this masked region
        A_coefficients, C_coefficients = calculate_coefficients(cos_theta_paths, phi_paths, mask)
        density_matrix = calculate_density_matrix(A_coefficients, C_coefficients)
        bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
        bell_value_grid[i, j] = bell_value

# Step 8: Plot the contour of Bell operator values
plt.figure(figsize=(8, 6))

# Define custom contour levels
custom_levels = np.linspace(np.min(bell_value_grid), np.max(bell_value_grid), 10)
contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, bell_value_grid, levels=custom_levels, cmap='viridis')
contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, bell_value_grid, levels=custom_levels, colors='black', linewidths=0.5)
plt.colorbar(contour_filled, label='Bell Operator Value')

plt.xlabel(r'$\cos{\Theta}$')  
plt.ylabel(r'$M_{ZZ} (GeV)$')
plot_filename = os.path.join(ZZ_path, "bell_operator_contour_ZZ.png")
plt.savefig(plot_filename)

# Step 9: Plot the 2D heatmap of Bell operator values
plt.figure(figsize=(8, 6))
plt.imshow(bell_value_grid, origin='lower', extent=[-1, 1, 200, 1000], aspect='auto', cmap='plasma')
plt.colorbar(label='Bell Operator Value')
plt.xlabel(r'$\cos{\Theta}$')  
plt.ylabel(r'$M_{ZZ} (GeV)$')

# Save the 2D heatmap of Bell operator values
heatmap_filename = os.path.join(ZZ_path, "bell_operator_heatmap_ZZ.png")
plt.savefig(heatmap_filename)