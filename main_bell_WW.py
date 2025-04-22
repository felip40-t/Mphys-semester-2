
from matplotlib.colors import ListedColormap
import numpy as np
import os
from histo_plotter import read_data
from coefficient_calculator_WW import calculate_coefficients, read_masked_data, calculate_coefficients_fgh, calculate_variance_fgh
from density_matrix_calculator import calculate_density_matrix_AC, O_bell_prime1, calculate_density_matrix_fgh, project_to_psd, unphysicality_score
from Bell_inequality_optimizer import bell_inequality_optimization, inequality_function, optimal_bell_operator
from Unitary_Matrix import euler_unitary_matrix
from concurrence_bound import concurrence_lower, check_density_matrix, concurrence_MB
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap


WW_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_WW_4l_final_process/Plots and data/organised_data"
WW_save = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_WW_4l_final_process/Plots and data/Plots"

regions = { 
        (i, j): [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for i in range(9)
        for j in range(20)
        for cos_min in [0.0 + 0.1 * i]
        for mass_min in [200.0 + 50.0 * j]
    }

def generate_event_count_heatmap(WW_path, WW_save, regions, num_x_bins=180, num_y_bins=200):
    import matplotlib.pyplot as plt

    event_count_grid = np.zeros((num_y_bins, num_x_bins))

    # Define bin edges for high-resolution mapping
    cos_psi_edges = np.linspace(0, 0.9, num_x_bins + 1)  # Bin edges in cos_psi
    inv_mass_edges = np.linspace(200, 1200, num_y_bins + 1)  # Bin edges in M_WW

    # Loop through each defined region and bin events
    for key, region in regions.items():
        print("Calculating event count for region:", region)
        save_dir = os.path.join(
            WW_path,
            f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}"
        )
        
        if os.path.exists(save_dir):
            print(f"Directory {save_dir} found.")
            
            # Load event data
            cos_psi_path = os.path.join(save_dir, "psi_data.txt")
            WW_inv_path = os.path.join(save_dir, "WW_inv_mass.txt")
    
            cos_psi_data = np.loadtxt(cos_psi_path)
            WW_inv_mass = np.loadtxt(WW_inv_path)
    
            # 2D histogram binning into defined grid
            hist2d, _, _ = np.histogram2d(WW_inv_mass, cos_psi_data, bins=[inv_mass_edges, cos_psi_edges])
            
            # Accumulate event counts into the main grid
            event_count_grid += hist2d

    # Plot the heatmap of event counts
    plt.figure(figsize=(12, 10))
    plt.imshow(event_count_grid, origin='lower', extent=[0, 0.9, 200, 1200],
               aspect='auto', cmap='inferno', vmin=0, vmax=2500)
    colorbar = plt.colorbar(label=r'Event Count', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(16)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
    plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)

    # Save the heatmap in both PDF and PNG formats
    heatmap_filename_pdf = os.path.join(WW_save, "event_count_heatmap_WW_180x200.pdf")
    plt.savefig(heatmap_filename_pdf)
    heatmap_filename_png = os.path.join(WW_save, "event_count_heatmap_WW_180x200.png")
    plt.savefig(heatmap_filename_png)
    plt.close()
    
    return event_count_grid

def generate_uniformity_heatmap(WW_path, WW_save, regions):
    """
    Compute and plot the uniformity score heatmap.
    
    Parameters:
        WW_path (str): Base path to the WW data.
        WW_save (str): Directory to save the plots and numpy file.
        regions (dict): Dictionary specifying regions.
    
    Returns:
        uniformity_grid (ndarray): The computed uniformity scores grid.
    """
    import matplotlib.pyplot as plt

    uniformity_grid = np.zeros((9, 20))  # To store uniformity scores

    for (i, j), region in regions.items():
        print(f"Calculating uniformity for region: {region}...")

        save_dir = os.path.join(
            WW_path,
            f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}"
        )
        
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

            # Store the uniformity score in the uniformity grid
            uniformity_grid[i, j] = uniformity

    # Transpose the uniformity_grid to match the expected grid shape
    uniformity_grid = uniformity_grid.T
    np.save(os.path.join(WW_save, "uniformity_scores.npy"), uniformity_grid)

    # Plot the heatmap of uniformity scores
    plt.figure(figsize=(12, 10))
    plt.imshow(uniformity_grid, origin='lower', extent=[0, 0.9, 200, 1200],
                aspect='auto', cmap='plasma_r', vmin=0.7, vmax=1)
    colorbar = plt.colorbar(label='Uniformity Score', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(16)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
    plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
    plt.yticks(np.arange(200, 1201, 100), fontsize=14)
    plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=14)

    # Add the value of the uniformity score to each square
    num_rows, num_cols = uniformity_grid.shape
    x_centers = np.linspace(0.05, 0.85, num_cols)
    y_centers = np.linspace(225.0, 1175.0, num_rows)
    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            score = uniformity_grid[i, j]
            plt.text(x, y, f"{score:.2f}", color="white", ha="center", va="center", fontsize=12)

    # Save the plot in both PDF and PNG formats
    heatmap_filename_pdf = os.path.join(WW_save, "uniformity_heatmap_WW.pdf")
    plt.savefig(heatmap_filename_pdf)
    heatmap_filename_png = os.path.join(WW_save, "uniformity_heatmap_WW.png")
    plt.savefig(heatmap_filename_png)
    plt.close()

    return uniformity_grid

def generate_unphysicality_heatmap(WW_path=WW_path, WW_save=WW_save, regions=regions, data=None):
    """
    Compute and plot the unphysicality score heatmap.
    
    Parameters:
        WW_path (str): Base path to the WW data.
        WW_save (str): Directory to save the plots and numpy file.
        regions (dict): Dictionary specifying regions.
    
    Returns:
        unphysicality_grid (ndarray): The computed unphysicality scores grid.
    """
    import matplotlib.pyplot as plt

    if data is None:
        unphysicality_grid = np.zeros((9, 20))  # To store unphysicality scores

        for (i, j), region in regions.items():
            print(f"Calculating unphysicality for region: {region}...")

            save_dir = os.path.join(
                WW_path,
                f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}"
            )
            
            if os.path.exists(save_dir):
                # Read file paths for theta and phi data
                theta_paths = {
                    1: os.path.join(save_dir, "e+_theta_data.txt"),
                    3: os.path.join(save_dir, "mu-_theta_data.txt")
                }
                phi_paths = {
                    1: os.path.join(save_dir, "e+_phi_data.txt"),
                    3: os.path.join(save_dir, "mu-_phi_data.txt")
                }

                # Calculate coefficients and density matrix from theta and phi data
                f_coefficients, g_coefficients, h_coefficients = calculate_coefficients_fgh(theta_paths, phi_paths)
                density_matrix = calculate_density_matrix_fgh(f_coefficients, g_coefficients, h_coefficients)
                # Calculate the unphysicality score for the density matrix
                unphysicality = unphysicality_score(density_matrix)
                print(f"\nUnphysicality score for region: {region} = {unphysicality:.4g}\n")
                # Store the unphysicality score in the unphysicality grid
                unphysicality_grid[i, j] = unphysicality
            else:
                print(f"Directory {save_dir} not found. Skipping region.")    


        # Transpose the unphysicality_grid to match the expected grid shape
        unphysicality_grid = unphysicality_grid.T
        np.save(os.path.join(WW_save, "unphysicality_scores.npy"), unphysicality_grid)
    else:
        unphysicality_grid = data
    
    # Plot the heatmap of unphysicality scores
    plt.figure(figsize=(12, 10))
    plt.imshow(unphysicality_grid, origin='lower', extent=[0, 0.9, 200, 1200],
                aspect='auto', cmap='plasma')
    colorbar = plt.colorbar(label='Unphysicality', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(16)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
    plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
    plt.yticks(np.arange(200, 1201, 100), fontsize=14)
    plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=14)
    # Add the value of the unphysicality score to each square
    num_rows, num_cols = unphysicality_grid.shape
    x_centers = np.linspace(0.05, 0.85, num_cols)
    y_centers = np.linspace(225.0, 1175.0, num_rows)
    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            score = unphysicality_grid[i, j]
            plt.text(x, y, f"{score:.2f}", color="white", ha="center", va="center", fontsize=12)
    
    # Save the plot in both PDF and PNG formats
    heatmap_filename_pdf = os.path.join(WW_save, "unphysicality_heatmap_WW.pdf")
    plt.savefig(heatmap_filename_pdf)
    heatmap_filename_png = os.path.join(WW_save, "unphysicality_heatmap_WW.png")
    plt.savefig(heatmap_filename_png)
    plt.close()

    return unphysicality_grid

def process_region(region_key, WW_path=WW_path, regions=regions, calc_bell=True, calc_concurrence=True, check_density=False, raw=False):
    """
    Process a single region to calculate density matrix, concurrence, and Bell operator values.

    Parameters:
        region_key (tuple): Key for the desired region in the regions dictionary.
        WW_path (str): Base path to the WW data.
        regions (dict): Dictionary specifying regions.

    Returns:
        dict: A dictionary containing the region, concurrence values (before and after PSD projection),
              Bell operator value, optimal parameters, and its uncertainty.
              Returns None if the save directory is not found.
    """

    region = regions[region_key]
    print(f"\n\nCalculating for region: {region}...\n")
    save_dir = os.path.join(
        WW_path,
        f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}"
    )
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} not found. Skipping region.")
        return None

    # Read file paths for theta and phi data
    theta_paths = {
        1: os.path.join(save_dir, "e+_theta_data.txt"),
        3: os.path.join(save_dir, "mu-_theta_data.txt")
    }
    phi_paths = {
        1: os.path.join(save_dir, "e+_phi_data.txt"),
        3: os.path.join(save_dir, "mu-_phi_data.txt")
    }

    # Calculate coefficients and density matrix from theta and phi data
    f_coefficients, g_coefficients, h_coefficients = calculate_coefficients_fgh(theta_paths, phi_paths)
    density_matrix = calculate_density_matrix_fgh(f_coefficients, g_coefficients, h_coefficients)
    if check_density:
        check_density_matrix(density_matrix)

    # Calculate the unphysicality score for the density matrix
    unphysicality = unphysicality_score(density_matrix)
    print(f"\nUnphysicality score for region: {region} = {unphysicality:.4g}\n")

    if not raw:
        # Project density matrix to positive semi-definite
        density_matrix = project_to_psd(density_matrix, const=unphysicality, normalize_trace=True)
        if check_density:
            check_density_matrix(density_matrix)

    concurrence_val = 0.0
    bell_value = 0.0
    optimal_params = np.zeros(12)
    uncertainty_bell = 0.0

    if calc_concurrence:
        # Calculate the concurrence value for the region
        concurrence_val = concurrence_lower(density_matrix)
        print(f"\nConcurrence bound for region: {region} = {concurrence_val:.4g}\n")

    if calc_bell:
        # Calculate the Bell operator value for the region
        bell_value, optimal_params = bell_inequality_optimization(density_matrix, O_bell_prime1)
        optimal_O_bell = optimal_bell_operator(O_bell_prime1, optimal_params)
        print(f"Bell operator value for region: {region} = {bell_value:.4g}\n")

        # Calculate the uncertainty in the Bell operator value
        variance = calculate_variance_fgh(theta_paths, phi_paths, optimal_O_bell).real
        uncertainty_bell = np.sqrt(variance)
        print(f"Uncertainty of Bell operator for region: {region} = {uncertainty_bell:.6g}\n")

    return {
        'region': region,
        'concurrence_val': concurrence_val,
        'bell_value': bell_value,
        'optimal_params': optimal_params,
        'uncertainty_bell': uncertainty_bell,
        'unphysicality': unphysicality
    }

def plot_contour_heatmap(WW_save, cos_psi_grid, inv_mass_grid, bell_value_grid, label, concurrence=False):
    """
    Plots contour and heatmap of the Bell operator values or Concurrence values

    Parameters:
        WW_save (str): Directory where plots will be saved.
        cos_psi_grid (ndarray): 2D mesh grid of cos(theta) centers.
        inv_mass_grid (ndarray): 2D mesh grid of M_WW centers.
        bell_value_grid (ndarray): 2D array of Bell operator values or concurrence values.
        label (str): Label for the color bar.
        concurrence (bool): If True, indicates that the values are concurrence values.
    """
    import matplotlib.pyplot as plt
    # Generate custom colormap for concurrence
    colors = ['darkblue', 'blue', 'purple', 'red']
    concurrence_cmap = LinearSegmentedColormap.from_list('concurrence_cmap', colors, N=256)

    # Plot the smoothed contour of Bell operator values
    bell_grid_smoothed = gaussian_filter(bell_value_grid, sigma=1.0)
    plt.figure(figsize=(12, 10))
    custom_levels = np.arange(np.round(np.min(bell_grid_smoothed), 1) - 0.1, np.round(np.max(bell_grid_smoothed), 1)+0.2, step=0.1)
    contour_filled = plt.contourf(cos_psi_grid, inv_mass_grid, bell_grid_smoothed, 
                                  levels=custom_levels, cmap=concurrence_cmap if concurrence else 'plasma')
    contour_lines = plt.contour(cos_psi_grid, inv_mass_grid, bell_grid_smoothed, 
                                levels=custom_levels, colors='black', linewidths=0.7)
    plt.clabel(contour_lines, inline=True, fontsize=12, fmt="%.2f")
    if concurrence:
        colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{C}_LB$', orientation='vertical')
    else:
        colorbar = plt.colorbar(contour_filled, label=r'$\mathcal{I}_3$', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(16)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
    plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
    plt.yticks(np.arange(300, 1200, 100), fontsize=14)
    plt.xticks(np.arange(0.1, 0.9, 0.1), fontsize=14)
    plt.tight_layout()
    if concurrence:
        name = f"concurrence_contour_WW_{label}.pdf"
    else:
        name = f"bell_operator_contour_WW_{label}.pdf"
    plot_filename = os.path.join(WW_save, name)
    plt.savefig(plot_filename)
    plot_filename = os.path.join(WW_save, name.replace('.pdf', '.png'))
    plt.savefig(plot_filename)
    plt.close()

    # Plot the 2D heatmap of values
    plt.figure(figsize=(12, 10))
    plt.imshow(bell_value_grid, origin='lower', extent=[0, 0.9, 200, 1200], 
               aspect='auto', cmap=concurrence_cmap if concurrence else 'plasma')
    if concurrence:
        colorbar = plt.colorbar(label=r'$\mathcal{C}_{LB}$', orientation='vertical')
    else:
        colorbar = plt.colorbar(label=r'$\mathcal{I}_3$', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(16)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=16)  
    plt.ylabel(r'$M_{WW} (GeV)$', fontsize=16)
    plt.yticks(np.arange(200, 1201, 100), fontsize=14)
    plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=14)

    # Add the value of the Bell operator as a label to each square
    num_rows, num_cols = bell_value_grid.shape
    x_centers = np.linspace(0.05, 0.85, num_cols)
    y_centers = np.linspace(225.0, 1175.0, num_rows)
    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            plt.text(x, y, f"{bell_value_grid[i, j]:.2f}", color="white", 
                     ha="center", va="center", fontsize=12)
    plt.tight_layout()

    if concurrence:
        name = f"concurrence_heatmap_WW_{label}.pdf"
    else:
        name = f"bell_operator_heatmap_WW_{label}.pdf"

    heatmap_filename = os.path.join(WW_save, name)
    plt.savefig(heatmap_filename)
    heatmap_filename = os.path.join(WW_save, name.replace('.pdf', '.png'))
    plt.savefig(heatmap_filename)
    plt.close()

# Generate mesh grid for cos_psi and WW_inv_mass based on regions (9 columns and 20 rows)
# Each cos_psi region is [cos_min, cos_min+0.1] so we take the center as cos_min + 0.05 for 0 <= cos_min < 0.9
# Each WW_inv_mass region is [mass_min, mass_min+50.0] so we take the center as mass_min + 25.0 for 200 <= mass_min < 1200
cos_psi_centers = np.arange(0.05, 0.9, 0.1)         
inv_mass_centers = np.arange(225.0, 1200.0, 50.0)   

cos_psi_grid, inv_mass_grid = np.meshgrid(cos_psi_centers, inv_mass_centers)

# Initialize the Bell operator, uncertainty, and concurrence grids for 20x9 regions
bell_value_grid = np.loadtxt(os.path.join(WW_path, "bell_operator_grid_WW_fgh_smooth_clip.txt"), delimiter=',')
uncertainty_grid = np.loadtxt(os.path.join(WW_path, "uncertainty_grid_WW_fgh_smooth_clip.txt"), delimiter=',')
concurrence_grid = np.loadtxt(os.path.join(WW_path, "concurrence_grid_WW_fgh_smooth_clip.txt"), delimiter=',')
optimal_params_grid = np.load(os.path.join(WW_path, "optimal_params_grid_WW_fgh_smooth_clip.npy"))


for key, region in regions.items():
    quantities = process_region(key, WW_path=WW_path, regions=regions, calc_bell=True, calc_concurrence=True, check_density=False, raw=False)
    if quantities is not None:
        i, j = key
        bell_value_grid[i, j] = quantities['bell_value']
        uncertainty_grid[i, j] = quantities['uncertainty_bell']
        concurrence_grid[i, j] = quantities['concurrence_val']
        optimal_params_grid[:, i, j] = quantities['optimal_params']

        # Save the Bell operator value grid to a file
        np.savetxt(os.path.join(WW_path, "bell_operator_grid_WW_fgh_smooth_clip.txt"), bell_value_grid, delimiter=',')

        # Save the concurrence value grid to a file
        np.savetxt(os.path.join(WW_path, "concurrence_grid_WW_fgh_smooth_clip.txt"), concurrence_grid, delimiter=',')

        # Save the optimal parameters grid to a npy file
        np.save(os.path.join(WW_path, "optimal_params_grid_WW_fgh_smooth_clip.npy"), optimal_params_grid)

        # Save the uncertainty grid to a file
        np.savetxt(os.path.join(WW_path, "uncertainty_grid_WW_fgh_smooth_clip.txt"), uncertainty_grid, delimiter=',')

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
unphysicality_grid = np.load(os.path.join(WW_save, "unphysicality_scores.npy"))
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

