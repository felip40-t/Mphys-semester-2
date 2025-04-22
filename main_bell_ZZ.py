import numpy as np
import os
from histo_plotter import read_data
from coefficient_calculator_ZZ import calculate_coefficients_AC, read_masked_data, calculate_coefficients_fgh, calculate_variance_AC
from density_matrix_calculator import calculate_density_matrix_AC, O_bell_prime1, calculate_density_matrix_fgh, project_to_psd, unphysicality_score
from Bell_inequality_optimizer import bell_inequality_optimization, inequality_function, optimal_bell_operator
from Unitary_Matrix import euler_unitary_matrix
from concurrence_bound import concurrence_lower, check_density_matrix
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

ZZ_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_ZZ_SM/Plots and data/reorganised_data"
ZZ_save = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_ZZ_SM/Plots and data/reorganised_data/Entanglement plots"

regions = { 
        (i, j): [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for i in range(9)
        for j in range(16)
        for cos_min in [0.0 + 0.1 * i]
        for mass_min in [200.0 + 50.0 * j]
    }

def generate_event_count_heatmap(ZZ_path, ZZ_save, regions, num_x_bins=180, num_y_bins=200):
    import matplotlib.pyplot as plt

    event_count_grid = np.zeros((num_y_bins, num_x_bins))

    # Define bin edges for high-resolution mapping
    cos_psi_edges = np.linspace(0, 0.9, num_x_bins + 1)  # Bin edges in cos_psi
    inv_mass_edges = np.linspace(200, 1200, num_y_bins + 1)  # Bin edges in M_ZZ

    # Loop through each defined region and bin events
    for key, region in regions.items():
        print("Calculating event count for region:", region)
        save_dir = os.path.join(
            ZZ_path,
            f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}"
        )
        
        if os.path.exists(save_dir):
            print(f"Directory {save_dir} found.")
            
            # Load event data
            cos_psi_path = os.path.join(save_dir, "psi_data_combined_new.txt")
            ZZ_inv_path = os.path.join(save_dir, "ZZ_inv_mass_combined_new.txt")
    
            cos_psi_data = np.loadtxt(cos_psi_path)
            ZZ_inv_mass = np.loadtxt(ZZ_inv_path)
    
            # 2D histogram binning into defined grid
            hist2d, _, _ = np.histogram2d(ZZ_inv_mass, cos_psi_data, bins=[inv_mass_edges, cos_psi_edges])
            
            # Accumulate event counts into the main grid
            event_count_grid += hist2d

    # Plot the heatmap of event counts
    plt.figure(figsize=(12, 10))
    plt.imshow(event_count_grid, origin='lower', extent=[0, 0.9, 200, 1200],
               aspect='auto', cmap='inferno', vmin=0, vmax=2500)
    colorbar = plt.colorbar(label=r'Event Count', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(16)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
    plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)

    # Save the heatmap in both PDF and PNG formats
    heatmap_filename_pdf = os.path.join(ZZ_save, "event_count_heatmap_ZZ_180x200.pdf")
    plt.savefig(heatmap_filename_pdf)
    heatmap_filename_png = os.path.join(ZZ_save, "event_count_heatmap_ZZ_180x200.png")
    plt.savefig(heatmap_filename_png)
    plt.close()
    
    return event_count_grid

def generate_uniformity_heatmap(ZZ_path, ZZ_save, regions):
    """
    Compute and plot the uniformity score heatmap.
    
    Parameters:
        ZZ_path (str): Base path to the ZZ data.
        ZZ_save (str): Directory to save the plots and numpy file.
        regions (dict): Dictionary specifying regions.
    
    Returns:
        uniformity_grid (ndarray): The computed uniformity scores grid.
    """
    import matplotlib.pyplot as plt

    uniformity_grid = np.zeros((9, 20))  # To store uniformity scores

    for (i, j), region in regions.items():
        print(f"Calculating uniformity for region: {region}...")

        save_dir = os.path.join(
            ZZ_path,
            f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}"
        )
        
        if os.path.exists(save_dir):
            cos_psi_path = os.path.join(save_dir, "psi_data_combined_new.txt")
            ZZ_inv_path = os.path.join(save_dir, "ZZ_inv_mass_combined_new.txt")

            cos_psi_data = np.loadtxt(cos_psi_path)
            ZZ_inv_mass = np.loadtxt(ZZ_inv_path)

            # Histogram within region: use 10x10 binning
            h, _, _ = np.histogram2d(ZZ_inv_mass, cos_psi_data, bins=[10, 10])
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
    np.save(os.path.join(ZZ_save, "uniformity_scores.npy"), uniformity_grid)

    # Plot the heatmap of uniformity scores
    plt.figure(figsize=(12, 10))
    plt.imshow(uniformity_grid, origin='lower', extent=[0, 0.9, 200, 1200],
                aspect='auto', cmap='plasma_r', vmin=0.7, vmax=1)
    colorbar = plt.colorbar(label='Uniformity Score', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(16)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
    plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)
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
    heatmap_filename_pdf = os.path.join(ZZ_save, "uniformity_heatmap_ZZ.pdf")
    plt.savefig(heatmap_filename_pdf)
    heatmap_filename_png = os.path.join(ZZ_save, "uniformity_heatmap_ZZ.png")
    plt.savefig(heatmap_filename_png)
    plt.close()

    return uniformity_grid

def generate_unphysicality_heatmap(ZZ_path=ZZ_path, ZZ_save=ZZ_save, regions=regions, unphysicality_grid=False):
    """
    Compute and plot the unphysicality score heatmap.
    
    Parameters:
        ZZ_path (str): Base path to the ZZ data.
        ZZ_save (str): Directory to save the plots and numpy file.
        regions (dict): Dictionary specifying regions.
    
    Returns:
        unphysicality_grid (ndarray): The computed unphysicality scores grid.
    """

    if not unphysicality_grid:
        unphysicality_grid = np.zeros((9, 16))  # To store unphysicality scores
    
        for (i, j), region in regions.items():
            print(f"Calculating unphysicality for region: {region}...")

            save_dir = os.path.join(
                ZZ_path,
                f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}"
            )
            
            if os.path.exists(save_dir):
                # Read file paths for theta and phi data
                theta_paths = {
                    1: os.path.join(save_dir, "e+_theta_data_combined_new.txt"),
                    3: os.path.join(save_dir, "mu+_theta_data_combined_new.txt")
                }
                phi_paths = {
                    1: os.path.join(save_dir, "e+_phi_data_combined_new.txt"),
                    3: os.path.join(save_dir, "mu+_phi_data_combined_new.txt")
                }

                # Calculate coefficients and density matrix from theta and phi data
                A_coefficients, C_coefficients = calculate_coefficients_AC(theta_paths, phi_paths)
                density_matrix = calculate_density_matrix_AC(A_coefficients, C_coefficients)
                # Calculate the unphysicality score for the density matrix
                unphysicality = unphysicality_score(density_matrix)
                print(f"\nUnphysicality score for region: {region} = {unphysicality:.4g}\n")
                # Store the unphysicality score in the unphysicality grid
                unphysicality_grid[i, j] = unphysicality
            else:
                print(f"Directory {save_dir} not found. Skipping region.")

        # Transpose the unphysicality_grid to match the expected grid shape
        unphysicality_grid = unphysicality_grid.T
        np.save(os.path.join(ZZ_save, "unphysicality_scores_ZZ.npy"), unphysicality_grid)

    else:
        unphysicality_grid = np.load(os.path.join(ZZ_save, "unphysicality_scores_ZZ.npy"))

    # Plot the heatmap of unphysicality scores
    plt.figure(figsize=(12, 10))
    plt.imshow(unphysicality_grid, origin='lower', extent=[0, 0.9, 200, 1000],
                aspect='auto', cmap='plasma')
    colorbar = plt.colorbar(label='Unphysicality', orientation='vertical')
    colorbar.ax.yaxis.label.set_fontsize(16)
    plt.xlabel(r'$\cos{\Theta}$', fontsize=16)
    plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)
    plt.yticks(np.arange(200, 1001, 100), fontsize=14)
    plt.xticks(np.arange(0.0, 1.0, 0.1), fontsize=14)
    # Add the value of the unphysicality score to each square
    num_rows, num_cols = unphysicality_grid.shape
    x_centers = np.linspace(0.05, 0.85, num_cols)
    y_centers = np.linspace(225.0, 975.0, num_rows)
    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            score = unphysicality_grid[i, j]
            plt.text(x, y, f"{score:.2f}", color="white", ha="center", va="center", fontsize=12)
    
    # Save the plot in both PDF and PNG formats
    heatmap_filename_pdf = os.path.join(ZZ_save, "unphysicality_heatmap_ZZ.pdf")
    plt.savefig(heatmap_filename_pdf)
    heatmap_filename_png = os.path.join(ZZ_save, "unphysicality_heatmap_ZZ.png")
    plt.savefig(heatmap_filename_png)
    plt.close()

    return unphysicality_grid

def process_region(region_key, ZZ_path=ZZ_path, regions=regions, calc_bell=True, calc_concurrence=True, check_density=False, raw=False):
    """
    Process a single region to calculate density matrix, concurrence, and Bell operator values.

    Parameters:
        region_key (tuple): Key for the desired region in the regions dictionary.
        ZZ_path (str): Base path to the ZZ data.
        regions (dict): Dictionary specifying regions.

    Returns:
        dict: A dictionary containing the region, concurrence values (before and after PSD projection),
              Bell operator value, optimal parameters, and its uncertainty.
              Returns None if the save directory is not found.
    """

    region = regions[region_key]
    print(f"\n\nCalculating for region: {region}...\n")
    save_dir = os.path.join(
        ZZ_path,
        f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}"
    )
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} not found. Skipping region.")
        return None

    # Read file paths for theta and phi data
    theta_paths = {
        1: os.path.join(save_dir, "e+_theta_data_combined_new.txt"),
        3: os.path.join(save_dir, "mu+_theta_data_combined_new.txt")
    }
    phi_paths = {
        1: os.path.join(save_dir, "e+_phi_data_combined_new.txt"),
        3: os.path.join(save_dir, "mu+_phi_data_combined_new.txt")
    }

    # Calculate coefficients and density matrix from theta and phi data
    A_coefficients, C_coefficients = calculate_coefficients_AC(theta_paths, phi_paths)
    density_matrix = calculate_density_matrix_AC(A_coefficients, C_coefficients)
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
        variance = calculate_variance_AC(theta_paths, phi_paths, optimal_O_bell).real
        print(f"Variance of Bell operator for region: {region} = {variance:.6g}\n")
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

def plot_contour_heatmap(ZZ_save, cos_psi_grid, inv_mass_grid, bell_value_grid, label, concurrence=False):
    """
    Plots contour and heatmap of the Bell operator values or Concurrence values

    Parameters:
        ZZ_save (str): Directory where plots will be saved.
        cos_psi_grid (ndarray): 2D mesh grid of cos(theta) centers.
        inv_mass_grid (ndarray): 2D mesh grid of M_ZZ centers.
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
    plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)
    plt.yticks(np.arange(300, 1200, 100), fontsize=14)
    plt.xticks(np.arange(0.1, 0.9, 0.1), fontsize=14)
    plt.tight_layout()
    if concurrence:
        name = f"concurrence_contour_ZZ_{label}.pdf"
    else:
        name = f"bell_operator_contour_ZZ_{label}.pdf"
    plot_filename = os.path.join(ZZ_save, name)
    plt.savefig(plot_filename)
    plot_filename = os.path.join(ZZ_save, name.replace('.pdf', '.png'))
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
    plt.ylabel(r'$M_{ZZ} (GeV)$', fontsize=16)
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
        name = f"concurrence_heatmap_ZZ_{label}.pdf"
    else:
        name = f"bell_operator_heatmap_ZZ_{label}.pdf"

    heatmap_filename = os.path.join(ZZ_save, name)
    plt.savefig(heatmap_filename)
    heatmap_filename = os.path.join(ZZ_save, name.replace('.pdf', '.png'))
    plt.savefig(heatmap_filename)
    plt.close()

process_region((8, 4), ZZ_path=ZZ_path, regions=regions, calc_bell=True, calc_concurrence=True, check_density=True, raw=True)

cos_psi_centers = np.arange(0.05, 0.9, 0.1)         
inv_mass_centers = np.arange(225.0, 1001.0, 50.0) 

cos_psi_grid, inv_mass_grid = np.meshgrid(cos_psi_centers, inv_mass_centers)

label = "raw"

bell_value_grid = np.loadtxt(os.path.join(ZZ_path, f"bell_operator_grid_ZZ_AC_{label}.txt"), delimiter=',')
uncertainty_grid = np.loadtxt(os.path.join(ZZ_path, f"uncertainty_grid_ZZ_AC_{label}.txt"), delimiter=',')
concurrence_grid = np.loadtxt(os.path.join(ZZ_path, f"concurrence_grid_ZZ_AC_{label}.txt"), delimiter=',')
optimal_params_grid = np.load(os.path.join(ZZ_path, f"optimal_params_grid_ZZ_AC_{label}.npy"))


for key, region in regions.items():
    quantities = process_region(key, ZZ_path=ZZ_path, regions=regions, calc_bell=True, calc_concurrence=True, check_density=False, raw=True)
    if quantities is not None:
        i, j = key
        bell_value_grid[i, j] = quantities['bell_value']
        uncertainty_grid[i, j] = quantities['uncertainty_bell']
        concurrence_grid[i, j] = quantities['concurrence_val']
        optimal_params_grid[:, i, j] = quantities['optimal_params']

        # Save the Bell operator value grid to a file
        np.savetxt(os.path.join(ZZ_path, f"bell_operator_grid_ZZ_AC_{label}.txt"), bell_value_grid, delimiter=',')

        # Save the concurrence value grid to a file
        np.savetxt(os.path.join(ZZ_path, f"concurrence_grid_ZZ_AC_{label}.txt"), concurrence_grid, delimiter=',')

        # Save the optimal parameters grid to a npy file
        np.save(os.path.join(ZZ_path, f"optimal_params_grid_ZZ_AC_{label}.npy"), optimal_params_grid)

        # Save the uncertainty grid to a file
        np.savetxt(os.path.join(ZZ_path, f"uncertainty_grid_ZZ_AC_{label}.txt"), uncertainty_grid, delimiter=',')



# bell_matrix = np.array([
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
# ])[::-1]

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
