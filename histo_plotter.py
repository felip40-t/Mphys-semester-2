import os
import numpy as np
import matplotlib.pyplot as plt
from lhe_reading_WW import find_latest_run_dir

process_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6/4_lepton_process"
base_dir = os.path.join(process_dir, "Events")

particle_directories = {
    'e-': os.path.join(process_dir, "Plots and data/e-"),  
    'e+': os.path.join(process_dir, "Plots and data/e+"), 
    'w-': os.path.join(process_dir, "Plots and data/w-"),  
    'w+': os.path.join(process_dir, "Plots and data/w+"), 
}

COM_ENERGY = 13_000  # GeV
NEVENTS = 1_000_000  # Total number of events

def read_data(file_path):
    """
    Reads the data from the specified .txt file and returns it as a numpy array.
    Each line in the file should be in the format:
    E, px, py, pz
    """
    return np.loadtxt(file_path, delimiter=',')

def calculate_transverse_p(data):
    """
    Calculate the transverse momentum p_T = sqrt(px^2 + py^2).
    """
    return np.sqrt(data[:, 1]**2 + data[:, 2]**2)

def calculate_pseudorapidity(data):
    """
    Calculate the pseudorapidity eta = arctanh(pz / |p|).
    """
    abs_p = np.linalg.norm(data[:, 1:], axis=1)
    return np.arctanh(data[:, 3] / abs_p)

def calculate_inv_mass(data):
    """
    Calculates the invariant mass from the 4-momenta data.
    """
    return np.sqrt(data[:, 0]**2 - np.linalg.norm(data[:, 1:], axis=1)**2)

def plot_histogram(data, particle_name, output_dir, xsec, num, hist_type='p_T'):
    """
    Generalized function to plot either p_T, eta, or inv_mass histogram.
    """
    if hist_type == 'p_T':
        # Calculate transverse momentum and set appropriate labels
        values = calculate_transverse_p(data)
        xlabel = rf'$p^{{T}}_{{{particle_name}}}$ (GeV/c)'
    elif hist_type == 'eta':
        # Calculate pseudorapidity and set appropriate labels
        values = calculate_pseudorapidity(data)
        xlabel = rf'$\eta_{{{particle_name}}}$'
    elif hist_type == 'inv_mass':
        # Calculate invariant mass and set appropriate labels
        values = calculate_inv_mass(data)
        xlabel = rf'$m_{{{particle_name}}}$ (GeV/c²)'

    # Define bins and normalization
    if hist_type == 'inv_mass':
        bins = np.linspace(0, 500, 41)  # 40 bins between 0 and 500 GeV
    else:
        bins = np.linspace(values.min(), values.max(), 51)
    normalisation = xsec * 1000 / NEVENTS  # Cross-section normalization
    weights = np.ones_like(values) * normalisation
    
    # Create the histogram
    plt.figure(figsize=(10, 6), dpi=100)
    plt.hist(values, bins=bins, weights=weights, histtype='step', edgecolor='blue')
    
    # Set axis labels, scales, and grid
    plt.yscale('log')
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(r'$\sigma$ (fb)', fontsize=16)
    plt.grid(axis='y', alpha=0.75)

    # Set xlim to 500 GeV for W bosons
    if hist_type == 'inv_mass' and particle_name in ['w-', 'w+']:
        plt.xlim(0, 500)  # Set x-axis limit for invariant mass
    
    # Save the plot
    figure_path = os.path.join(output_dir, f"{particle_name}_{hist_type}_histo_run_{num}.pdf")
    figure_path_png = os.path.join(output_dir, f"{particle_name}_{hist_type}_histo_run_{num}.png")
    plt.savefig(figure_path)
    plt.savefig(figure_path_png)

def process_and_plot(particle_directories):
    """
    Reads data from electron and positron directories, then plots p_T, eta, and inv_mass histograms.
    """
    _, number = find_latest_run_dir(base_dir)
    xsec_file = os.path.join(particle_directories['e-'], f"cross_sec_{number}.txt")
    cross_section = read_data(xsec_file)

    for particle_name, directory in particle_directories.items():
        data_file = os.path.join(directory, f"data_{number}.txt")
        
        if os.path.exists(data_file):
            print(f"Reading data from {data_file}...")
            data = read_data(data_file)
            
            # Plot both p_T and eta histograms for electron and positron
            if particle_name in ['e-', 'e+']:
                plot_histogram(data, particle_name, directory, cross_section, number, hist_type='p_T')
                plot_histogram(data, particle_name, directory, cross_section, number, hist_type='eta')
                
            # Plot invariant mass for W bosons
            if particle_name in ['w-', 'w+']:
                plot_histogram(data, particle_name, directory, cross_section, number, hist_type='inv_mass')

        else:
            print(f"File not found: {data_file}")

if __name__ == "__main__":
    process_and_plot(particle_directories)

