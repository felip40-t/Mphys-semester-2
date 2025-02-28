import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Function to read data from the file (assuming it's a column of numbers)
def read_data(filepath):
    return np.loadtxt(filepath)

# Define the base directory structure for the three processes
processes = {
    "ZZ": {
        "base_path": "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_ZZ_SM/Plots and data",
        "particles": {"mu+": "mu+/phi_data_4.txt", "e+": "e+/phi_data_4.txt"},
    },
    "WZ": {
        "base_path": "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_WZ_SM/Plots and data",
        "particles": {"mu+": "mu+/phi_data_2.txt", "e+": "e+/phi_data_2.txt"},
    },
    "WW": {
        "base_path": "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_WW_SM/Plots and data",
        "particles": {"mu-": "mu-/phi_data_1.txt", "e+": "e+/phi_data_1.txt"},
    }
}

# Number of bins for the histogram
num_bins = 60

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate phi histograms for specified processes.')
parser.add_argument('--process', choices=['ZZ', 'WZ', 'WW', 'all'], default='all', help='Specify the process to run (ZZ, WZ, WW, or all)')
args = parser.parse_args()

# Determine which processes to run
if args.process == 'all':
    processes_to_run = processes.keys()
else:
    processes_to_run = [args.process]

# Loop over the processes
for process_name in processes_to_run:
    process_info = processes[process_name]
    base_path = process_info["base_path"]
    particle_files = process_info["particles"]
    
    # Read phi data for both particles
    if 'mu+' in particle_files:
        phi_values_1 = read_data(os.path.join(base_path, particle_files["mu+"]))
    else:
        phi_values_1 = read_data(os.path.join(base_path, particle_files["mu-"]))
    phi_values_2 = read_data(os.path.join(base_path, particle_files["e+"]))
    
    # Create the histograms
    counts_1, bin_edges_1 = np.histogram(phi_values_1, bins=num_bins)
    counts_2, bin_edges_2 = np.histogram(phi_values_2, bins=num_bins)
    
    # Plot the histograms on the same figure
    plt.figure(figsize=(12, 8), dpi=800)
    
    # Plot histogram for the first particle (mu+ or mu-)
    if 'mu+' in particle_files:
        plt.hist(phi_values_1, bins=num_bins, density=True, histtype='step', edgecolor='blue', label=r"$\mu^+$")
    else:
        plt.hist(phi_values_1, bins=num_bins, density=True, histtype='step', edgecolor='blue', label=r"$\mu^-$")
    
    # Plot histogram for the second particle (e+)
    plt.hist(phi_values_2, bins=num_bins, density=True, histtype='step', edgecolor='red', label=r"$e^+$")
    
    # Set the x-axis label for phi and the y-axis label for normalized differential cross section
    plt.xlabel(r'$\phi$', fontsize=20)
    plt.ylabel(r'$1/\sigma{\cdot}d\sigma/d\phi$', fontsize=20)
    
    # Adjust tick parameters
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='y', labelright=False, right=True)
    
    # Add grid lines
    plt.grid(True)
    
    plt.xlim(-np.pi, np.pi)
    plt.ylim(0.1,0.2)
    
    if process_name == 'WW':
        plt.text(-3, 0.24, r"$ p \; p \; \to \; e^+ \; \nu_e \; \mu^- \; \bar{\nu}_{\mu}$" + '\n' + r"$\sqrt{s} = 13 \, \mathrm{TeV}$", 
         fontsize=16, verticalalignment='top', horizontalalignment='left', 
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'),
         #fontname='consolas'
         )
    elif process_name == 'WZ':
        plt.text(-3, 0.24, r"$p \; p \; \to \; e^+ \; \nu_e \; \mu^+ \mu^-$" + '\n' + r"$\sqrt{s} = 13 \, \mathrm{TeV}$", 
         fontsize=16, verticalalignment='top', horizontalalignment='left', 
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'),
         #fontname='consolas'
         )
    else:
        plt.text(-3, 0.24, r"$p \; p \; \to \; e^+ \; e^- \; \mu^+ \; \mu^-$" + '\n' + r"$\sqrt{s} = 13 \, \mathrm{TeV}$", 
         fontsize=16, verticalalignment='top', horizontalalignment='left', 
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'),
         #fontname='consolas'
         )

    

    # Add a legend to distinguish between the two histograms
    plt.legend(loc='lower right', fontsize=20)
    
    # Save the figure with the correct process name
    if process_name == 'WW':
        figure_name = "WWmu-_e+_phi_histo_run_1.pdf"
    elif process_name == 'WZ':
        figure_name = "WZmu+_e+_phi_histo_run_2.pdf"
    else:
        figure_name = "ZZmu+_e+_phi_histo_run_4.pdf"
        
    figure_name_png = figure_name.replace(".pdf", ".png")
    
    figure_path = os.path.join(base_path, figure_name)
    figure_path_png = os.path.join(base_path, figure_name_png)
    
    plt.savefig(figure_path)
    plt.savefig(figure_path_png)

