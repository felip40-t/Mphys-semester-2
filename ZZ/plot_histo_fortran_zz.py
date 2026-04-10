import os
import numpy as np
import matplotlib.pyplot as plt

# Function to read data from the file
def read_data(filepath):
    return np.loadtxt(filepath)

# Function to parse .HwU file for the desired histogram
def parse_hwU_file(filename, target_histogram):
    with open(filename, 'r') as f:
        histogram_found = False
        data = []

        for line in f:
            # Check if the line contains the target histogram title
            if line.startswith('<histogram>'):
                if target_histogram in line:
                    histogram_found = True  # Start parsing the relevant histogram data
                else:
                    histogram_found = False  # Stop if another histogram starts
                
            # Only read lines related to the target histogram
            if histogram_found and not line.startswith('<histogram>') and line.strip():
                # Split the line into its components (only first 4 values)
                components = line.split()
                
                # We expect at least 4 values: 2 for bin edges, 1 for the central cross section, 1 for uncertainty
                if len(components) >= 4:
                    bin_lower_edge = float(components[0])
                    bin_upper_edge = float(components[1])
                    central_cross_section = float(components[2])
                    uncertainty = float(components[3])
                    
                    # Store the parsed values (bin edges, central value, uncertainty)
                    data.append((bin_lower_edge, bin_upper_edge, central_cross_section, uncertainty))

    return data

# Function to plot the histograms
def plot_combined_histogram(hwU_data, phi_values_e_plus, filepath):
    # Unpack .HwU data into bin edges, cross sections
    bin_lower_edges = [entry[0] for entry in hwU_data]
    bin_upper_edges = [entry[1] for entry in hwU_data]
    central_values = [entry[2] for entry in hwU_data]
    
    # Calculate the total cross section for normalization
    total_cross_section = sum(central_values)
    
    # Calculate bin widths
    bin_widths = [upper - lower for lower, upper in zip(bin_lower_edges, bin_upper_edges)]
    
    # Make the central values differential by dividing by the bin width
    differential_values = [val / width for val, width in zip(central_values, bin_widths)]
    
    # Normalize the differential values by dividing by total cross section
    normalized_values = [val / total_cross_section for val in differential_values]
    
    # Calculate bin centers for plotting
    bin_centers = [(lower + upper) / 2 for lower, upper in zip(bin_lower_edges, bin_upper_edges)]
    
    bin_centers = np.array(bin_centers)
    normalized_values = np.array(normalized_values)

    plt.figure(figsize=(12, 8), dpi=800)

    # Plot normalized central values from HwU file (blue)
    plt.hist(bin_centers, bins=bin_lower_edges + [bin_upper_edges[-1]], 
             weights=normalized_values, histtype='step', edgecolor='blue', label='Fortran')

    # Plot histogram for e+ data from the separate file (red)
    plt.hist(phi_values_e_plus, bins=len(bin_lower_edges), density=True, histtype='step', edgecolor='red', label='Python')

    # Set the x-axis label and y-axis label
    plt.xlabel(r'$\phi^*_{\mu^+}$', fontsize=20)
    plt.ylabel(r'$1/\sigma \cdot d\sigma/d\phi^*_{\mu^+}$', fontsize=20)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(0.1, 0.2)
    plt.yticks([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])
    
    # Add plot title and text
    plt.text(-3, 0.195, r"$p \; p \; \to \; e^+ \; e^- \; \mu^+ \mu^-$" + '\n' + r"$\sqrt{s} = 13 \, \mathrm{TeV}$", 
         fontsize=16, verticalalignment='top', horizontalalignment='left', 
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'))
    
    # Add a legend to distinguish between the two histograms
    plt.legend(loc='lower right', fontsize=20)
    
    # Add grid lines
    plt.grid(True)
    
    # Save the figure
    path = os.path.join(filepath, "phi_hist_comparison_fortran_mu+_zz_run4.png")
    plt.savefig(path)
    path = os.path.join(filepath, "phi_hist_comparison_fortran_mu+_zz_run4.pdf")
    plt.savefig(path)

file_path = os.path.join("/home/felipetcach/project/MG5_aMC_v3_5_6/pp_4l_zz_LOonly/Events/run_04_LO", "MADatNLO.HwU")
target_histogram = 'phi st mp T INC    LONP0'

# Parse the .HwU file for the desired histogram
parsed_hwU_data = parse_hwU_file(file_path, target_histogram)

# Read e+ data for Python
wz_process = {
    "base_path": "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_ZZ_SM/Plots and data",
    "particles": {"mu+": "mu+/phi_data_4.txt"},
}
phi_values_e_plus = read_data(os.path.join(wz_process["base_path"], wz_process["particles"]["mu+"]))

# Plot the combined histogram (Fortran and Python)
plot_combined_histogram(parsed_hwU_data, phi_values_e_plus, "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_4l_zz_LOonly")

# Model function
def model_function(phi,a22):
    return 1/(2*np.pi) + a22*np.cos(2*phi)*(np.sqrt(30*np.pi)/(6*np.pi))