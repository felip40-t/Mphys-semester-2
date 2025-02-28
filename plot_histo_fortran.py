from zipfile import Path
import numpy as np
import matplotlib.pyplot as plt
import os

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

def plot_histogram(data, filepath):
    # Unpack data into bin edges, cross sections, and uncertainties
    bin_lower_edges = [entry[0] for entry in data]
    bin_upper_edges = [entry[1] for entry in data]
    central_values = [entry[2] for entry in data]
    uncertainties = [entry[3] for entry in data]
    
    # Calculate the total cross section for normalization
    total_cross_section = sum(central_values)
    print(total_cross_section)
    
    # Calculate bin widths
    bin_widths = [upper - lower for lower, upper in zip(bin_lower_edges, bin_upper_edges)]
    
    # Make the central values differential by dividing by the bin width
    differential_values = [val / width for val, width in zip(central_values, bin_widths)]
    
    # Make uncertainties differential by dividing by the bin width
    differential_uncertainties = [unc / width for unc, width in zip(uncertainties, bin_widths)]
    
    # Normalize the differential values and uncertainties by dividing by total cross section
    normalized_values = [val / total_cross_section for val in differential_values]
    normalized_uncertainties = [unc / total_cross_section for unc in differential_uncertainties]
    
    # Calculate bin centers for plotting
    bin_centers = [(lower + upper) / 2 for lower, upper in zip(bin_lower_edges, bin_upper_edges)]
    
    # Convert data to numpy arrays for easier plotting
    bin_centers = np.array(bin_centers)
    normalized_values = np.array(normalized_values)
    normalized_uncertainties = np.array(normalized_uncertainties)
    
    # Plot the histograms with 'step' style and add error bars manually
    plt.figure(figsize=(12, 8), dpi=800)

    # Plot normalized central values as a step plot (histtype='step')
    plt.hist(bin_centers, bins=bin_lower_edges + [bin_upper_edges[-1]], 
             weights=normalized_values, histtype='step', edgecolor='blue', label='Fortran')
    
    # Add normalized error bars manually using plt.errorbar
    #plt.errorbar(bin_centers, normalized_values, yerr=normalized_uncertainties, fmt='o', color='blue', capsize=3, label='Uncertainty')

    # Set the x-axis label (phi) and y-axis label (differential cross section)
    plt.xlabel(r'$\phi$', fontsize=20)
    plt.ylabel(r'$1/\sigma \cdot d\sigma/d\phi$', fontsize=20)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(0.1,0.2)
    plt.text(-3, 0.19, r"$p \; p \; \to \; e^+ \; \nu_e \; \mu^+ \mu^-$" + '\n' + r"$\sqrt{s} = 13 \, \mathrm{TeV}$", 
         fontsize=16, verticalalignment='top', horizontalalignment='left', 
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5')
         )
    plt.legend(loc='lower right', fontsize=20)
    plt.grid(True)
    path = os.path.join(filepath, "phi_hist_e+_inc_T_comparison.pdf")
    plt.savefig(path)


# Example usage
file_path = os.path.join("/home/felipetcach/project/MG5_aMC_v3_5_6/pp_4l_wz_LOonly/Events/run_03_LO", "MADatNLO.HwU")
target_histogram = 'phi st ep T INC    LONP0'

# Parse the .HwU file for the desired histogram
parsed_data = parse_hwU_file(file_path, target_histogram)

# Plot the histogram with the parsed data
plot_histogram(parsed_data, "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_4l_wz_LOonly")
