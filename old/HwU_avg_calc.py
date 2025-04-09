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
def calc_average(hwU_data):
    # Unpack .HwU data into bin edges, cross sections
    bin_lower_edges = [entry[0] for entry in hwU_data]
    bin_upper_edges = [entry[1] for entry in hwU_data]
    central_values = [entry[2] for entry in hwU_data]
    uncertainties = [entry[3] for entry in hwU_data]
    
    # Calculate the total cross section for normalization
    total_cross_section = sum(central_values)
    
    # Calculate bin centers
    bin_centers = [(lower + upper) / 2 for lower, upper in zip(bin_lower_edges, bin_upper_edges)]
    
    bin_centers = np.array(bin_centers)

    # Calculate the average
    average = np.sum(np.array(central_values) * np.array(bin_centers)) / total_cross_section

    # Calculate the uncertainty
    uncertainty = np.sqrt(np.sum(np.array(uncertainties)**2)) / total_cross_section

    return average, uncertainty
    

file_path = os.path.join("/home/felipetcach/project/MG5_aMC_v3_5_6/pp_4l_zz_LOonly/Events/run_26_LO", "MADatNLO.HwU")
target_histogram1 = 'cos(2\phi1)        LONP0'
target_histogram2 = 'heta1)^2     LONP0'

# Parse the .HwU file for the desired histogram
parsed_hwU_data_phi = parse_hwU_file(file_path, target_histogram1)
parsed_hwU_data_theta = parse_hwU_file(file_path, target_histogram2)

# Calculate the average
average_phi, phi_unc = calc_average(parsed_hwU_data_phi)
average_theta, th_unc = calc_average(parsed_hwU_data_theta)

print("Average for cos(2*phi1):", average_phi, "+-", phi_unc)
print("Average for cos(theta1)^2:", average_theta, "+-", th_unc)

coeff = 0.25 * (np.sqrt(15/(2*np.pi))) * average_phi * (1 - average_theta)
coeff_unc = 0.25 * (np.sqrt(15/(2*np.pi))) * np.sqrt(average_phi**2 * th_unc**2 + average_theta**2 * phi_unc**2)
print("Coefficient:", coeff, "+-", coeff_unc)
