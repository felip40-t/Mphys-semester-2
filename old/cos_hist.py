import os
import numpy as np
import matplotlib.pyplot as plt
from histo_plotter import read_data

# Define the base paths for e- and e+ directories
base_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_WW_SM/Plots and data"

# Define the data directories
data_dir_e_plus = os.path.join(base_path, "e+/theta_data_1.txt")
data_dir_mu_minus = os.path.join(base_path, "mu-/theta_data_1.txt")


# Read data for both e- and e+ from run 10
cos_theta_values_mu_minus = read_data(data_dir_mu_minus)
cos_theta_values_e_plus = read_data(data_dir_e_plus)

# Number of bins for the histogram
num_bins = 40

# Create the histograms
counts_e_minus, bin_edges_e_minus = np.histogram(cos_theta_values_mu_minus, bins=num_bins)
counts_e_plus, bin_edges_e_plus = np.histogram(cos_theta_values_e_plus, bins=num_bins)

# Plot the histograms on the same figure
plt.figure(figsize=(10, 7), dpi=800)

# Plot histogram for e-
plt.hist(cos_theta_values_mu_minus, bins=num_bins, density=True, histtype='step', edgecolor='blue', label=r"$\mu^-$")

# Plot histogram for e+
plt.hist(cos_theta_values_e_plus, bins=num_bins, density=True, histtype='step', edgecolor='red', label=r"$e^+$")

# Set the x-axis label (cos(theta)) and the y-axis label (normalized differential cross section)
plt.xlabel(r'$\cos\theta$', fontsize=20)
plt.ylabel(r'$1/\sigma{\cdot}d\sigma/d\cos\theta$', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='y', labelright=False, right=True)

# Add grid lines 
plt.grid(axis='y')

# Set axes limits
plt.ylim(0, 1.0)
plt.xlim(-1, 1)

plt.text(-0.95, 0.95, r"$ p \; p \; \to \; e^+ \; \nu_e \; \mu^- \; \bar{\nu}_{\mu}$" + '\n' + r"$\sqrt{s} = 13 \, \mathrm{TeV}$", 
         fontsize=16, verticalalignment='top', horizontalalignment='left', 
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'),
         #fontname='consolas'
         )

# Add a legend to distinguish between the two histograms (e- and e+)
plt.legend(loc='lower right', fontsize=20)

# Save the figure
figure_path = os.path.join(base_path, "WW_mu-_e+_cos_histo_run_1.pdf")
figure_path_png = os.path.join(base_path, "WW_mu-_e+_cos_histo_run_1.png")


plt.savefig(figure_path)
plt.savefig(figure_path_png)




