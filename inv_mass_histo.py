import os
from pdb import run
import numpy as np
from lhe_reading_WW import find_latest_run_dir
from histo_plotter import read_data

# Update the MadGraph5 directory and process directory to ZZ_process
mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
process_dir = os.path.join(mg5_install_dir, "pp_ZZ_SM")  # Updated process directory
base_dir = os.path.join(process_dir, "Events")

# Particle directories corresponding to the ZZ process
particle_directories = {
    'mu+': os.path.join(process_dir, "Plots and data/mu+"),  # mu+
    'mu-': os.path.join(process_dir, "Plots and data/mu-"),  # mu-
    'e+': os.path.join(process_dir, "Plots and data/e+"),    # e+
    'e-': os.path.join(process_dir, "Plots and data/e-"),    # e-
}

run_number = 9

# Read the four-momenta data for all particles in the process
particle_arrays = {particle_name: read_data(os.path.join(directory, f"data_{run_number}.txt"))
                    for particle_name, directory in particle_directories.items()}

# Calculate the invariant mass of total system for each event
invariant_masses = []
for i in range(len(particle_arrays['mu+'])):
    total_momentum = np.zeros(4)
    for particle_array in particle_arrays.values():
        total_momentum += particle_array[i]
    invariant_masses.append(np.sqrt(total_momentum[0]**2 - total_momentum[1]**2 - total_momentum[2]**2 - total_momentum[3]**2))

# Plot histogram of invariant masses
import matplotlib.pyplot as plt
plt.hist(invariant_masses, bins=100)
plt.xlabel("Invariant mass (GeV)")
plt.ylabel("Frequency")

# Save the plot
plt.savefig(os.path.join(process_dir, "Plots and data", f"invariant_mass_{run_number}.pdf"))