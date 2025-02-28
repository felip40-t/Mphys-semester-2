import os
import numpy as np
from lhe_reading_WZ import find_latest_run_dir
from histo_plotter import read_data

# Update the MadGraph5 directory and process directory
mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
process_dir = os.path.join(mg5_install_dir, "pp_ZZ_fiducial")  # Updated process directory
base_dir = os.path.join(process_dir, "Events")

_, run_number = find_latest_run_dir(base_dir)

# Update particle directories to reflect the correct particles for z bosons
particle_directories = {
    'mu+': os.path.join(process_dir, "Plots and data/mu+"),  # mu+
    'mu-': os.path.join(process_dir, "Plots and data/mu-"),  # mu-
    'e+': os.path.join(process_dir, "Plots and data/e+"),    # e+
    'e-': os.path.join(process_dir, "Plots and data/e-"),    # e-
    'z1': os.path.join(process_dir, "Plots and data/z1"),    # z boson
    'z2': os.path.join(process_dir, "Plots and data/z2"),      # z boson 
}

# Initialize particle arrays for storing the 4-momentum data
particle_arrays = {
    'mu+': np.array([]),
    'mu-': np.array([]),
    'e+': np.array([]),
    'e-': np.array([]),
    'z1': np.array([]),  # For storing the reconstructed z boson
    'z2': np.array([]),   # For storing the reconstructed z boson
}

def extract_data():
    """
    Extract the 4-momentum data from each particle's data file.
    """
    for particle_name, directory in particle_directories.items():
        if particle_name not in ['z1', 'z2']:  # Only read data for mu+, mu-, e+, e-
            data_file = os.path.join(directory, f"data_{run_number}.txt")
            particle_arrays[particle_name] = read_data(data_file)

def reconstruct_particles():
    """
    Reconstruct the four-momenta of the z bosons.
    - z1 = e+ + e-
    - z2  = mu+ + mu-
    """
    particle_arrays['z1'] = particle_arrays['e+'] + particle_arrays['e-']
    particle_arrays['z2'] = particle_arrays['mu+'] + particle_arrays['mu-']

def write_particle(particle_name):
    """
    Write the reconstructed four-momentum of the particle to a file.
    """
    file_path = os.path.join(particle_directories[particle_name], f"data_{run_number}.txt")
    with open(file_path, 'w') as f:  # Open the file once for writing
        for line in particle_arrays[particle_name]:
            data_line = f"{line[0]}, {line[1]}, {line[2]}, {line[3]}\n"
            f.write(data_line)

def main():
    # Extract 4-momenta of individual particles (mu+, mu-, e+, e-)
    extract_data()

    # Reconstruct the four-momenta of z bosons
    reconstruct_particles()

    # Write the reconstructed four-momenta to respective files
    write_particle('z1')
    write_particle('z2')

if __name__ == "__main__":
    main()