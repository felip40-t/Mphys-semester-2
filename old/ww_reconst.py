import os
import numpy as np
from lhe_reading_WZ import find_latest_run_dir
from histo_plotter import read_data

# Update the MadGraph5 directory and process directory
mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
process_dir = os.path.join(mg5_install_dir, "pp_WW_SM")  # Updated process directory
base_dir = os.path.join(process_dir, "Events")

_, run_number = find_latest_run_dir(base_dir)

# Update particle directories to reflect the correct particles for w+ and z bosons
particle_directories = {
    'vm~': os.path.join(process_dir, "Plots and data/vm~"),  # mu+
    'mu-': os.path.join(process_dir, "Plots and data/mu-"),  # mu-
    'e+': os.path.join(process_dir, "Plots and data/e+"),    # e+
    've': os.path.join(process_dir, "Plots and data/ve"),    # ve
    'w+': os.path.join(process_dir, "Plots and data/w+"),    # w+ (lowercase)
    'w-': os.path.join(process_dir, "Plots and data/w-"),      # z boson (lowercase)
}

# Initialize particle arrays for storing the 4-momentum data
particle_arrays = {
    'vm~': np.array([]),
    'mu-': np.array([]),
    'e+': np.array([]),
    've': np.array([]),
    'w+': np.array([]),  # For storing the reconstructed w+
    'w-': np.array([]),   # For storing the reconstructed z boson
}

def extract_data():
    """
    Extract the 4-momentum data from each particle's data file.
    """
    for particle_name, directory in particle_directories.items():
        if particle_name not in ['w+', 'w-']:  # Only read data for mu+, mu-, e+, ve
            data_file = os.path.join(directory, f"data_{run_number}.txt")
            particle_arrays[particle_name] = read_data(data_file)

def reconstruct_particles():
    """
    Reconstruct the four-momenta of the w+ and w- bosons.
    - w+ = e+ + ve
    - w-  = mu- + vm~
    """
    particle_arrays['w+'] = particle_arrays['e+'] + particle_arrays['ve']
    particle_arrays['w-'] = particle_arrays['mu-'] + particle_arrays['vm~']

def write_particle(particle_name):
    """
    Write the reconstructed four-momentum of the particle (w+ or w-) to a file.
    """
    file_path = os.path.join(particle_directories[particle_name], f"data_{run_number}.txt")
    with open(file_path, 'w') as f:  # Open the file once for writing
        for line in particle_arrays[particle_name]:
            data_line = f"{line[0]}, {line[1]}, {line[2]}, {line[3]}\n"
            f.write(data_line)

def main():
    # Extract 4-momenta of individual particles 
    extract_data()

    # Reconstruct the four-momenta of w+ and w- bosons
    reconstruct_particles()

    # Write the reconstructed four-momenta to respective files
    write_particle('w+')
    write_particle('w-')

if __name__ == "__main__":
    main()
