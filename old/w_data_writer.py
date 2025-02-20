import os
import numpy as np
from lhe_reading_WZ import find_latest_run_dir
from histo_plotter import read_data

mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
process_dir = os.path.join(mg5_install_dir, "WW_process")
base_dir = os.path.join(process_dir, "Events")

_, run_number = find_latest_run_dir(base_dir)

particle_directories = {
    'e-': os.path.join(process_dir, "Plots and data/e-"),  # e-
    'e+': os.path.join(process_dir, "Plots and data/e+"), # e+
    've': os.path.join(process_dir, "Plots and data/ve"),  # ve
    've~': os.path.join(process_dir, "Plots and data/ve~"), # ve~
    'w+': os.path.join(process_dir, "Plots and data/w+"),  # w+
    'w-': os.path.join(process_dir, "Plots and data/w-"), # w-
    }

particle_arrays = {
    'e-': np.array([]),  # e-
    'e+': np.array([]), # e+
    've': np.array([]),  # ve
    've~': np.array([]), # ve~
    'w+': np.array([]),  # w+
    'w-': np.array([]), # w-
    }

def extract_data():
    for particle_name, directory in list(particle_directories.items())[:-2]:
        data_file = os.path.join(directory, f"data_{run_number}.txt")
        particle_arrays[particle_name] = read_data(data_file)

def reconstruct_w():
    particle_arrays['w-'] = particle_arrays['e-'] + particle_arrays['ve~']
    particle_arrays['w+'] = particle_arrays['e+'] + particle_arrays['ve']

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
    extract_data()
    reconstruct_w()
    write_particle('w+')
    write_particle('w-')

main()