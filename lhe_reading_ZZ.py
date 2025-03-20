import os
import subprocess
import pylhe
import numpy as np
from glob import glob


def run_madgraph(mg5_install_dir, process_dir, energy, nevents):
    """
    Modify the run_card and execute the MadGraph5 process.
    """
    run_card_path = os.path.join(process_dir, 'Cards', 'run_card.dat')
    
    # Modify run_card in-memory to avoid multiple file I/O
    with open(run_card_path, 'r') as file:
        run_card = file.readlines()
    
    energy_half = energy / 2
    for i, line in enumerate(run_card):
        if 'ebeam1' in line:
            run_card[i] = f"  {energy_half}    = ebeam1 ! beam 1 total energy in GeV\n"
        elif 'ebeam2' in line:
            run_card[i] = f"  {energy_half}    = ebeam2 ! beam 2 total energy in GeV\n"
        elif 'nevents' in line:
            run_card[i] = f"  {nevents}    = nevents ! Number of unweighted events requested\n"

    with open(run_card_path, 'w') as file:
        file.writelines(run_card)

    # Run the MadGraph process using subprocess, streamlined
    subprocess.run(
        [os.path.join(mg5_install_dir, 'bin', 'mg5_aMC')],
        input='launch pp_ZZ_SM\n', text=True,
        cwd=process_dir, check=True
    )


def find_latest_run_dir(base_dir):
    """
    Efficiently find the latest 'run_*' directory.
    """
    run_dirs = glob(os.path.join(base_dir, 'run_*'))
    
    if not run_dirs:
        raise FileNotFoundError("No run directories found.")
    
    # Extract run numbers and find the max
    run_numbers = [int(d.split('_')[-1]) for d in run_dirs]
    latest_run_number = max(run_numbers)
    latest_run_dir = f"run_{latest_run_number:02d}"
    
    return os.path.join(base_dir, latest_run_dir), latest_run_number


def read_lhe_write_data(lhe_file_path, particle_directories, number):
    """
    Read the LHE file and extract the relevant 4-momentum data into the respective particle folders.
    """
    init_info = pylhe.read_lhe_init(lhe_file_path)
    cross_section = init_info['procInfo'][0]['xSection']

    # Open files once and buffer writes for efficiency
    file_handles = {}
    for particle_id, directory in particle_directories.items():
        os.makedirs(directory, exist_ok=True)
        file_handles[particle_id] = open(os.path.join(directory, f"data_{number}.txt"), 'w')
        # Write cross section
        with open(os.path.join(directory, f"cross_sec_{number}.txt"), 'w') as f:
            f.write(f"{cross_section}\n")

    # Read LHE events and extract final state particles
    for event in pylhe.read_lhe_with_attributes(lhe_file_path):
        for particle in event.particles:
            if particle.status == 1:  # final state particle
                particle_id = particle.id
                if particle_id in file_handles:
                    data_line = f"{particle.e}, {particle.px}, {particle.py}, {particle.pz}\n"
                    file_handles[particle_id].write(data_line)

    # Close all file handles
    for f in file_handles.values():
        f.close()


def process_last_run(base_dir, particle_directories):
    """
    Process the latest run directory by reading its LHE file.
    """
    latest_run_dir, number = find_latest_run_dir(base_dir)
    print(f"Processing latest run directory: {latest_run_dir}")
    
    # Locate LHE file in the latest run directory
    lhe_file = glob(os.path.join(latest_run_dir, "*.lhe.gz"))
    if not lhe_file:
        raise FileNotFoundError("No LHE file found in the latest run directory.")
    
    lhe_file_path = lhe_file[0]
    print(f"Found LHE file: {lhe_file_path}")
    read_lhe_write_data(lhe_file_path, particle_directories, number)

def process_multiple_runs(base_dir, particle_directories, run_number_start, run_number_end):
    """
    Process multiple runs by reading their LHE files.
    """
    for number in range(run_number_start, run_number_end + 1):
        run_dir = os.path.join(base_dir, f"run_{number}")
        print(f"Processing run directory: {run_dir}")

        # Locate LHE file in the run directory
        lhe_file = glob(os.path.join(run_dir, "*.lhe.gz"))
        if not lhe_file:
            raise FileNotFoundError("No LHE file found in the run directory.")
        
        lhe_file_path = lhe_file[0]
        print(f"Found LHE file: {lhe_file_path}")
        read_lhe_write_data(lhe_file_path, particle_directories, number)

def combine_data(particle_directories, run_number_start, run_number_end):
    """
    Combine data from multiple runs for each particle.
    """
    for particle_id, directory in particle_directories.items():
        data_files = [os.path.join(directory, f"data_{i}.txt") for i in range(run_number_start, run_number_end + 1)]
        combined_data = np.concatenate([np.loadtxt(f, delimiter=',') for f in data_files])
        np.savetxt(os.path.join(directory, "combined_data_temp.txt"), combined_data, delimiter=',')

def add_data(particle_directories, run_number):
    """
    Add data from new runs to the existing combined data by appending directly to the file.
    """
    for particle_id, directory in particle_directories.items():
        data_file = os.path.join(directory, f"combined_data_temp.txt")
        combined_data_file = os.path.join(directory, "combined_data_new.txt")
        
        # Open the combined_data.txt file in append mode and write the new data
        with open(data_file, 'r') as new_data, open(combined_data_file, 'a') as combined_data:
            for line in new_data:
                combined_data.write(line)

def add_phase_points(directory, run_number):
    """
    Add phase points from run_number to combined data (ZZ_inv_mass_combined.txt and psi_data_combined.txt).
    """
    # Paths to the new data files
    cos_psi_file = os.path.join(directory, f"psi_data_combined_temp.txt")
    inv_mass_file = os.path.join(directory, f"ZZ_inv_mass_combined_temp.txt")
    
    # Paths to the combined data files
    combined_cos_psi_file = os.path.join(directory, "psi_data_combined_new.txt")
    combined_inv_mass_file = os.path.join(directory, "ZZ_inv_mass_combined_new.txt")
    
    # Append new data directly to the combined files
    with open(cos_psi_file, 'r') as cos_psi_data, open(combined_cos_psi_file, 'a') as combined_cos_psi:
        for line in cos_psi_data:
            combined_cos_psi.write(line)
    
    with open(inv_mass_file, 'r') as inv_mass_data, open(combined_inv_mass_file, 'a') as combined_inv_mass:
        for line in inv_mass_data:
            combined_inv_mass.write(line)

def add_angles(particle_directories, run_number):
    """
    Add decay angles after lorentz boost.
    """
    
    for particle_id, directory in {k: v for k, v in particle_directories.items() if k in [-13, -11]}.items():
        # Paths to the new data files
        cos_theta_file = os.path.join(directory, f"theta_data_combined_temp.txt")
        phi_file = os.path.join(directory, f"phi_data_combined_temp.txt")

        combined_cos_theta_file = os.path.join(directory, "theta_data_combined_new.txt")
        combined_phi_file = os.path.join(directory, "phi_data_combined_new.txt")
    
        # Append new data directly to the combined files
        with open(cos_theta_file, 'r') as cos_theta_data, open(combined_cos_theta_file, 'a') as combined_cos_theta:
            for line in cos_theta_data:
                combined_cos_theta.write(line)
        
        with open(phi_file, 'r') as phi_data, open(combined_phi_file, 'a') as combined_phi:
            for line in phi_data:
                combined_phi.write(line)

def main():
    mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
    process_dir = os.path.join(mg5_install_dir, "pp_ZZ_SM")
    base_dir = os.path.join(process_dir, "Events")
    
    particle_directories = {
        13: os.path.join(process_dir, "Plots and data/mu-"),  # mu-
        -13: os.path.join(process_dir, "Plots and data/mu+"), # mu+
        -11: os.path.join(process_dir, "Plots and data/e+"),   # e+
        11: os.path.join(process_dir, "Plots and data/e-"),  # e-
    }

    ENERGY = 13_000  # GeV
    NEVENTS = 1_000_000

    # Run MadGraph and process data
    # run_madgraph(mg5_install_dir, process_dir, ENERGY, NEVENTS)
    # process_last_run(base_dir, particle_directories)
    # process_multiple_runs(base_dir, particle_directories, 112, 135)
    # combine_data(particle_directories, 112, 135)
    add_data(particle_directories, 29)
    add_phase_points(os.path.join(process_dir, "Plots and data"), 29)
    add_angles(particle_directories, 29)


if __name__ == "__main__":
    main()
