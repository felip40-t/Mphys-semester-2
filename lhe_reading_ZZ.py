from multiprocessing import process
import os
import subprocess
import sys
import pylhe
import numpy as np
from coefficient_calculator_ZZ import read_masked_data
import glob
from lorentz_boost_zz import boostinvp, calc_inv_mass, calc_scattering_angle, phistar
import tarfile


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
    run_dirs = glob.glob(os.path.join(base_dir, 'run_*'))
    
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

def process_and_combine_runs(base_dir, particle_directories, run_number_start, run_number_end, batch_size=250000):
    """
    Process multiple runs by reading their LHE files and combine data directly into combined_data_temp
    without keeping all data in memory at once. Data is written incrementally to avoid memory exhaustion.
    
    Parameters:
    - base_dir: The base directory containing run directories.
    - particle_directories: A dictionary where keys are particle IDs and values are directories for storing combined data.
    - run_number_start: The starting run number.
    - run_number_end: The ending run number.
    - batch_size: The number of events to accumulate in memory before writing to disk.
    """
    # Make sure output directories exist and open files for each particle
    output_files = {}
    for particle_id, directory in particle_directories.items():
        os.makedirs(directory, exist_ok=True)
        combined_data_file = os.path.join(directory, "combined_data_temp.txt")
        # Open file in append mode
        output_files[particle_id] = open(combined_data_file, 'a')

    try:
        for number in range(run_number_start, run_number_end + 1):
            run_dir = os.path.join(base_dir, f"run_{number}")
            print(f"Processing run directory: {run_dir}")

            # Locate LHE file in the run directory
            lhe_file = glob.glob(os.path.join(run_dir, "*.lhe.gz"))
            if not lhe_file:
                raise FileNotFoundError(f"No LHE file found in the run directory: {run_dir}")
            
            lhe_file_path = lhe_file[0]
            print(f"Found LHE file: {lhe_file_path}")

            # Read events in batches and write to disk incrementally
            event_buffer = {particle_id: [] for particle_id in particle_directories.keys()}

            for event in pylhe.read_lhe_with_attributes(lhe_file_path):
                for particle in event.particles:
                    if particle.status == 1:  # final state particle
                        particle_id = particle.id
                        if particle_id in event_buffer:
                            data_line = [particle.e, particle.px, particle.py, particle.pz]
                            event_buffer[particle_id].append(data_line)

                            # Write in batches
                            if len(event_buffer[particle_id]) >= batch_size:
                                np.savetxt(output_files[particle_id], event_buffer[particle_id], delimiter=',')
                                event_buffer[particle_id] = []  # Clear buffer

            # Write any remaining data in the buffer (less than batch_size)
            for particle_id in event_buffer:
                if event_buffer[particle_id]:
                    np.savetxt(output_files[particle_id], event_buffer[particle_id], delimiter=',')
                    event_buffer[particle_id] = []

    finally:
        # Close all output files
        for f in output_files.values():
            f.close()

def add_data(particle_directories):
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

def add_phase_points(directory):
    """
    Add phase points to combined data (ZZ_inv_mass_combined.txt and psi_data_combined.txt).
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

def add_angles(particle_directories):
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


def filter_data(particle_directories, directory):
    """
    Check invariant mass and scattering angle and delete events in saturated region.
    """
    # Define file paths
    combined_cos_psi_file = os.path.join(directory, "psi_data_combined_new.txt")
    combined_inv_mass_file = os.path.join(directory, "ZZ_inv_mass_combined_new.txt")

    # Load data once
    cos_psi_data = np.loadtxt(combined_cos_psi_file)
    inv_mass_data = np.loadtxt(combined_inv_mass_file)

    # Define saturated regions
    saturated_region_cos_psi = (0.99, 1.0)
    saturated_region_inv_mass = (550.0, 700.0)

    # Generate mask
    mask = read_masked_data(cos_psi_data, inv_mass_data, saturated_region_cos_psi, saturated_region_inv_mass)

    # Randomly keep 75% of the excluded events
    excluded_indices = np.where(mask)[0]
    keep_indices = np.random.choice(excluded_indices, size=int(len(excluded_indices) * 0.7), replace=False)
    mask[keep_indices] = False


    # Save filtered data
    filtered_cos_psi_data = cos_psi_data[~mask]
    filtered_inv_mass_data = inv_mass_data[~mask]
    np.savetxt(combined_cos_psi_file, filtered_cos_psi_data)
    np.savetxt(combined_inv_mass_file, filtered_inv_mass_data)

    # Process particle-specific files
    for particle_id, directory in particle_directories.items():
        # Process combined data file
        data_file = os.path.join(directory, "combined_data_new.txt")
        if os.path.exists(data_file):
            np.savetxt(data_file, np.loadtxt(data_file, delimiter=',')[~mask], delimiter=',')

        # Process theta and phi files
        for file_name in ["theta_data_combined_new.txt", "phi_data_combined_new.txt"]:
            file_path = os.path.join(directory, file_name)
            if os.path.exists(file_path):
                np.savetxt(file_path, np.loadtxt(file_path)[~mask])

def reorganise_data(particle_directories, directory):
    """
    Check invariant mass and scattering angle and reorganise events.
    """
    regions = [
        [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for cos_min in [0.0 + 0.1 * i for i in range(9)]
        for mass_min in [200.0 + 50.0 * j for j in range(16)]
    ]

    # regions += [
    #     [(0.9, 0.95), (mass_min, mass_min + 100.0)]
    #     for mass_min in [200.0 + 100.0 * j for j in range(8)]
    # ]

    reorganised_path = os.path.join(directory, "reorganised_data")
    # Define file paths
    combined_cos_psi_file = os.path.join(directory, "psi_data_combined_new.txt")
    combined_inv_mass_file = os.path.join(directory, "ZZ_inv_mass_combined_new.txt")

    # Load data once
    cos_psi_data = np.loadtxt(combined_cos_psi_file)
    inv_mass_data = np.loadtxt(combined_inv_mass_file)

    for region in regions:
        print("Processing region:", region)
        new_path = os.path.join(reorganised_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
        os.makedirs(new_path, exist_ok=True)

        # Generate mask
        mask = read_masked_data(cos_psi_data, inv_mass_data, region[0], region[1])
        print("Size of region:", np.sum(mask))
        # Save filtered data
        filtered_cos_psi_data = cos_psi_data[mask]
        filtered_inv_mass_data = inv_mass_data[mask]
        np.savetxt(os.path.join(new_path, "psi_data_combined_new.txt"), filtered_cos_psi_data)
        np.savetxt(os.path.join(new_path, "ZZ_inv_mass_combined_new.txt"), filtered_inv_mass_data)

        # Process particle-specific files
        for particle_id, particle_directory in particle_directories.items():
            
            # Process theta and phi files
            for file_name in ["theta_data_combined_new.txt", "phi_data_combined_new.txt"]:
                file_path = os.path.join(particle_directory, file_name)
                if os.path.exists(file_path):
                    if particle_id == -11:
                        # Filter data for e+
                        filtered_data = np.loadtxt(file_path)[mask]
                        new_file_path = os.path.join(new_path, f"e+_{file_name}")
                        np.savetxt(new_file_path, filtered_data)
                    elif particle_id == -13:
                        # Filter data for mu+
                        filtered_data = np.loadtxt(file_path)[mask]
                        new_file_path = os.path.join(new_path, f"mu+_{file_name}")
                        np.savetxt(new_file_path, filtered_data)
                

def read_boost_data(Events_dir, reorganised_path):
    """
    Function to read data from lhe files, boost them, calculate decay angles, and save them to the correct region as in reorganise_data.
    """
    # Define order of regions
    regions = [
        [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for cos_min in [0.0 + 0.1 * i for i in range(9)]
        for mass_min in [200.0 + 50.0 * j for j in range(16)]
    ]

    run_start = 2

    for i, region in enumerate(regions):
        print("Processing region:", region)
        run_directory = os.path.join(Events_dir, f"run_{(run_start + i):02d}")
        # Check that the run directory exists
        if not os.path.exists(run_directory):
            print(f"Error: Run directory {run_directory} does not exist. Skipping.")
            continue
        else:
            print(f"Run directory {run_directory} exists. Proceeding with processing.")
        # Create the save directory for the current region
        save_dir = os.path.join(reorganised_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
        
        # Check that the save directory exists
        if not os.path.exists(save_dir):
            print(f"Error: Save directory {save_dir} does not exist. Skipping run directory {run_directory}.")
            continue
        else:
            print(f"Save directory {save_dir} exists. Proceeding with processing.")

        # Initialize numpy arrays for data
        cos_psi_data = np.array([])
        inv_mass_data = np.array([])
        ep_theta_data = np.array([])
        ep_phi_data = np.array([])
        mp_theta_data = np.array([])
        mp_phi_data = np.array([])

        # Read lhe file
        try:
            lhe_file = glob.glob(os.path.join(run_directory, "*.lhe.gz"))
            if not lhe_file:
                raise FileNotFoundError(f"No LHE file found in the run directory: {run_directory}")
            lhe_file_path = lhe_file[0]
            print(f"Found LHE file: {lhe_file_path}")
        except Exception as e:
            print(f"Error locating LHE file: {e}")
            continue

        try:
            lhe_data = pylhe.read_lhe_with_attributes(lhe_file_path)
        except Exception as e:
            print(f"Error reading LHE file {lhe_file_path}: {e}")
            continue

        # Process events
        try:
            for event in lhe_data:
                # dict to store data
                particle_data = {
                    11: [],
                    -11: [],
                    13: [],
                    -13: []
                }
                # Read particles
                for particle in event.particles:
                    if particle.status == 1:
                        particle_data[particle.id].append(particle.e)
                        particle_data[particle.id].append(particle.px)
                        particle_data[particle.id].append(particle.py)
                        particle_data[particle.id].append(particle.pz)
                # Reconstruct diboson frame and first z
                diboson = diboson = np.array(particle_data[-11]) + np.array(particle_data[11]) + np.array(particle_data[-13]) + np.array(particle_data[13])
                z1 = np.array(particle_data[-11]) + np.array(particle_data[11])
                z1_boosted = np.zeros(4)
                # Boost z into diboson CM frame
                boostinvp(z1, diboson, z1_boosted)
                cos_psi = calc_scattering_angle(np.array(z1_boosted))
                cos_psi_data = np.append(cos_psi_data, cos_psi)
                inv_mass = calc_inv_mass(np.array(diboson))
                inv_mass_data = np.append(inv_mass_data, inv_mass)
                # Calculate decay angles
                ep_phi, mp_phi, ep_theta, mp_theta = phistar(np.array(particle_data[11]), np.array(particle_data[-11]), np.array(particle_data[13]), np.array(particle_data[-13]))
                ep_phi_data = np.append(ep_phi_data, ep_phi)
                mp_phi_data = np.append(mp_phi_data, mp_phi)
                ep_theta_data = np.append(ep_theta_data, ep_theta)
                mp_theta_data = np.append(mp_theta_data, mp_theta)
        except Exception as e:
            import traceback
            print(f"Error processing events in {lhe_file_path}: {e}")
            print("Traceback:")
            traceback.print_exc()
            sys.exit(1)
            

        # Append decay angle data to files with error handling
        try:
            with open(os.path.join(save_dir, "e+_phi_data_combined_new.txt"), 'a') as f:
                np.savetxt(f, ep_phi_data)
            with open(os.path.join(save_dir, "mu+_phi_data_combined_new.txt"), 'a') as f:
                np.savetxt(f, mp_phi_data)
            with open(os.path.join(save_dir, "e+_theta_data_combined_new.txt"), 'a') as f:
                np.savetxt(f, ep_theta_data)
            with open(os.path.join(save_dir, "mu+_theta_data_combined_new.txt"), 'a') as f:
                np.savetxt(f, mp_theta_data)
            with open(os.path.join(save_dir, "psi_data_combined_new.txt"), 'a') as f:
                np.savetxt(f, cos_psi_data)
            with open(os.path.join(save_dir, "ZZ_inv_mass_combined_new.txt"), 'a') as f:
                np.savetxt(f, inv_mass_data)
        except Exception as e:
            print(f"Error writing output files in directory {save_dir}: {e}")


def main():
    mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
    process_dir = os.path.join(mg5_install_dir, "pp_ZZ_SM")
    # Open the tar file without extracting all files

    # "\\wsl.localhost\Ubuntu\home\felipetcach\project\MG5_aMC_v3_5_6\Felipe_pp_ZZ_4l.tar.gz"

    tar_path = os.path.join(mg5_install_dir, "Felipe_pp_ZZ_4l.tar.gz")
    with tarfile.open(tar_path, "r") as tar:
        # Extract only the Events directory to a temporary location
        temp_dir = "/tmp/Felipe_pp_ZZ_4l_Events"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        members = [member for member in tar.getmembers() if member.name.startswith("Felipe_pp_ZZ_4l/Events")]
        total_members = len(members)
        for i, member in enumerate(members, start=1):
            tar.extract(member, path=temp_dir)
            print(f"Extracting {i}/{total_members}: {member.name}")

    new_data_dir = os.path.join(temp_dir, "Felipe_pp_ZZ_4l/Events")
    # base_dir = os.path.join(process_dir, "Events")
    
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
    # process_and_combine_runs(base_dir, particle_directories, 309, 452)
    # add_phase_points(os.path.join(process_dir, "Plots and data"))
    # add_angles(particle_directories)
    # filter_data(particle_directories, os.path.join(process_dir, "Plots and data"))
    # reorganise_data(particle_directories, os.path.join(process_dir, "Plots and data"))
    read_boost_data(new_data_dir, os.path.join(process_dir, "Plots and data/reorganised_data"))


if __name__ == "__main__":
    main()
