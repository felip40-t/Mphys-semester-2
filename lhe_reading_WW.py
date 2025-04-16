import os 
import sys
import subprocess
import trace
import pylhe
import numpy as np
from lorentz_boost_zz import boostinvp, calc_inv_mass, calc_scattering_angle, phistar
import glob
from multiprocessing import Pool
import traceback
import logging

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
        input='launch pp_WW_SM\n', text=True,
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
        file_handles[particle_id] = open(os.path.join(directory, f"data_{number}.txt"), 'a')
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




def read_boost_data(Events_dir, reorganised_path):
    """
    Function to read data from lhe files, boost them, calculate decay angles, and save them to the correct region as in organised_data.
    """
    # Define order of regions
    regions = [
        [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for cos_min in [0.0 + 0.1 * i for i in range(9)]
        for mass_min in [200.0 + 50.0 * j for j in range(20)]
    ]

    run_start = 2

    for i, region in enumerate(regions):
        print("Processing region:", region)
        run_directory = os.path.join(Events_dir, f"run_{(run_start + i):02d}")
        # Check that the run directory exists
        if not os.path.exists(run_directory):
            print(f"Error: Run directory {run_directory} does not exist. Skipping.")
            continue
        # Create the save directory for the current region
        save_dir = os.path.join(reorganised_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Save directory created: {save_dir}")
        # Check that the save directory exists
        if not os.path.exists(save_dir):
            print(f"Error: Save directory {save_dir} does not exist. Skipping run directory {run_directory}.")
            continue

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
                    -11: [],
                    12: [],
                    13: [],
                    -14: []
                }
                # Read particles
                for particle in event.particles:
                    if particle.status == 1:
                        particle_data[particle.id].append(particle.e)
                        particle_data[particle.id].append(particle.px)
                        particle_data[particle.id].append(particle.py)
                        particle_data[particle.id].append(particle.pz)
                # Reconstruct diboson frame and first w
                diboson = np.array(particle_data[-11]) + np.array(particle_data[12]) + np.array(particle_data[13]) + np.array(particle_data[-14])
                w1 = np.array(particle_data[-11]) + np.array(particle_data[12])
                w1_boosted = np.zeros(4)
                # Boost W into diboson CM frame
                boostinvp(w1, diboson, w1_boosted)
                cos_psi = calc_scattering_angle(np.array(w1_boosted))
                cos_psi_data = np.append(cos_psi_data, cos_psi)
                inv_mass = calc_inv_mass(np.array(diboson))
                inv_mass_data = np.append(inv_mass_data, inv_mass)
                # Calculate decay angles
                ep_phi, mp_phi, ep_theta, mp_theta = phistar(np.array(particle_data[-11]), np.array(particle_data[12]), np.array(particle_data[13]), np.array(particle_data[-14]))
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
            with open(os.path.join(save_dir, "e+_phi_data_combined.txt"), 'w') as f:
                np.savetxt(f, ep_phi_data)
            with open(os.path.join(save_dir, "mu-_phi_data_combined.txt"), 'w') as f:
                np.savetxt(f, mp_phi_data)
            with open(os.path.join(save_dir, "e+_theta_data_combined.txt"), 'w') as f:
                np.savetxt(f, ep_theta_data)
            with open(os.path.join(save_dir, "mu-_theta_data_combined.txt"), 'w') as f:
                np.savetxt(f, mp_theta_data)
            with open(os.path.join(save_dir, "psi_data_combined.txt"), 'w') as f:
                np.savetxt(f, cos_psi_data)
            with open(os.path.join(save_dir, "ZZ_inv_mass_combined.txt"), 'w') as f:
                np.savetxt(f, inv_mass_data)
        except Exception as e:
            print(f"Error writing output files in directory {save_dir}: {e}")

def change_filename(reorganised_path):
    """
    Change the filename of the combined data files to remove the "_combined" suffix and change ZZ to WW.
    """
    regions = [
        [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for cos_min in [0.0 + 0.1 * i for i in range(9)]
        for mass_min in [200.0 + 50.0 * j for j in range(20)]
    ]
    for region in regions:
        print("Changing filenames for region:", region)
        save_dir = os.path.join(reorganised_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
        # Check that the save directory exists
        if not os.path.exists(save_dir):
            print(f"Error: Save directory {save_dir} does not exist. Skipping.")
            continue
        # Rename files
        try:
            for file in ["e+_phi_data_combined.txt", "mu-_phi_data_combined.txt", "e+_theta_data_combined.txt", "mu-_theta_data_combined.txt", "psi_data_combined.txt", "ZZ_inv_mass_combined.txt"]:
                old_file_path = os.path.join(save_dir, file)
                new_file_path = os.path.join(save_dir, file.replace("_combined", ""))
                os.rename(old_file_path, new_file_path)
            old_file_path = os.path.join(save_dir, "ZZ_inv_mass.txt")
            new_file_path = os.path.join(save_dir, "WW_inv_mass.txt")
            os.rename(old_file_path, new_file_path)
        except Exception as e:
            print(f"Error renaming files in directory {save_dir}: {e}")

def add_regions(reorganised_path, events_dir):
    # Define order of regions
    regions = [
        [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for cos_min in [0.0 + 0.1 * i for i in range(9)]
        for mass_min in [200.0 + 50.0 * j for j in range(20)]
    ]

    new_regions = [regions[176], regions[98], regions[178]]

    run_start = 4

    for i, region in enumerate(new_regions):
        print("Processing region:", region)
        run_directory = os.path.join(events_dir, f"run_{(run_start + i):02d}")
        # Check that the run directory exists
        if not os.path.exists(run_directory):
            print(f"Error: Run directory {run_directory} does not exist. Skipping.")
            continue
        # Create the save directory for the current region
        save_dir = os.path.join(reorganised_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
        # Check that the save directory exists
        if not os.path.exists(save_dir):
            print(f"Error: Save directory {save_dir} does not exist.")
            sys.exit(1)

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
                    -11: [],
                    12: [],
                    13: [],
                    -14: []
                }
                # Read particles
                for particle in event.particles:
                    if particle.status == 1:
                        particle_data[particle.id].append(particle.e)
                        particle_data[particle.id].append(particle.px)
                        particle_data[particle.id].append(particle.py)
                        particle_data[particle.id].append(particle.pz)
                # Reconstruct diboson frame and first w
                diboson = np.array(particle_data[-11]) + np.array(particle_data[12]) + np.array(particle_data[13]) + np.array(particle_data[-14])
                w1 = np.array(particle_data[-11]) + np.array(particle_data[12])
                w1_boosted = np.zeros(4)
                # Boost W into diboson CM frame
                boostinvp(w1, diboson, w1_boosted)
                cos_psi = calc_scattering_angle(np.array(w1_boosted))
                cos_psi_data = np.append(cos_psi_data, cos_psi)
                inv_mass = calc_inv_mass(np.array(diboson))
                inv_mass_data = np.append(inv_mass_data, inv_mass)
                # Calculate decay angles
                ep_phi, mp_phi, ep_theta, mp_theta = phistar(np.array(particle_data[-11]), np.array(particle_data[12]), np.array(particle_data[13]), np.array(particle_data[-14]))
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
            with open(os.path.join(save_dir, "e+_phi_data.txt"), 'a') as f:
                np.savetxt(f, ep_phi_data)
            with open(os.path.join(save_dir, "mu-_phi_data.txt"), 'a') as f:
                np.savetxt(f, mp_phi_data)
            with open(os.path.join(save_dir, "e+_theta_data.txt"), 'a') as f:
                np.savetxt(f, ep_theta_data)
            with open(os.path.join(save_dir, "mu-_theta_data.txt"), 'a') as f:
                np.savetxt(f, mp_theta_data)
            with open(os.path.join(save_dir, "psi_data.txt"), 'a') as f:
                np.savetxt(f, cos_psi_data)
            with open(os.path.join(save_dir, "ZZ_inv_mass.txt"), 'a') as f:
                np.savetxt(f, inv_mass_data)
        except Exception as e:
            print(f"Error writing output files in directory {save_dir}: {e}")


def combine_files(reorganised_path):
    """
    Combine files with _combined suffix into existing files without the suffix.
    """
    regions = [
        [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
        for cos_min in [0.0 + 0.1 * i for i in range(9)]
        for mass_min in [200.0 + 50.0 * j for j in range(20)]
    ]

    new_regions = [regions[176], regions[98], regions[178]]

    for region in new_regions:
        print("Combining files for region:", region)
        save_dir = os.path.join(reorganised_path, f"cos_psi_{region[0][0]}_{region[0][1]}_inv_mass_{region[1][0]}_{region[1][1]}")
        # Check that the save directory exists
        if not os.path.exists(save_dir):
            print(f"Error: Save directory {save_dir} does not exist. Skipping.")
            continue
        # delete files
        try:
            for file in ["e+_phi_data_combined.txt", "mu-_phi_data_combined.txt", "e+_theta_data_combined.txt", "mu-_theta_data_combined.txt", "psi_data_combined.txt", "ZZ_inv_mass_combined.txt", "ZZ_inv_mass.txt"]:
                old_file_path = os.path.join(save_dir, file)
                if os.path.exists(old_file_path):
                    os.remove(old_file_path)
        except Exception as e:
            print(f"Error deleting files in directory {save_dir}: {e}")




def main():
    mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
    process_dir1 = os.path.join(mg5_install_dir, "Felipe_pp_WW_4l")
    process_dir2 = os.path.join(mg5_install_dir, "pp_WW_4l_final_process")
    events_dir = os.path.join(process_dir1, "Events")
    save_dir = os.path.join(process_dir2, "Plots and data/organised_data")

    combine_files(save_dir)
    

if __name__ == "__main__":
    main()