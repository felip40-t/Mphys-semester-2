import os
import subprocess
import pylhe
import numpy as np
from glob import glob


def run_madgraph(mg5_install_dir, process_dir, energy, nevents, bwcutoff):
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
        elif 'bwcutoff' in line:
            run_card[i] = f"  {bwcutoff}   = bwcutoff      ! (M+/-bwcutoff*Gamma)\n"


    with open(run_card_path, 'w') as file:
        file.writelines(run_card)

    # Run the MadGraph process using subprocess, streamlined
    subprocess.run(
        [os.path.join(mg5_install_dir, 'bin', 'mg5_aMC')],
        input='launch WZ_madspin\n', text=True,
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

def calc_inv_mass(E, px, py, pz):
    return np.sqrt(E**2 - (px**2 + py**2 + pz**2))

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
        valid = False  # Flag to track if the event passes the filter

        # List to hold the two muons
        muons = []

        # Collect particles that are final state and are muons or anti-muons
        for particle in event.particles:
            if particle.status == 1:  # Final state particle
                particle_id = particle.id
                if particle_id == 13 or particle_id == -13:
                    muons.append(particle)

        if len(muons) == 2:
            muon1, muon2 = muons[0], muons[1]

            # Calculate invariant mass of the muon pair
            inv_mass = calc_inv_mass(muon1.e + muon2.e, muon1.px + muon2.px, 
                                     muon1.py + muon2.py, muon1.pz + muon2.pz)

            # Check if invariant mass is within the range [81, 101] GeV
            if 81 <= inv_mass <= 101:
                valid = True  # Mark event as valid

        # If the event is valid, write the data
        if valid:
            for particle in event.particles:
                if particle.status == 1:  # Final state particle
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
    _, number = find_latest_run_dir(base_dir)
    latest_run_dir = os.path.join(base_dir, f"run_{int(number):02d}_decayed_1")
    print(f"Processing latest run directory: {latest_run_dir}")
    
    # Locate LHE file in the latest run directory
    lhe_file = glob(os.path.join(latest_run_dir, "*.lhe.gz"))
    if not lhe_file:
        raise FileNotFoundError("No LHE file found in the latest run directory.")
    
    lhe_file_path = lhe_file[0]
    print(f"Found LHE file: {lhe_file_path}")
    read_lhe_write_data(lhe_file_path, particle_directories, number)


def main():
    mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
    process_dir = os.path.join(mg5_install_dir, "WZ_madspin")
    base_dir = os.path.join(process_dir, "Events")
    
    particle_directories = {
        13: os.path.join(process_dir, "Plots and data/mu-"),  # mu-
        -13: os.path.join(process_dir, "Plots and data/mu+"), # mu+
        -11: os.path.join(process_dir, "Plots and data/e+"),   # e+
        12: os.path.join(process_dir, "Plots and data/ve"),  # ve
    }

    ENERGY = 13_000  # GeV
    NEVENTS = 500_000
    BWCUTOFF = 5.0

    # Run MadGraph and process data
    run_madgraph(mg5_install_dir, process_dir, ENERGY, NEVENTS, BWCUTOFF)
    process_last_run(base_dir, particle_directories)


if __name__ == "__main__":
    main()
