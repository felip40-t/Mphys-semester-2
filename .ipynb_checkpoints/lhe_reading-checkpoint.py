import os 
import re
import subprocess
import pylhe
import numpy as np



def run_madgraph(mg5_install_dir, process_dir, energy):

    # directory to run card
    run_card_path = os.path.join(process_dir, 'Cards', 'run_card.dat')
    
    with open(run_card_path, 'r') as file:
        run_card = file.readlines()
    
    # set energy
    for i, line in enumerate(run_card):
        if 'ebeam1' in line:
            run_card[i] = f"  {energy/2}    = ebeam1 ! beam 1 total energy in GeV\n"
        if 'ebeam2' in line:
            run_card[i] = f"  {energy/2}    = ebeam2 ! beam 2 total energy in GeV\n"
        if 'nevents' in line:
            run_card[i] = f"  {10000}    = nevents ! Number of unweighted events requested\n"



    with open(run_card_path, 'w') as file:
        file.writelines(run_card)

    # run process
    subprocess.run([os.path.join(mg5_install_dir, 'bin', 'mg5_aMC')], input='launch pp_2_4_process\n', text=True, cwd=process_dir, check=True, capture_output=True)
    

def find_latest_run_dir(base_dir):
    
    all_items = os.listdir(base_dir)

    # Filter out only directories that start with 'run_' and are actual directories
    run_dirs = []
    for item in all_items:
        item_path = os.path.join(base_dir, item) 
        if item.startswith('run_') and os.path.isdir(item_path):  
            run_dirs.append(item)

    # Extract the run numbers, sort, and get the latest
    run_numbers = [int(d.split('_')[1]) for d in run_dirs] 
    latest_run_number = max(run_numbers)  
    latest_run_dir = f"run_{latest_run_number:02d}"

    return os.path.join(base_dir, latest_run_dir)




def read_lhe_write_data(lhe_file_path, particle_directories):
    """
    Reads compressed .lhe file and extracts data to write to corresponding particle folder.

    lhe_file_path = path to compressed lhe file which is found by helper function
    particle_directories = dictionary mapping PDG codes to folders
    """
    
    for event in pylhe.read_lhe_with_attributes(lhe_file_path):
        for particle in event.particles:
            # Only consider final state particles (status == 1)
            if particle.status == 1:
                # Prepare the data line to write: "P_x, P_y, P_z, particle energy"
                data_line = f"{particle.px}, {particle.py}, {particle.pz}, {particle.e}\n"
                
                # Get the directory for this particle's PDG ID
                particle_id = particle.id
                if particle_id in particle_directories:
                    file_path = os.path.join(particle_directories[particle_id], "4_vec_data.txt")
                    # Append the data to the corresponding file
                    with open(file_path, "a") as f:
                        f.write(data_line)



def process_last_run(base_dir, particle_directories):

    # get directory of relevant run using helper function
    latest_run_dir = find_latest_run_dir(base_dir)
    print(f"Processing latest run directory: {latest_run_dir}")
    
    # find .lhe file
    for file in os.listdir(latest_run_dir):
        if file.endswith(".lhe.gz"):
            lhe_file_path = os.path.join(latest_run_dir, file)
            print(f"Found LHE file: {lhe_file_path}")
            read_lhe_write_data(lhe_file_path, particle_directories)    



def main():
    mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
    process_dir = os.path.join(mg5_install_dir, "pp_2_4_process")
    base_dir = os.path.join(process_dir, "Events")
    
    particle_directories = {
    11: os.path.join(process_dir, "Plots and data/e-"),  # e-
    -11: os.path.join(process_dir, "Plots and data/e+"), # e+
    12: os.path.join(process_dir, "Plots and data/ve"),  # ve
    -12: os.path.join(process_dir, "Plots and data/ve~"), # ve~
    }

    ENERGY = 13000 # GeV
    
    # Run MadGraph
    run_madgraph(mg5_install_dir, process_dir, ENERGY)

    # Read and write data for this current run
    process_last_run(base_dir, particle_directories)

if __name__ == "__main__":
    main()