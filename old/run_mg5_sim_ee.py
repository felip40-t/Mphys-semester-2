import os
import re
import subprocess

# Function to run MadGraph5 and capture its output
def run_madgraph_with_energy(mg5_install_dir, process_dir, energy):
    # Modify the run_card.dat with the new energy value
    run_card_path = os.path.join(process_dir, 'Cards', 'run_card.dat')
    
    # Update beam energies in the run_card.dat
    with open(run_card_path, 'r') as file:
        run_card = file.readlines()
    
    for i, line in enumerate(run_card):
        if 'ebeam1' in line:
            run_card[i] = f"  {energy/2}    = ebeam1 ! beam 1 total energy in GeV\n"
        if 'ebeam2' in line:
            run_card[i] = f"  {energy/2}    = ebeam2 ! beam 2 total energy in GeV\n"
    
    with open(run_card_path, 'w') as file:
        file.writelines(run_card)
    
    # Run MadGraph5 and capture its output
    process = subprocess.run([os.path.join(mg5_install_dir, 'bin', 'mg5_aMC')], input='launch pp_process\n', text=True, cwd=process_dir, check=True, capture_output=True)
    
    # Capture and return the output
    return process.stdout

# Function to extract the cross-section and uncertainty from MadGraph5 output
def extract_cross_section(output):
    # Search for the line containing the cross-section using a regex pattern
    match = re.search(r'Cross-section\s*:\s*([0-9.]+)\s*\+-\s*([0-9.]+)\s*pb', output)
    
    if match:
        cross_section = float(match.group(1))
        uncertainty = float(match.group(2))
        return cross_section, uncertainty
    else:
        return None, None

# Function to write data to a file in the format: energy, cross-section, uncertainty
def write_data_to_file(data, filename="/home/felipetcach/project/MG5_aMC_v3_5_5/pp_process/Plots and data/cross_sect_data_run_1.txt"):
    with open(filename, "w") as file:
        for energy, cross_section, uncertainty in data:
            file.write(f"{energy}, {cross_section}, {uncertainty}\n")
    print(f"Data saved to {filename}")

# Main function
def main():
    # Path to your MadGraph5 process directory
    mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_5"
    process_dir = os.path.join(mg5_install_dir, "pp_process")
    
    # Array to store cross-sections and uncertainties
    cross_sections = []
    
    # Loop over different energy values
    energies = [ i for i in range(162,400,20)]  # Example energies in GeV
    for energy in energies:
        print(f"Running simulation at {energy} GeV...")
        
        # Run MadGraph with the current energy
        output = run_madgraph_with_energy(mg5_install_dir, process_dir, energy)
        
        # Extract cross-section and uncertainty from the output
        cross_section, uncertainty = extract_cross_section(output)
        
        if cross_section is not None:
            print(f"Cross-section at {energy} GeV: {cross_section} pb ± {uncertainty} pb")
            cross_sections.append((energy, cross_section, uncertainty))
        else:
            print(f"Failed to extract cross-section at {energy} GeV")
    
    # Alternatively, print the array for quick inspection
    print("Collected cross-sections:")
    print(cross_sections)

    write_data_to_file(cross_sections)

if __name__ == "__main__":
    main()
