import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from lhe_reading_WW import find_latest_run_dir


process_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6/4_lepton_process"
base_dir = os.path.join(process_dir, "Events")


particle_directories = {
    11: os.path.join(process_dir, "Plots and data/e-"),  
    -11: os.path.join(process_dir, "Plots and data/e+"),  
    }

COM_ENERGY = 13000 # GeV


def get_particle_name(particle_id):
    
    pdg_names = {
        11: 'e-',
        -11: 'e+',
    }
    return pdg_names[particle_id]

def read_data(file_path):
    """
    Reads the data from the data.txt file and returns it as a numpy array.
    Each line in data.txt is expected to have the format:
    E, px, py, pz
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split line into values and convert to floats
            values = [float(val) for val in line.strip().split(',')]
            data.append(values)
    return np.array(data) 


def calculate_transverse_p(data):
    p_T = np.sqrt(data[:, 1]**2 + data[:, 2]**2)
    return p_T

def calculate_pseudorapidity(data):
    abs_p = np.sqrt(data[:, 1]**2 + data[:, 2]**2 + data[:, 3]**2)
    eta = np.arctanh(data[:,3]/abs_p)
    return eta



def plot_histo_p_T(data, particle_name, output_dir, xsec, num):
    """
    Plots histogram for the transverse momentum
    """
    # Calculating transverse momentum
    transverse_p = calculate_transverse_p(data)
    # Print mean
    print(f"The mean of the whole data set is {np.mean(calculate_transverse_p(data))}.")
    
    # Generate bins and weights
    bins=np.linspace(0,500,41)
    normalisation = 0.5
    weights = np.ones_like(transverse_p) * normalisation

    plt.figure(figsize=(10, 6), dpi=80)
    # Plot histogram original
    plt.hist(transverse_p, bins=bins, weights=weights, \
        label=f"Me ({normalisation})",histtype='step', edgecolor='blue')

    # Creating data sequence: middle of each bin
    xData = np.array([6.25,18.75,31.25,43.75,56.25,68.75,81.25,93.75,106.25,118.75,131.25,143.75,156.25,168.75,181.25,193.75,206.25,218.75,231.25,243.75,256.25,268.75,281.25,293.75,306.25,318.75,331.25,343.75,356.25,368.75,381.25,393.75,406.25,418.75,431.25,443.75,456.25,468.75,481.25,493.75])
    # Creating weights for histo:
    if particle_name == 'e-':
        PT_0_weights = np.array([76.41739024150007,799.107602525401,1185.5610037467004,980.3260030981012,552.3886017457007,299.5562009466804,169.4283005354403,96.06758030360012,63.75394020148008,41.48373013110006,25.32691008004004,15.72015004968002,13.97347004416003,11.790110037260007,7.423404023460009,5.676721017940007,5.676721017940007,3.0566960096600044,3.9300370124200037,1.31001200414,1.31001200414,1.31001200414,0.873341602760001,0.873341602760001,0.873341602760001,0.0,0.4366708013800005,0.4366708013800005,0.4366708013800005,0.4366708013800005,0.4366708013800005,0.0,0.0,0.4366708013800005,0.0,0.0,0.0,0.0,0.4366708013800005,0.0])
    elif particle_name == 'e+':
        PT_0_weights = np.array([93.44755444264025,889.0618422673654,1160.2340551593088,943.2089448416018,535.7951254725232,285.5827135770408,166.37160790956176,102.1810048578419,51.52716244968044,36.680351743840255,24.01689114179989,18.776840892679854,12.226780581279927,13.53680064356029,6.986733332160033,5.240050249120037,4.3667082076000145,1.7466830830399962,3.0566961453200294,3.0566961453200294,1.3100120622799853,0.8733416415200029,0.8733416415200029,1.3100120622799853,0.43667082076000147,0.0,0.43667082076000147,0.8733416415200029,0.43667082076000147,0.43667082076000147,0.0,0.0,0.0,0.43667082076000147,0.43667082076000147,0.43667082076000147,0.0,0.0,0.0,0.0])
    # Plot histogram MA5
    plt.hist(x=xData, bins=bins, weights=PT_0_weights, \
        label=f"MadAnalysis ({xsec[0]:.3f})", histtype='step', edgecolor='red')

    # Set y axis
    plt.yscale('log', )
    plt.ylim(0.01, 2000)
    # Set labels
    plt.title(f'Transverse momentum for {particle_name}', fontsize=20)
    plt.xlabel(r'$p_{T}$ (Gev/c)', fontsize=16)
    plt.ylabel(r'Events ($L_{int} = 10 fb^{-1}$)', fontsize=16)
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    # Save file
    figure_path = os.path.join(output_dir, f"{particle_name}_p_T_histo3_run_8.pdf")
    figure_path_png = os.path.join(output_dir, f"{particle_name}_p_T_histo3_run_8.png")
    plt.savefig(figure_path)
    plt.savefig(figure_path_png)



def plot_histo_eta(data, particle_name, output_dir, xsec, num):
    """
    Plots histogram for the pseudorapidity
    """
    # Calculating pseudorapidity
    eta = calculate_pseudorapidity(data)
    
    # Print mean
    print(f"The mean of the whole data set is {np.mean(calculate_pseudorapidity(data))}.")
    
    # Generate bins and weights
    bins=np.linspace(-10,10,41)
    normalisation = 0.5 
    weights = np.ones_like(eta) * normalisation

    plt.figure(figsize=(10, 6))
    # Plot histogram
    plt.hist(eta, bins=bins, weights=weights, \
        label=f"Me ({normalisation})",histtype='step', edgecolor='blue')

    # Creating data sequence: middle of each bin
    xData = np.array([-9.75,-9.25,-8.75,-8.25,-7.75,-7.25,-6.75,-6.25,-5.75,-5.25,-4.75,-4.25,-3.75,-3.25,-2.75,-2.25,-1.75,-1.25,-0.75,-0.25,0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75])
    # Creating weights for histo:
    if particle_name == 'e-':
        ETA_0_weights = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,318.33300728999995,420.5140096300007,463.3077106099999,482.9579110600002,509.5948116699998,490.3813112300001,494.74801132999994,442.3475101299998,396.49710908000054,348.0266079699996,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    elif particle_name == 'e+':
        ETA_0_weights = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,329.24979245999975,410.47059059999907,448.4608897300005,504.35478844999966,519.2015881099998,459.8143894699991,479.4644890200011,457.19428953000084,414.8372904999993,343.6598921300006,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    # Plot histogram MA5
    plt.hist(x=xData, bins=bins, weights=ETA_0_weights, \
        label=f"MadAnalysis ({xsec[0]:.3f})", histtype='step', edgecolor='red')
    
    # Set y axis
    plt.yscale('log', )
    plt.ylim(2, 700)
    # Set labels
    plt.title(f'Pseudorapidity for {particle_name}', fontsize=20)
    plt.xlabel(r'$\eta$', fontsize=16)
    plt.ylabel(r'Events ($L_{int} = 10 fb^{-1}$)', fontsize=16)
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    # Save files
    figure_path = os.path.join(output_dir, f"{particle_name}_eta_histo2_run_{num}.pdf")
    figure_path_png = os.path.join(output_dir, f"{particle_name}_eta_histo2_run_{num}.png")
    plt.savefig(figure_path)
    plt.savefig(figure_path_png)


def process_and_plot(particle_directories):
    """
    Reads data from all directories and plots the results for each particle.
    """
    _, number = find_latest_run_dir(base_dir)

    for particle_id, directory in particle_directories.items():
        data_file = os.path.join(directory, f"data_{number}.txt")
        xsec_file = os.path.join(directory, f"cross_sec_{number}.txt")
        
        if os.path.exists(data_file):
            print(f"Reading data from {data_file}...")
            
            data = read_data(data_file)
            cross_section = read_data(xsec_file)[0]
            
            particle_name = get_particle_name(particle_id)
            
            plot_histo_p_T(data, particle_name, directory, cross_section, number)
            plot_histo_eta(data, particle_name, directory, cross_section, number)

        else:
            print(f"File not found: {data_file}")



if __name__ == "__main__":
    process_and_plot(particle_directories)
