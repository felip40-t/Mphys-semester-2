import os
import numpy as np
import wadllib
from lhe_reading_WW import find_latest_run_dir
from histo_plotter import read_data

# Update the MadGraph5 directory and process directory to WZ_process
mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
process_dir = os.path.join(mg5_install_dir, "pp_WZ_SM")
base_dir = os.path.join(process_dir, "Events")

# Particle directories corresponding to the WZ process
particle_directories = {
    'mu-': os.path.join(process_dir, "Plots and data/mu-"),  # mu-
    'mu+': os.path.join(process_dir, "Plots and data/mu+"),  # mu+
    'e+': os.path.join(process_dir, "Plots and data/e+"),    # e+
    've': os.path.join(process_dir, "Plots and data/ve"),    # ve
    'w+': os.path.join(process_dir, "Plots and data/w+"),    # w+
    'z': os.path.join(process_dir, "Plots and data/z"),      # z 
}

# Function to compute the Lorentz boost matrix
def lorentz_boost(beta_vector):
    total_beta = np.linalg.norm(beta_vector)
    gamma = 1 / np.sqrt(1 - total_beta**2)
    
    boost = np.identity(4)
    beta_vector = np.array(beta_vector)

    # Update the boost matrix based on gamma and the beta vector
    boost[1:, 1:] += (gamma - 1) * np.outer(beta_vector, beta_vector) / total_beta**2
    boost[0, 1:] = boost[1:, 0] = -gamma * beta_vector
    boost[0, 0] = gamma

    return boost

# Calculate the velocity (beta) from the four-momentum
def find_beta(four_momentum):
    return four_momentum[1:4] / four_momentum[0]

# Apply Lorentz boost to a set of vectors
def execute_boost(vec_array, boost_array):
    vec_array_boosted = [lorentz_boost(find_beta(boost_vec)) @ vec for vec, boost_vec in zip(vec_array, boost_array)]
    return np.array(vec_array_boosted)

def azimuthal_angle(lep_vec, parent_axis):
    parent_axis = parent_axis / np.linalg.norm(parent_axis)

    # Project lep_vec onto the plane perpendicular to parent_axis
    v_parallel = np.dot(lep_vec, parent_axis) * parent_axis  
    v_perpendicular = lep_vec - v_parallel                   

    # Reference direction: projection of the global x-axis onto the plane perpendicular to parent_axis
    e_z = np.array([0, 0, 1])                               
    e_z_parallel = np.dot(e_z, parent_axis) * parent_axis    
    e_z_perpendicular = e_z - e_z_parallel                 

    # Normalise
    v_perpendicular = v_perpendicular / np.linalg.norm(v_perpendicular)
    e_z_perpendicular = e_z_perpendicular / np.linalg.norm(e_z_perpendicular)

    sin_phi = np.linalg.norm(np.cross(v_perpendicular, e_z_perpendicular))
    cos_phi = np.dot(v_perpendicular, e_z_perpendicular)

    # Use the dot product with a reference direction to determine the sign of phi
    e_ref = np.cross(parent_axis, e_z_perpendicular)                                                 
    e_ref = e_ref / np.linalg.norm(e_ref)  # normalise

    # Check the sign of phi
    if np.dot(v_perpendicular, e_ref) < 0:
        sin_phi = - sin_phi  

    # Compute the azimuthal angle 
    phi = np.arctan2(sin_phi, cos_phi)

    return phi

def calc_polar_angle(lep_array, parent_array, name, run_num):
    lep_vec = lep_array[:, 1:]  # Take spatial components of leptons
    parent_vec = parent_array[:, 1:]  # Take spatial components of parent (W or Z)

    # Compute norms for normalization
    lep_norm = np.linalg.norm(lep_vec, axis=1)
    parent_norm = np.linalg.norm(parent_vec, axis=1)

    # Calculate cos(theta) for each event
    cos_theta = np.array([(lep_vec[i] @ parent_vec[i]) / (lep_norm[i] * parent_norm[i]) for i in range(len(lep_norm))])

    # Normalize parent_vec to use as the parent axis for azimuthal angle calculation
    parent_axis = parent_vec / parent_norm[:, np.newaxis]  # Ensure proper normalization

    # Calculate azimuthal angles for each event
    phi = np.array([azimuthal_angle(lep_vec[i], parent_axis[i]) for i in range(len(lep_vec))])

    # Save cos(theta) data to a file
    file_path_theta = os.path.join(particle_directories[name], f"theta_data_{run_num}.txt")
    np.savetxt(file_path_theta, cos_theta)

    # Save phi data to a file
    file_path_phi = os.path.join(particle_directories[name], f"phi_data_{run_num}.txt")
    np.savetxt(file_path_phi, phi)

# Main function for processing data and calculating the Lorentz boost and angles
def main():
    # Find the latest run directory and run number
    #_, run_number = find_latest_run_dir(base_dir)
    run_number = 3
    # Read the four-momenta data for all particles in the process
    particle_arrays = {particle_name: read_data(os.path.join(directory, f"data_{run_number}.txt"))
                      for particle_name, directory in particle_directories.items()}

    # Reconstruct the total diboson system (w+ + z)
    diboson_array = particle_arrays['w+'] + particle_arrays['z']

    # Boost calculations for e+ 
    w_boosted = execute_boost(particle_arrays['w+'], diboson_array)
    e_plus_intermed = execute_boost(particle_arrays['e+'], diboson_array)
    e_plus_boosted = execute_boost(e_plus_intermed, w_boosted)
    calc_polar_angle(e_plus_boosted, w_boosted, 'e+', run_number)

    # Boost calculations for mu+
    z_boosted = execute_boost(particle_arrays['z'], diboson_array)
    mu_plus_intermed = execute_boost(particle_arrays['mu+'], diboson_array)
    mu_plus_boosted = execute_boost(mu_plus_intermed, z_boosted)
    calc_polar_angle(mu_plus_boosted, z_boosted, 'mu+', run_number)

    # Boost calculations for mu-
    mu_minus_intermed = execute_boost(particle_arrays['mu-'], diboson_array)
    mu_minus_boosted = execute_boost(mu_minus_intermed, z_boosted)
    calc_polar_angle(mu_minus_boosted, z_boosted, 'mu-', run_number)




if __name__ == "__main__":
    main()
