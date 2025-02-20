
import os
import numpy as np
from lhe_reading_WW import find_latest_run_dir
from histo_plotter import read_data

# Update the MadGraph5 directory and process directory to WZ_process
mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
process_dir = os.path.join(mg5_install_dir, "pp_ZZ_SM")
base_dir = os.path.join(process_dir, "Events")

# Particle directories corresponding to the WZ process
particle_directories = {
    'mu+': os.path.join(process_dir, "Plots and data/mu+"),  # mu+
    'mu-': os.path.join(process_dir, "Plots and data/mu-"),  # mu-
    'e+': os.path.join(process_dir, "Plots and data/e+"),    # e+
    'e-': os.path.join(process_dir, "Plots and data/e-"),    # e-
    'z1': os.path.join(process_dir, "Plots and data/z1"),    # z boson
    'z2': os.path.join(process_dir, "Plots and data/z2"),      # z boson 
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

# Calculate invariant mass
def calc_inv_mass(four_vec):
    return np.sqrt(four_vec[0]**2 - np.sum(four_vec[1:]**2))

def azimuthal_angle(lep_vec, parent_axis, e_z):
    # Normalise
    parent_axis = parent_axis / np.linalg.norm(parent_axis) # Boson flight path in the diboson CM frame
    e_z = e_z / np.linalg.norm(e_z) # Boson flight path in the lab frame, or 'new' z axis
    # Calculate the normal to the production plane
    n_p = np.cross(parent_axis, e_z)
    n_p = n_p / np.linalg.norm(n_p)
    # Calculate the normal to the decay plane
    n_d = np.cross(lep_vec, parent_axis)
    n_d = n_d / np.linalg.norm(n_d)
    # Calculate the azimuthal angle
    cos_phi = n_p @ n_d
    phi = np.arccos(cos_phi)
    # Determine sign of phi
    sign = np.sign(e_z @ np.cross(n_p, n_d))
    return sign * phi

def calc_scattering_angle(parent_axis):
    beam_axis = np.array([0,0,1])
    cos_psi = parent_axis @ beam_axis
    return cos_psi

def calc_polar_angle(lep_array, parent_array, name, run_num, e_z):
    lep_vec = lep_array[:, 1:]  # Take spatial components of leptons
    parent_vec = parent_array[:, 1:]  # Take spatial components of parent (W or Z)
    e_z_vec = e_z[:, 1:]  # Take spatial components

    # Compute norms for normalization
    lep_norm = np.linalg.norm(lep_vec, axis=1)
    parent_norm = np.linalg.norm(parent_vec, axis=1)

    # Calculate cos(theta) for each event
    cos_theta = np.array([(lep_vec[i] @ parent_vec[i]) / (lep_norm[i] * parent_norm[i]) for i in range(len(lep_norm))])

    # Save cos(theta) data to a file
    file_path_theta = os.path.join(particle_directories[name], f"theta_data_{run_num}.txt")
    np.savetxt(file_path_theta, cos_theta)

    # Normalize parent_vec to use as the parent axis for azimuthal angle calculation
    parent_axis = parent_vec / parent_norm[:, np.newaxis]

    # Calculate scattering angle for each event
    cos_psi = np.array([calc_scattering_angle(parent_axis[i]) for i in range(len(lep_vec))])

    # Save cos(psi) data to file
    file_path_psi = os.path.join(process_dir, f"Plots and data/psi_data_{run_num}.txt")
    np.savetxt(file_path_psi, cos_psi)

    # Calculate azimuthal angles for each event
    phi = np.array([azimuthal_angle(lep_vec[i], parent_vec[i], e_z_vec[i]) for i in range(len(lep_vec))])
    # Save phi data to a file
    file_path_phi = os.path.join(particle_directories[name], f"phi_data_{run_num}.txt")
    np.savetxt(file_path_phi, phi)

# Main function for processing data and calculating the Lorentz boost and angles
def main():
    # Find the latest run directory and run number
    _, run_number = find_latest_run_dir(base_dir)

    # Read the four-momenta data for all particles in the process
    particle_arrays = {particle_name: read_data(os.path.join(directory, f"data_{run_number}.txt"))
                      for particle_name, directory in particle_directories.items()}

    # Reconstruct the total diboson system (z1 + z2)
    diboson_array = particle_arrays['z1'] + particle_arrays['z2']
    beam_axis = np.array([0, 0, 1])
    beam_boosted = execute_boost(beam_axis[np.newaxis, :], particle_arrays['z1'])
    # Boost calculations for e+ 
    z1_boosted = execute_boost(particle_arrays['z1'], diboson_array)
    e_plus_intermed = execute_boost(particle_arrays['e+'], diboson_array)
    e_plus_boosted = execute_boost(e_plus_intermed, z1_boosted)
    calc_polar_angle(e_plus_boosted, z1_boosted, 'e+', run_number, beam_boosted)

    # Boost calculations for mu+
    z2_boosted = execute_boost(particle_arrays['z2'], diboson_array)
    mu_plus_intermed = execute_boost(particle_arrays['mu+'], diboson_array)
    mu_plus_boosted = execute_boost(mu_plus_intermed, z2_boosted)
    calc_polar_angle(mu_plus_boosted, z2_boosted, 'mu+', run_number, beam_boosted)

    ZZ_inv_mass = np.apply_along_axis(calc_inv_mass, 1, diboson_array)
    file_path_inv_mass = os.path.join(process_dir, f"Plots and data/ZZ_inv_mass_{run_number}.txt")
    np.savetxt(file_path_inv_mass, ZZ_inv_mass)

if __name__ == "__main__":
    main()



