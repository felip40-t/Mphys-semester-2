import os
import numpy as np
from lhe_reading_WW import find_latest_run_dir
from histo_plotter import read_data

mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
process_dir = os.path.join(mg5_install_dir, "pp_WW_SM")
base_dir = os.path.join(process_dir, "Events")

particle_directories = {
    'mu-': os.path.join(process_dir, "Plots and data/mu-"),
    'e+': os.path.join(process_dir, "Plots and data/e+"),
    've': os.path.join(process_dir, "Plots and data/ve"),
    'vm~': os.path.join(process_dir, "Plots and data/vm~"),
    'w+': os.path.join(process_dir, "Plots and data/w+"),
    'w-': os.path.join(process_dir, "Plots and data/w-"),
}

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

def find_beta(four_momentum):
    # Calculate beta from the four-momentum vector
    return four_momentum[1:4] / four_momentum[0]

def execute_boost(vec_array, boost_array):
    # Calculate boosted vectors using vectorized operations
    vec_array_boosted = [lorentz_boost(find_beta(boost_vec)) @ vec for vec, boost_vec in zip(vec_array, boost_array)]
    return np.array(vec_array_boosted)

# Calculate invariant mass
def calc_inv_mass(four_vec):
    return np.sqrt(four_vec[0]**2 - np.sum(four_vec[1:]**2))

def azimuthal_angle(lep_vec, parent_axis):
    # Normalise
    parent_axis = parent_axis / np.linalg.norm(parent_axis)

    # Project lep_vec onto the plane perpendicular to parent_axis
    v_parallel = np.dot(lep_vec, parent_axis) * parent_axis  
    v_perpendicular = lep_vec - v_parallel                   

    # Zero axis: projection of the beam axis onto the plane perpendicular to the parent axis
    e_z = np.array([0, 0, 1])                               
    e_z_parallel = np.dot(e_z, parent_axis) * parent_axis    
    e_zero = e_z - e_z_parallel                 

    # Normalise
    v_perpendicular = v_perpendicular / np.linalg.norm(v_perpendicular)
    e_zero = e_zero / np.linalg.norm(e_zero)

    # Use the cross product between parent axis and zero axis to determine the reference axis.
    e_ref = np.cross(parent_axis, e_zero)                                                 
    e_ref = e_ref / np.linalg.norm(e_ref)  

    # Calculate sin and cosine of angle 
    sin_phi = np.dot(v_perpendicular, e_ref)
    cos_phi = np.dot(v_perpendicular, e_zero)

    # Use arctan2 to get correct sign 
    phi = np.arctan2(sin_phi, cos_phi)
    return phi

def calc_scattering_angle(parent_axis):
    # Normalise
    parent_axis = parent_axis / np.linalg.norm(parent_axis)
    beam_axis = np.array([0,0,1])
    cos_psi = parent_axis @ beam_axis
    return cos_psi


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

    # Calculate scattering angle for each event
    cos_psi = np.array([calc_scattering_angle(parent_axis[i]) for i in range(len(lep_vec))])

    # Save cos(theta) data to a file
    file_path_theta = os.path.join(particle_directories[name], f"theta_data_{run_num}.txt")
    np.savetxt(file_path_theta, cos_theta)

    # Save phi data to a file
    file_path_phi = os.path.join(particle_directories[name], f"phi_data_{run_num}.txt")
    np.savetxt(file_path_phi, phi)

    # Save cos(psi) data to file
    file_path_psi = os.path.join(process_dir, f"Plots and data/psi_data_{run_num}.txt")
    np.savetxt(file_path_psi, cos_psi)

def main():
    _, run_number = find_latest_run_dir(base_dir)

    particle_arrays = {particle_name: read_data(os.path.join(directory, f"data_{run_number}.txt"))
                      for particle_name, directory in particle_directories.items()}

    diboson_array = particle_arrays['w+'] + particle_arrays['w-']

    # Boost calculations
    w_boosted = execute_boost(particle_arrays['w+'], diboson_array)
    e_plus_intermed = execute_boost(particle_arrays['e+'], diboson_array)
    e_plus_boosted = execute_boost(e_plus_intermed, w_boosted)
    calc_polar_angle(e_plus_boosted, w_boosted, 'e+', run_number)

    wm_boosted = execute_boost(particle_arrays['w-'], diboson_array)
    mu_intermed = execute_boost(particle_arrays['mu-'], diboson_array)
    mu_boosted = execute_boost(mu_intermed, wm_boosted)
    calc_polar_angle(mu_boosted, wm_boosted, 'mu-', run_number)

    WW_inv_mass = np.apply_along_axis(calc_inv_mass, 1, diboson_array)
    file_path_inv_mass = os.path.join(process_dir, f"Plots and data/WW_inv_mass_{run_number}.txt")
    np.savetxt(file_path_inv_mass, WW_inv_mass)


if __name__ == "__main__":
    main()
