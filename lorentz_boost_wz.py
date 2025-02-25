import os
import numpy as np
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


# Calculate invariant mass
def calc_inv_mass(four_vec):
    return np.sqrt(four_vec[0]**2 - np.sum(four_vec[1:]**2))


# Old method to calculate azimuthal angle
def azimuthal_angle(lep_vec, parent_axis):
    # Normalise
    parent_axis = parent_axis / np.linalg.norm(parent_axis) # Boson flight path in the diboson CM frame
    e_z = np.array([0,0,1]) # Z-axis or beam axis
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

def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)
    rotation = np.array([[np.cos(theta) + axis[0]**2 * (1 - np.cos(theta)),
                          axis[0] * axis[1] * (1 - np.cos(theta)) - axis[2] * np.sin(theta),
                          axis[0] * axis[2] * (1 - np.cos(theta)) + axis[1] * np.sin(theta)],
                         [axis[1] * axis[0] * (1 - np.cos(theta)) + axis[2] * np.sin(theta),
                          np.cos(theta) + axis[1]**2 * (1 - np.cos(theta)),
                          axis[1] * axis[2] * (1 - np.cos(theta)) - axis[0] * np.sin(theta)],
                         [axis[2] * axis[0] * (1 - np.cos(theta)) - axis[1] * np.sin(theta),
                          axis[2] * axis[1] * (1 - np.cos(theta)) + axis[0] * np.sin(theta),
                          np.cos(theta) + axis[2]**2 * (1 - np.cos(theta))]])
    return rotation

# Code from fortran code to rotate a vector
def rotinvp(vec, ref):
    """
    Rotate the 3-vector 'vec' so that the 3-vector 'ref' becomes aligned with the z-axis.
    This uses the Rodrigues rotation formula.
    """
    norm_ref = np.linalg.norm(ref)
    if norm_ref == 0:
        # No meaningful rotation if the reference vector is zero.
        return vec.copy()
    # Unit vector along the ref direction.
    u = ref / norm_ref
    # Target direction: along the positive z-axis.
    target = np.array([0.0, 0.0, 1.0])
    dot = np.clip(np.dot(u, target), -1.0, 1.0)
    angle = np.arccos(dot)
    # Determine the rotation axis.
    axis = np.cross(u, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        # If u is parallel (or anti-parallel) to target.
        if dot < 0:
            # For anti-parallel, choose an arbitrary orthogonal axis.
            axis = np.array([1.0, 0.0, 0.0])
            angle = np.pi
        else:
            return vec.copy()
    else:
        axis = axis / axis_norm
    # Rodrigues rotation formula:
    vec_rot = (vec * np.cos(angle) +
               np.cross(axis, vec) * np.sin(angle) +
               axis * np.dot(axis, vec) * (1 - np.cos(angle)))
    return vec_rot

def azimuthal_angle2(lep_vec, parent_axis):
    # Normalise
    parent_axis = parent_axis / np.linalg.norm(parent_axis) # Boson flight path in the diboson CM frame
    e_z = np.array([0,0,1]) # Z-axis
    # Find rotation axis
    rot_axis = np.cross(e_z, parent_axis)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    # Find rotation angle
    cos_theta = e_z @ parent_axis
    theta = np.arccos(cos_theta)
    # Rotate lepton vector
    lep_vec_rot = rotation_matrix(rot_axis, theta) @ lep_vec
    # Calculate azimuthal angle
    phi = np.arctan2(lep_vec_rot[1], lep_vec_rot[0])
    return phi

def calc_scattering_angle(parent_axis):
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

    # Save cos(theta) data to a file
    file_path_theta = os.path.join(particle_directories[name], f"theta_data_{run_num}.txt")
    np.savetxt(file_path_theta, cos_theta)

    # Calculate scattering angle for each event
    parent_axis = parent_vec / parent_norm[:, np.newaxis]
    cos_psi = np.array([calc_scattering_angle(parent_axis[i]) for i in range(len(lep_vec))]) # Psi is scattering angle
    # Save cos(psi) data to file
    file_path_psi = os.path.join(process_dir, f"Plots and data/psi_data_{run_num}.txt")
    np.savetxt(file_path_psi, cos_psi)

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
    phi = np.array([azimuthal_angle2(e_plus_intermed[:, 1:][i], w_boosted[:, 1:][i]) for i in range(len(e_plus_boosted[:, 1:]))])
    # Save phi data to a file
    file_path_phi = os.path.join(particle_directories['e+'], f"phi_data_{run_number}.txt")
    np.savetxt(file_path_phi, phi)


    # Boost calculations for mu+
    z_boosted = execute_boost(particle_arrays['z'], diboson_array)
    mu_plus_intermed = execute_boost(particle_arrays['mu+'], diboson_array)
    mu_plus_boosted = execute_boost(mu_plus_intermed, z_boosted)
    calc_polar_angle(mu_plus_boosted, z_boosted, 'mu+', run_number)
    phi = np.array([azimuthal_angle2(mu_plus_intermed[:, 1:][i], z_boosted[:, 1:][i]) for i in range(len(mu_plus_boosted[:, 1:]))])
    # Save phi data to a file
    file_path_phi = os.path.join(particle_directories['mu+'], f"phi_data_{run_number}.txt")
    np.savetxt(file_path_phi, phi)

    # Boost calculations for mu-
    mu_minus_intermed = execute_boost(particle_arrays['mu-'], diboson_array)
    mu_minus_boosted = execute_boost(mu_minus_intermed, z_boosted)
    calc_polar_angle(mu_minus_boosted, z_boosted, 'mu-', run_number)




if __name__ == "__main__":
    main()
