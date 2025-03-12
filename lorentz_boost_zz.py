import os
import numpy as np
from lhe_reading_WW import find_latest_run_dir
from histo_plotter import read_data

# Update the MadGraph5 directory and process directory to ZZ_process
mg5_install_dir = "/home/felipetcach/project/MG5_aMC_v3_5_6"
process_dir = os.path.join(mg5_install_dir, "pp_ZZ_SM")  # Updated process directory
base_dir = os.path.join(process_dir, "Events")

# Particle directories corresponding to the ZZ process
particle_directories = {
    'mu+': os.path.join(process_dir, "Plots and data/mu+"),  # mu+
    'mu-': os.path.join(process_dir, "Plots and data/mu-"),  # mu-
    'e+': os.path.join(process_dir, "Plots and data/e+"),    # e+
    'e-': os.path.join(process_dir, "Plots and data/e-"),    # e-
}

# Function to compute the Lorentz boost matrix
def lorentz_boost(beta_vector):
    total_beta = np.linalg.norm(beta_vector)
    gamma = 1 / np.sqrt(1 - total_beta**2)
    boost = np.identity(4)
    beta_vector = np.array(beta_vector)
    boost[1:, 1:] += (gamma - 1) * np.outer(beta_vector, beta_vector) / total_beta**2
    boost[0, 1:] = boost[1:, 0] = -gamma * beta_vector
    boost[0, 0] = gamma
    return boost

# Calculate beta from the four-momentum
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
    parent_vec = parent_array[:, 1:]  # Take spatial components of parent boson

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

def boostinvp(q, pboost, qprime=None):
    """
    Boost routine for relativistic transformations.
    Boosts a 4-momentum into a different reference frame.
    
    Parameters:
    -----------
    q : numpy.ndarray
        4-momentum to be boosted (E, px, py, pz)
    pboost : numpy.ndarray
        4-momentum defining the boost (E, px, py, pz)
    qprime : numpy.ndarray, optional
        Array to store the result, if None a new array is created
        
    Returns:
    --------
    qprime : numpy.ndarray
        Boosted 4-momentum (E, px, py, pz)
    """
    if qprime is None:
        qprime = np.zeros(4)
    
    # Calculate invariant mass squared of pboost
    rmboost = pboost[0]**2 - pboost[1]**2 - pboost[2]**2 - pboost[3]**2
    rmboost = np.sqrt(max(rmboost, 0.0))
    
    # Calculate scalar product
    aux = (q[0]*pboost[0] - q[1]*pboost[1] - q[2]*pboost[2] - q[3]*pboost[3]) / rmboost
    aaux = (aux + q[0]) / (pboost[0] + rmboost)
    
    # Apply the boost
    qprime[0] = aux
    qprime[1] = q[1] - aaux * pboost[1]
    qprime[2] = q[2] - aaux * pboost[2]
    qprime[3] = q[3] - aaux * pboost[3]
    
    return qprime


def rotinvp(p, q, pp=None):
    """
    Rotation routine for relativistic transformations.
    Rotates a 3-momentum vector.
    
    Parameters:
    -----------
    p : numpy.ndarray
        3-momentum to be rotated (px, py, pz)
    q : numpy.ndarray
        3-momentum defining the rotation axis (px, py, pz)
    pp : numpy.ndarray, optional
        Array to store the result, if None a new array is created
        
    Returns:
    --------
    pp : numpy.ndarray
        Rotated 3-momentum (px, py, pz)
    """
    if pp is None:
        pp = np.zeros(3)
    
    # Calculate transverse and total momentum
    qmodt = q[0]**2 + q[1]**2
    qmod = qmodt + q[2]**2
    
    qmodt = np.sqrt(qmodt)
    qmod = np.sqrt(qmod)
    
    if qmod == 0.0:
        raise ValueError("ERROR in subroutine rotinvp: spatial q components are 0.0!")
    
    cth = q[2] / qmod
    sth = 1.0 - cth**2
    
    if sth == 0.0:
        pp[0] = p[0]
        pp[1] = p[1]
        pp[2] = p[2]
        return pp
    
    sth = np.sqrt(sth)
    
    if qmodt == 0.0:
        pp[0] = p[0]
        pp[1] = p[1]
        pp[2] = p[2]
        return pp
    
    cfi = q[0] / qmodt
    sfi = q[1] / qmodt
    
    # Store p values to avoid problems if p and pp are the same vector
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    
    # Perform the rotation
    pp[0] = cth * cfi * p1 + cth * sfi * p2 - sth * p3
    pp[1] = -sfi * p1 + cfi * p2
    pp[2] = sth * cfi * p1 + sth * sfi * p2 + cth * p3
    
    return pp


def phistar(v1, v2, v3, v4):
    """
    Calculate azimuthal decay angles.
    This function calculates the azimuthal decay angles in a specific reference frame.
    The procedure follows these steps:
    1. Boost all particles to the center-of-mass frame of the system
    2. Rotate to align the bosons with the z-axis
    3. Boost charged leptons to their respective boson rest frames
    4. Calculate azimuthal angles
    """
    # Add the 4-vectors to get the combined systems
    v12 = v1 + v2
    v34 = v3 + v4
    vv = v12 + v34
    
    # Initialize arrays for all the boosted and rotated vectors
    bv12 = np.zeros(4)
    bv34 = np.zeros(4)
    bv1 = np.zeros(4)
    bv2 = np.zeros(4)
    bv3 = np.zeros(4)
    bv4 = np.zeros(4)
    
    # Boost into the center-of-mass frame of the entire system
    boostinvp(v12, vv, bv12)
    boostinvp(v34, vv, bv34)
    boostinvp(v1, vv, bv1)
    boostinvp(v2, vv, bv2)
    boostinvp(v3, vv, bv3)
    boostinvp(v4, vv, bv4)
    
    # Create vectors for the rotated particles
    bbv1 = np.zeros(4)
    bbv2 = np.zeros(4)
    bbv3 = np.zeros(4)
    bbv4 = np.zeros(4)
    
    # Keep the energy component unchanged
    bbv1[0] = bv1[0]
    bbv2[0] = bv2[0]
    bbv3[0] = bv3[0]
    bbv4[0] = bv4[0]
    
    # Rotate the spatial components to align with z-axis
    rotinvp(bv1[1:4], bv12[1:4], bbv1[1:4])
    rotinvp(bv2[1:4], bv12[1:4], bbv2[1:4])
    rotinvp(bv3[1:4], bv34[1:4], bbv3[1:4])
    rotinvp(bv4[1:4], bv34[1:4], bbv4[1:4])
    
    # Combine the rotated vectors
    bbv12 = bbv1 + bbv2
    bbv34 = bbv3 + bbv4
    
    # Final boost to each boson rest frame
    bbbv1 = np.zeros(4)
    bbbv3 = np.zeros(4)
    boostinvp(bbv1, bbv12, bbbv1)
    boostinvp(bbv3, bbv34, bbbv3)
    # Calculate the azimuthal angles    
    phi1 = np.arctan2(bbbv1[2], bbbv1[1])  # Using components 2,1 for y,x
    phi3 = np.arctan2(bbbv3[2], bbbv3[1])

    # Calculate polar angles
    theta1 = np.arccos(bbbv1[3] / np.linalg.norm(bbbv1[1:4]))
    theta3 = np.arccos(bbbv3[3] / np.linalg.norm(bbbv3[1:4]))
    
    return phi1, phi3, theta1, theta3


# Main function for processing data and calculating the Lorentz boost and angles
def main():
    # Find the latest run directory and run number
    _, run_number = find_latest_run_dir(base_dir)
    run_number = 4

    # Read the four-momenta data for all particles in the process
    particle_arrays = {particle_name: read_data(os.path.join(directory, f"data_{run_number}.txt"))
                      for particle_name, directory in particle_directories.items()}

    # Reconstruct the total diboson system
    diboson_array = sum(particle_arrays.values())

    # # Reconstruct Zs
    # z1_array = particle_arrays['e+'] + particle_arrays['e-']
    # z2_array = particle_arrays['mu+'] + particle_arrays['mu-']

    # # Boost into diboson CM
    # z1_boosted = execute_boost(particle_arrays['z1'], diboson_array)
    # e_plus_intermed = execute_boost(particle_arrays['e+'], diboson_array)

    # # Rotate system
    # e_plus_rotated = []
    # z1_rotated = []
    # for e_plus_vec, diboson_vec in zip(e_plus_intermed, diboson_array):
    #     rotated_vec = rotinvp(e_plus_vec[1:], diboson_vec[1:])
    #     rotated_vec = np.insert(rotated_vec, 0, e_plus_vec[0])
    #     e_plus_rotated.append(rotated_vec)
    # e_plus_rotated = np.array(e_plus_rotated)

    # for z1_vec, diboson_vec in zip(z1_boosted, diboson_array):
    #     rotated_vec = rotinvp(z1_vec[1:], diboson_vec[1:])
    #     rotated_vec = np.insert(rotated_vec, 0, z1_vec[0])
    #     z1_rotated.append(rotated_vec)
    # z1_rotated = np.array(z1_rotated)

    # # Boost into the z1 rest frame
    # e_plus_boosted = execute_boost(e_plus_rotated, z1_rotated)

    # # Calculate polar angles for each event
    # calc_polar_angle(e_plus_boosted, z1_rotated, 'e+', run_number)
    # # Calculate azimuthal angles for each event
    # phi = np.array([np.arctan2(row[2], row[1]) for row in e_plus_boosted])

    # Calculate azimuthal angles for each event
    phi1_list = []
    phi3_list = []
    theta1_list = []
    theta3_list = []
    for i in range(len(particle_arrays['e+'])):
        phi1, phi3, theta1, theta3 = phistar(particle_arrays['e+'][i], particle_arrays['e-'][i], particle_arrays['mu+'][i], particle_arrays['mu-'][i])
        phi1_list.append(phi1)
        phi3_list.append(phi3)
        theta1_list.append(theta1)
        theta3_list.append(theta3)
    phi1 = np.array(phi1_list)
    phi3 = np.array(phi3_list)
    theta1 = np.array(theta1_list)
    theta3 = np.array(theta3_list)
    # Save phi data to a file
    file_path_phi_ep = os.path.join(particle_directories['e+'], f"phi_data_{run_number}_new.txt")
    np.savetxt(file_path_phi_ep, phi1)
    file_path_phi_mp = os.path.join(particle_directories['mu+'], f"phi_data_{run_number}_new.txt")
    np.savetxt(file_path_phi_mp, phi3)

    # Save theta data to a file
    file_path_theta_ep = os.path.join(particle_directories['e+'], f"theta_data_{run_number}_new.txt")
    np.savetxt(file_path_theta_ep, theta1)
    file_path_theta_mp = os.path.join(particle_directories['mu+'], f"theta_data_{run_number}_new.txt")
    np.savetxt(file_path_theta_mp, theta3)


    # # Boost calculations for mu+
    # z2_boosted = execute_boost(particle_arrays['z2'], diboson_array)
    # mu_plus_intermed = execute_boost(particle_arrays['mu+'], diboson_array)

    # # Rotate system
    # mu_plus_rotated = []
    # z2_rotated = []
    # for mu_plus_vec, diboson_vec in zip(mu_plus_intermed, diboson_array):
    #     rotated_vec = rotinvp(mu_plus_vec[1:], diboson_vec[1:])
    #     rotated_vec = np.insert(rotated_vec, 0, mu_plus_vec[0])
    #     mu_plus_rotated.append(rotated_vec)
    # mu_plus_rotated = np.array(mu_plus_rotated)

    # for z2_vec, diboson_vec in zip(z2_boosted, diboson_array):
    #     rotated_vec = rotinvp(z2_vec[1:], diboson_vec[1:])
    #     rotated_vec = np.insert(rotated_vec, 0, z2_vec[0])
    #     z2_rotated.append(rotated_vec)
    # z2_rotated = np.array(z2_rotated)

    # # Boost into the z2 rest frame
    # mu_plus_boosted = execute_boost(mu_plus_rotated, z2_rotated)

    # # Calculate polar angles for each event
    # calc_polar_angle(mu_plus_boosted, z2_rotated, 'mu+', run_number)
    # # Calculate azimuthal angles for each event
    # phi = np.array([np.arctan2(row[2], row[1]) for row in mu_plus_boosted])
    # Save phi data to a file

    ZZ_inv_mass = np.apply_along_axis(calc_inv_mass, 1, diboson_array)
    file_path_inv_mass = os.path.join(process_dir, f"Plots and data/ZZ_inv_mass_{run_number}.txt")
    np.savetxt(file_path_inv_mass, ZZ_inv_mass)

if __name__ == "__main__":
    main()



