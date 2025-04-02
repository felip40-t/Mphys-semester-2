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
}

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

def calc_scattering_angle(parent_4_mom):
    parent_axis = parent_4_mom[1:] / np.linalg.norm(parent_4_mom[1:])
    beam_axis = np.array([0,0,1])
    cos_psi = parent_axis @ beam_axis
    return cos_psi

# Calculate invariant mass
def calc_inv_mass(four_vec):
    return np.sqrt(four_vec[0]**2 - np.sum(four_vec[1:]**2))


def main():

    particle_arrays = {particle_name: read_data(os.path.join(directory, f"combined_data_temp.txt"))
                      for particle_name, directory in particle_directories.items()}

    # Reconstruct total diboson system
    diboson_array = sum(particle_arrays.values())

    # Reconstruct first w boson
    wp_array = particle_arrays['e+'] + particle_arrays['ve']

    # Calculate scattering angle of boosted w
    wp_boosted_list = []
    for i in range(len(wp_array)):
        wp_boosted = np.zeros(4)
        boostinvp(wp_array[i], diboson_array[i], wp_boosted)
        wp_boosted_list.append(wp_boosted)
    wp_boosted = np.array(wp_boosted_list)

    cos_psi = np.array([calc_scattering_angle(wp_boosted[i] for i in range(len(wp_boosted)))])
    file_path_psi = os.path.join(process_dir, f"Plots and data/psi_data_combined_temp.txt")
    np.savetxt(file_path_psi, cos_psi)

    # Calculate diboson invariant mass
    WW_inv_mass = np.apply_along_axis(calc_inv_mass, 1, diboson_array)
    file_path_inv_mass = os.path.join(process_dir, f"Plots and data/WW_inv_mass_combined_temp.txt")
    np.savetxt(file_path_inv_mass, WW_inv_mass)

    # Calculate decay angles for each event
    phi1_list = []
    phi3_list = []
    theta1_list = []
    theta3_list = []
    for i in range(len(particle_arrays['e+'])):
        if (i % 100000 == 0):
            print(f"Processing event {i}")
        phi1, phi3, theta1, theta3 = phistar(particle_arrays['e+'][i], particle_arrays['ve'][i], particle_arrays['mu-'][i], particle_arrays['vm~'][i])
        phi1_list.append(phi1)
        phi3_list.append(phi3)
        theta1_list.append(theta1)
        theta3_list.append(theta3)
    phi1 = np.array(phi1_list)
    phi3 = np.array(phi3_list)
    theta1 = np.array(theta1_list)
    theta3 = np.array(theta3_list)
    
    # Save phi data to a file
    file_path_phi_ep = os.path.join(particle_directories['e+'], f"phi_data_combined_temp.txt")
    np.savetxt(file_path_phi_ep, phi1)
    file_path_phi_mp = os.path.join(particle_directories['mu-'], f"phi_data_combined_temp.txt")
    np.savetxt(file_path_phi_mp, phi3)

    # Save theta data to a file
    file_path_theta_ep = os.path.join(particle_directories['e+'], f"theta_data_combined_temp.txt")
    np.savetxt(file_path_theta_ep, theta1)
    file_path_theta_mp = os.path.join(particle_directories['mu-'], f"theta_data_combined_temp.txt")
    np.savetxt(file_path_theta_mp, theta3)


if __name__ == "__main__":
    main()
