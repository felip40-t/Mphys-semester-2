import os
import numpy as np
from config import MG5_INSTALL_DIR  # also bootstraps src/ onto sys.path
from diboson.physics.kinematics import boostinvp, calc_inv_mass, calc_scattering_angle, phistar

process_dir = os.path.join(MG5_INSTALL_DIR, "pp_WW_SM")
base_dir = os.path.join(process_dir, "Events")

particle_directories = {
    'mu-': os.path.join(process_dir, "Plots and data/mu-"),
    'e+': os.path.join(process_dir, "Plots and data/e+"),
    've': os.path.join(process_dir, "Plots and data/ve"),
    'vm~': os.path.join(process_dir, "Plots and data/vm~"),
}


def main():

    particle_arrays = {particle_name: np.loadtxt(os.path.join(directory, "combined_data_temp.txt"), delimiter=',')
                      for particle_name, directory in particle_directories.items()}

    diboson_array = sum(particle_arrays.values())

    wp_array = particle_arrays['e+'] + particle_arrays['ve']

    # Calculate scattering angle of boosted w
    wp_boosted_list = []
    for i in range(len(wp_array)):
        wp_boosted = np.zeros(4)
        boostinvp(wp_array[i], diboson_array[i], wp_boosted)
        wp_boosted_list.append(wp_boosted)
    wp_boosted = np.array(wp_boosted_list)

    cos_psi = np.array([calc_scattering_angle(wp_boosted[i]) for i in range(len(wp_boosted))])
    file_path_psi = os.path.join(process_dir, f"Plots and data/psi_data_combined_temp.txt")
    np.savetxt(file_path_psi, cos_psi)

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

    file_path_phi_ep = os.path.join(particle_directories['e+'], f"phi_data_combined_temp.txt")
    np.savetxt(file_path_phi_ep, phi1)
    file_path_phi_mp = os.path.join(particle_directories['mu-'], f"phi_data_combined_temp.txt")
    np.savetxt(file_path_phi_mp, phi3)

    file_path_theta_ep = os.path.join(particle_directories['e+'], f"theta_data_combined_temp.txt")
    np.savetxt(file_path_theta_ep, theta1)
    file_path_theta_mp = os.path.join(particle_directories['mu-'], f"theta_data_combined_temp.txt")
    np.savetxt(file_path_theta_mp, theta3)


if __name__ == "__main__":
    main()
