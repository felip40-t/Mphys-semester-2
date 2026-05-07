import os
import numpy as np
from utils.histo_plotter import read_data
from config import MG5_INSTALL_DIR  # also bootstraps src/ onto sys.path
from diboson.physics.kinematics import boostinvp, calc_inv_mass, calc_scattering_angle, phistar

process_dir = os.path.join(MG5_INSTALL_DIR, "pp_ZZ_SM")
base_dir = os.path.join(process_dir, "Events")

particle_directories = {
    'mu+': os.path.join(process_dir, "Plots and data/mu+"),
    'mu-': os.path.join(process_dir, "Plots and data/mu-"),
    'e+': os.path.join(process_dir, "Plots and data/e+"),
    'e-': os.path.join(process_dir, "Plots and data/e-"),
}


def main():

    particle_arrays = {particle_name: read_data(os.path.join(directory, f"combined_data_temp.txt"))
                      for particle_name, directory in particle_directories.items()}

    diboson_array = sum(particle_arrays.values())

    z1_array = particle_arrays['e+'] + particle_arrays['e-']

    # Calculate scattering angle of boosted z1 with beam axis
    z1_boosted_list = []
    for i in range(len(z1_array)):
        z1_boosted = np.zeros(4)
        boostinvp(z1_array[i], diboson_array[i], z1_boosted)
        z1_boosted_list.append(z1_boosted)
        if (i % 100000 == 0):
            print(f"Boosting Z event {i}")
    z1_boosted = np.array(z1_boosted_list)
    cos_psi = np.array([calc_scattering_angle(z1_boosted[i]) for i in range(len(z1_boosted))])
    file_path_psi = os.path.join(process_dir, f"Plots and data/psi_data_combined_temp.txt")
    np.savetxt(file_path_psi, cos_psi)

    # Calculate decay angles for each event
    phi1_list = []
    phi3_list = []
    theta1_list = []
    theta3_list = []
    for i in range(len(particle_arrays['e+'])):
        if (i % 100000 == 0):
            print(f"Processing event {i}")
        phi1, phi3, theta1, theta3 = phistar(particle_arrays['e+'][i], particle_arrays['e-'][i], particle_arrays['mu+'][i], particle_arrays['mu-'][i])
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
    file_path_phi_mp = os.path.join(particle_directories['mu+'], f"phi_data_combined_temp.txt")
    np.savetxt(file_path_phi_mp, phi3)

    file_path_theta_ep = os.path.join(particle_directories['e+'], f"theta_data_combined_temp.txt")
    np.savetxt(file_path_theta_ep, theta1)
    file_path_theta_mp = os.path.join(particle_directories['mu+'], f"theta_data_combined_temp.txt")
    np.savetxt(file_path_theta_mp, theta3)

    ZZ_inv_mass = np.apply_along_axis(calc_inv_mass, 1, diboson_array)
    file_path_inv_mass = os.path.join(process_dir, f"Plots and data/ZZ_inv_mass_combined_temp.txt")
    np.savetxt(file_path_inv_mass, ZZ_inv_mass)

if __name__ == "__main__":
    main()
