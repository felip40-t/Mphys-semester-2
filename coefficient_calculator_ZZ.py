import numpy as np
import os
import csv
from scipy.special import sph_harm_y
from histo_plotter import read_data

ZZ_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_ZZ_SM/Plots and data"
# Read theta and phi values for both datasets
cos_theta_paths = {
    1: os.path.join(ZZ_path, "mu+/theta_data_4.txt"),
    3: os.path.join(ZZ_path, "e+/theta_data_4.txt")
}
phi_paths = {
    1: os.path.join(ZZ_path, "mu+/phi_data_4.txt"),
    3: os.path.join(ZZ_path, "e+/phi_data_4.txt")
}

# Constants
ETA = 0.214
l_values = [1, 2]
m_values = {1: [-1, 0, 1], 2: [-2, -1, 0, 1, 2]}

def calculate_coefficients_AC(theta_paths, phi_paths, mask=None):
    """
    Calculate the A and C coefficients and return them as dictionaries.
    If a mask is provided, it will be applied to the data.
    """
    theta_values = {1: np.arccos(read_data(theta_paths[1])), 3: np.arccos(read_data(theta_paths[3]))}
    phi_values = {1: read_data(phi_paths[1]), 3: read_data(phi_paths[3])}
    
    # Apply mask if provided
    if mask is not None:
        theta_values = {key: theta[mask] for key, theta in theta_values.items()}
        phi_values = {key: phi[mask] for key, phi in phi_values.items()}

    A_coefficients = {1: {}, 3: {}}
    C_coefficients = {}
    alpha_values = {1: {}, 3: {}}
    gamma_values = {}

    # Compute A coefficients
    for dataset in [1, 3]:
        for l in l_values:
            for m in m_values[l]:
                alpha = np.mean(sph_harm_y(l, m, theta_values[dataset], phi_values[dataset])).astype(complex)
                alpha_values[dataset][(l, m)] = alpha
                if l == 1:
                    A_coefficients[dataset][(l, m)] = -np.sqrt(8 * np.pi) * alpha / ETA
                elif l == 2:
                    A_coefficients[dataset][(l, m)] = np.sqrt(40 * np.pi) * alpha

    # Compute C coefficients
    for l1, l3 in [(1, 1), (2, 2), (1, 2), (2, 1)]:
        for m1 in m_values[l1]:
            for m3 in m_values[l3]:
                gamma = np.mean(sph_harm_y(l1, m1, theta_values[1], phi_values[1]) *
                                sph_harm_y(l3, m3, theta_values[3], phi_values[3])).astype(complex)
                gamma_values[(l1, m1, l3, m3)] = gamma
                
                if l1 == l3:
                    if l1 == 1:
                        C_coefficients[(l1, m1, l3, m3)] = 8 * np.pi * gamma / (ETA ** 2)
                    elif l1 == 2:
                        C_coefficients[(l1, m1, l3, m3)] = 40 * np.pi * gamma
                else:
                    C_coefficients[(l1, m1, l3, m3)] = - 8 * np.pi * np.sqrt(5) * gamma / ETA

    return A_coefficients, C_coefficients

def save_coefficients(A_coefficients, C_coefficients, alpha_values, gamma_values, ZZ_path):
    """
    Save the A and C coefficients, and alpha and gamma values to CSV files.
    """
    A_coeff_path = os.path.join(ZZ_path, "A_coefficients_run_4.csv")
    C_coeff_path = os.path.join(ZZ_path, "C_coefficients_run_4.csv")
    alpha_path = os.path.join(ZZ_path, "alpha_values_run_4.csv")
    gamma_path = os.path.join(ZZ_path, "gamma_values_run_4.csv")

    with open(A_coeff_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "l", "m", "A_coeff"])
        for dataset in A_coefficients:
            for (l, m), A_value in A_coefficients[dataset].items():
                writer.writerow([dataset, l, m, A_value])

    with open(C_coeff_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["l1", "m1", "l3", "m3", "C_coeff"])
        for (l1, m1, l3, m3), C_value in C_coefficients.items():
            writer.writerow([l1, m1, l3, m3, C_value])

    with open(alpha_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "l", "m", "alpha"])
        for dataset in alpha_values:
            for (l, m), alpha_value in alpha_values[dataset].items():
                writer.writerow([dataset, l, m, alpha_value])

    with open(gamma_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["l1", "m1", "l3", "m3", "gamma"])
        for (l1, m1, l3, m3), gamma_value in gamma_values.items():
            writer.writerow([l1, m1, l3, m3, gamma_value])

def read_masked_data(cos_psi_data, ZZ_inv_mass, psi_range, mass_range):
    """
    Apply a mask based on psi and ZZ invariant mass.
    """
    return (cos_psi_data > psi_range[0]) & (cos_psi_data < psi_range[1]) & (ZZ_inv_mass > mass_range[0]) & (ZZ_inv_mass < mass_range[1])

# def main():
#     A_coefficients, C_coefficients, alpha_values, gamma_values = calculate_coefficients(cos_theta_paths, phi_paths)
#     save_coefficients(A_coefficients, C_coefficients, alpha_values, gamma_values, ZZ_path)

