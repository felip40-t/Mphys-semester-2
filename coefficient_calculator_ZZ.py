from hepunits import g
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
g_L = -0.2766
g_R = 0.2234

l_values = [1, 2]
m_values = {1: [-1, 0, 1], 2: [-2, -1, 0, 1, 2]}

a_matrix = ( 1 / (g_L**2 - g_R**2) ) * np.array([[g_R**2, 0, 0, 0, 0, g_L**2, 0, 0], 
                                                 [0, g_R**2, 0, 0, 0, 0, g_L**2, 0], 
                                                 [0, 0, g_R**2 - 0.5 * g_L**2, 0, 0, 0, 0, (np.sqrt(3)/2) * g_L**2], 
                                                 [0, 0, 0, g_R**2 - g_L**2, 0, 0, 0, 0], 
                                                 [0, 0, 0, 0, g_R**2 - g_L**2, 0, 0, 0], 
                                                 [g_L**2, 0, 0, 0, 0, g_R**2, 0, 0], 
                                                 [0, g_L**2, 0, 0, 0, 0, g_R**2, 0], 
                                                 [0, 0, (np.sqrt(3)/2) * g_L**2, 0, 0, 0, 0, 0.5 * g_L**2 - g_R**2] ])

# Helper function to determine if it's plus or minus based on dataset value
def plus_minus(dataset):
    if dataset == 1:
        return +1
    elif dataset == 3:
        return -1

# Projector 1
def projector_1(theta, phi, dataset):
    value = np.sqrt(2) * np.sin(theta) * (5 * np.cos(theta) + plus_minus(dataset) * 1) * np.cos(phi)
    return value

# Projector 2
def projector_2(theta, phi, dataset):
    value = np.sqrt(2) * np.sin(theta) * (5 * np.cos(theta) + plus_minus(dataset) * 1) * np.sin(phi)
    return value

# Projector 3
def projector_3(theta, phi, dataset):
    value = (1/4) * (5 + plus_minus(dataset) * 4 * np.cos(theta) + 15 * np.cos(2*theta))
    return value

# Projector 4
def projector_4(theta, phi, dataset):
    return 5 * np.sin(theta)**2 * np.cos(2 * phi)

# Projector 5
def projector_5(theta, phi, dataset):
    return 5 * np.sin(theta)**2 * np.sin(2 * phi)

# Projector 6
def projector_6(theta, phi, dataset):
    value = np.sqrt(2) * np.sin(theta) * (-5 * np.cos(theta) + plus_minus(dataset) * 1) * np.cos(phi)
    return value

# Projector 7
def projector_7(theta, phi, dataset):
    value = np.sqrt(2) * np.sin(theta) * (-5 * np.cos(theta) + plus_minus(dataset) * 1) * np.sin(phi)
    return value

# Projector 8
def projector_8(theta, phi, dataset):
    value = (1 / (4 * np.sqrt(3))) * (-5 + plus_minus(dataset) * 12 * np.cos(theta) + 15 * np.cos(2*theta))
    return value

# Define the vector of projectors
def projector_vector(theta, phi, dataset):
    # Call each projector function and store their results in a list or array
    vector = np.array([
        projector_1(theta, phi, dataset),
        projector_2(theta, phi, dataset),
        projector_3(theta, phi, dataset),
        projector_4(theta, phi, dataset),
        projector_5(theta, phi, dataset),
        projector_6(theta, phi, dataset),
        projector_7(theta, phi, dataset),
        projector_8(theta, phi, dataset)
    ])
    return vector


def calculate_coefficients_fgh(theta_paths, phi_paths, mask=None):
    """
    Calculate the f, g, and h coefficients and return them as dictionaries.
    """
    # Read data and apply mask if provided
    theta_values = {1: np.arccos(read_data(theta_paths[1])), 3: np.arccos(read_data(theta_paths[3]))}
    phi_values = {1: read_data(phi_paths[1]), 3: read_data(phi_paths[3])}
    if mask is not None:
        theta_values = {key: theta[mask] for key, theta in theta_values.items()}
        phi_values = {key: phi[mask] for key, phi in phi_values.items()}

    # Initialize coefficients
    f_coefficients = np.zeros(8)
    g_coefficients = np.zeros(8)
    h_coefficients = np.zeros((8, 8))

    # Calculate projector vectors
    p_1 = a_matrix @ projector_vector(theta_values[1], phi_values[1], 1)
    p_3 = a_matrix @ projector_vector(theta_values[3], phi_values[3], 1)

    # Calculate f and g coefficients using vectorized operations
    f_coefficients = 0.5 * np.mean(p_1, axis=1)
    g_coefficients = 0.5 * np.mean(p_3, axis=1)

    # Calculate h coefficients using vectorized operations
    h_coefficients = 0.25 * np.mean(p_1[:, np.newaxis, :] * p_3[np.newaxis, :, :], axis=2)

    return f_coefficients, g_coefficients, h_coefficients


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

    return A_coefficients, C_coefficients, alpha_values, gamma_values

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

def main():
    A_coefficients, C_coefficients, alpha_values, gamma_values = calculate_coefficients_AC(cos_theta_paths, phi_paths)
    save_coefficients(A_coefficients, C_coefficients, alpha_values, gamma_values, ZZ_path)

main()

