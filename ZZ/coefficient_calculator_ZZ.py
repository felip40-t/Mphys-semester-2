import numpy as np
import os
import csv
from scipy.special import sph_harm_y
from utils.histo_plotter import read_data
from core.density_matrix_calculator import T1_operators, T2_operators
from config import ZZ_DATA_DIR  # also bootstraps src/ onto sys.path
from diboson.physics.projectors import (
    plus_minus, projector_1, projector_2, projector_3, projector_4,
    projector_5, projector_6, projector_7, projector_8, projector_vector,
    read_masked_data,
)

ZZ_path = ZZ_DATA_DIR
# Read theta and phi values for both datasets
cos_theta_paths = {
    1: os.path.join(ZZ_path, "mu+/theta_data_combined.txt"),
    3: os.path.join(ZZ_path, "e+/theta_data_combined.txt")
}
phi_paths = {
    1: os.path.join(ZZ_path, "mu+/phi_data_combined.txt"),
    3: os.path.join(ZZ_path, "e+/phi_data_combined.txt")
}

# Constants
ETA = 0.213
g_L = -0.26953
g_R = 0.2317

l_values = [1, 2]
m_values = {1: [-1, 0, 1], 2: [-2, -1, 0, 1, 2]}

a_matrix = ( 1 / (g_R**2 - g_L**2) ) * np.array([[g_R**2, 0, 0, 0, 0, g_L**2, 0, 0], 
                                                 [0, g_R**2, 0, 0, 0, 0, g_L**2, 0], 
                                                 [0, 0, g_R**2 - 0.5 * g_L**2, 0, 0, 0, 0, (np.sqrt(3)/2) * g_L**2], 
                                                 [0, 0, 0, g_R**2 - g_L**2, 0, 0, 0, 0], 
                                                 [0, 0, 0, 0, g_R**2 - g_L**2, 0, 0, 0], 
                                                 [g_L**2, 0, 0, 0, 0, g_R**2, 0, 0], 
                                                 [0, g_L**2, 0, 0, 0, 0, g_R**2, 0], 
                                                 [0, 0, (np.sqrt(3)/2) * g_L**2, 0, 0, 0, 0, 0.5 * g_L**2 - g_R**2] ])

def calculate_coefficients_fgh(theta_paths, phi_paths, mask=None):
    """
    Calculate the f, g, and h coefficients and return them as dictionaries.
    """
    # Read data and apply mask if provided
    theta_values = {1: read_data(theta_paths[1]), 3: read_data(theta_paths[3])}
    phi_values = {1: read_data(phi_paths[1]), 3: read_data(phi_paths[3])}
    
    # Initialize coefficients
    f_coefficients = np.zeros(8)
    g_coefficients = np.zeros(8)
    h_coefficients = np.zeros((8, 8))

    # Calculate projector vectors
    p_1 = a_matrix @ projector_vector(theta_values[1], phi_values[1], 1)
    p_3 = a_matrix @ projector_vector(theta_values[3], phi_values[3], 1)

    # Calculate f and g coefficients using vectorized operations
    for i in range(8):
        f_coefficients[i] = 0.5 * 0.5 * 0.5 * np.mean(p_1[i] + p_3[i])

    g_coefficients = np.copy(f_coefficients)

    # Calculate h coefficients using vectorized operations
    for i in range(8):
        for j in range(8):
            h_coefficients[i,j] = 0.5 * 0.25 * 0.25 * np.mean(p_1[i] * p_3[j] + p_3[i] * p_1[j])

    return f_coefficients, g_coefficients, h_coefficients


def calculate_coefficients_AC(theta_paths, phi_paths, mask=None):
    """
    Calculate the A and C coefficients and return them as dictionaries.
    If a mask is provided, it will be applied to the data.
    # """
    theta_values = {1: np.loadtxt(theta_paths[1]), 3: np.loadtxt(theta_paths[3])}
    phi_values = {1: np.loadtxt(phi_paths[1]), 3: np.loadtxt(phi_paths[3])}
    
    # theta_values = theta_paths
    # phi_values = phi_paths

    # Apply mask if provided
    if mask is not None:
        theta_values = {key: theta[mask] for key, theta in theta_values.items()}
        phi_values = {key: phi[mask] for key, phi in phi_values.items()}

    A_coefficients = {1: {}, 3: {}}
    C_coefficients = {}

    # Compute A coefficients
    for dataset in [1, 3]:
        for l in l_values:
            for m in m_values[l]:
                alpha = np.mean(sph_harm_y(l, m, theta_values[dataset], phi_values[dataset]))
                if l == 1:
                    A_coefficients[dataset][(l, m)] = -np.sqrt(8 * np.pi) * alpha / ETA
                elif l == 2:
                    A_coefficients[dataset][(l, m)] = np.sqrt(40 * np.pi) * alpha
            

    # ComputeC coefficients
    for l1, l3 in [(1, 1), (2, 2), (1, 2), (2, 1)]:
        for m1 in m_values[l1]:
            for m3 in m_values[l3]:
                sph_harm_1 = sph_harm_y(l1, m1, theta_values[1], phi_values[1])
                sph_harm_3 = sph_harm_y(l3, m3, theta_values[3], phi_values[3])
                product = sph_harm_1 * sph_harm_3
                gamma = np.mean(product)

                if l1 == l3:
                    if l1 == 1:
                        coeff = 8 * np.pi * gamma / (ETA ** 2)
                    elif l1 == 2:
                        coeff = 40 * np.pi * gamma
                else:
                    coeff = -8 * np.pi * np.sqrt(5) * gamma / ETA

                C_coefficients[(l1, m1, l3, m3)] = coeff

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

def find_nonzero_trace_terms(O, threshold=1e-5):
    non_zero_A1 = []
    non_zero_A3 = []
    non_zero_C = []

    I = np.identity(3)

    for l in l_values:
        for m in m_values[l]:
            T_op = T1_operators[m] if l == 1 else T2_operators[m]
            trace_value = np.trace(np.dot(O, np.kron(T_op, I)))
            if abs(trace_value) > threshold:
                non_zero_A1.append((l, m, trace_value))
            trace_value = np.trace(np.dot(O, np.kron(I, T_op)))
            if abs(trace_value) > threshold:
                non_zero_A3.append((l, m, trace_value))

    for (l1, l3) in [(1, 1), (2, 2), (1, 2), (2, 1)]:
        for m1 in m_values[l1]:
            for m3 in m_values[l3]:
                T1_op = T1_operators[m1] if l1 == 1 else T2_operators[m1]
                T3_op = T1_operators[m3] if l3 == 1 else T2_operators[m3]
                trace_value = np.trace(np.dot(O, np.kron(T1_op, T3_op)))
                if abs(trace_value) > threshold:
                    non_zero_C.append((l1, m1, l3, m3, trace_value))

    return non_zero_A1, non_zero_A3, non_zero_C

def calculate_variance_AC(theta_paths, phi_paths, O):
    """
    Calculate the variance of the Bell operator
    """
    # Load data
    theta_values = {1: np.loadtxt(theta_paths[1]), 3: np.loadtxt(theta_paths[3])}
    phi_values = {1: np.loadtxt(phi_paths[1]), 3: np.loadtxt(phi_paths[3])}
    n_samples = len(theta_values[1])

    non_zero_A1, non_zero_A3, non_zero_C = find_nonzero_trace_terms(O)

    # Build coefficient matrix and trace vector
    coeff_columns = []
    trace_vector = []

    for (l, m, trace_value) in non_zero_A1:
        if l == 1:
            const = -np.sqrt(8 * np.pi) / ETA 
        else:
            const = np.sqrt(40 * np.pi)
        A_coeff = const * sph_harm_y(l, m, theta_values[1], phi_values[1])
        coeff_columns.append(A_coeff)
        trace_vector.append(trace_value)

    for (l, m, trace_value) in non_zero_A3:
        if l == 1:
            const = -np.sqrt(8 * np.pi) / ETA 
        else:
            const = np.sqrt(40 * np.pi)
        A_coeff = const * sph_harm_y(l, m, theta_values[3], phi_values[3])
        coeff_columns.append(A_coeff)
        trace_vector.append(trace_value)

    for (l1, m1, l3, m3, trace_value) in non_zero_C:
        if l1 == l3:
            if l1 == 1:
                const = 8 * np.pi / (ETA ** 2)
            else:
                const = 40 * np.pi
        else:
            const = -8 * np.pi * np.sqrt(5) / ETA

        C_coeff = const * sph_harm_y(l1, m1, theta_values[1], phi_values[1]) * sph_harm_y(l3, m3, theta_values[3], phi_values[3])
        coeff_columns.append(C_coeff)
        trace_vector.append(trace_value)

    # Stack coefficients into a matrix
    coeff_matrix = np.column_stack(coeff_columns)
    trace_vector = np.array(trace_vector) / 9

    # Covariance matrix
    covariance_matrix = np.cov(coeff_matrix, rowvar=False) / n_samples

    # Variance
    variance = np.conj(trace_vector.T) @ covariance_matrix @ trace_vector

    return variance

