import numpy as np
import os
import csv
from scipy.special import sph_harm_y
from histo_plotter import read_data
from density_matrix_calculator import lambda_operators, O_bell_prime1

# Constants
ETA = 1
l_values = [1, 2]
m_values = {1: [-1, 0, 1], 2: [-2, -1, 0, 1, 2]}

wp_indices = [2, 3, 4, 8]
wm_indices = [1, 2, 3, 4, 5, 7, 8]

double_indices = [
                  (1, 1), (1, 2), (1, 3), (1, 5), (1, 6), (1, 7),
                  (2, 1), (2, 3), (2, 5), (2, 7), (2, 8),
                  (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 7), (3, 8), 
                  (4, 1), (4, 3), (4, 4), (4, 5), (4, 7),  
                  (5, 2), (5, 5), (5, 6), (5, 7), (5, 8),
                  (6, 1), (6, 2), (6, 4), (6, 5), (6, 6), (6, 8), 
                  (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), 
                  (8, 2), (8, 3), (8, 5), (8, 6), (8, 7), (8, 8)    
                ]


# Convert to 0-based indexing
wp_indices = [i - 1 for i in wp_indices]
wm_indices = [i - 1 for i in wm_indices]
double_indices = [(i - 1, j - 1) for (i, j) in double_indices]


def calculate_coefficients(theta_paths, phi_paths, mask=None):
    """
    Calculate the A and C coefficients and return them as dictionaries.
    If a mask is provided, it will be applied to the data.
    """
    # theta_values = {1: read_data(theta_paths[1]), 3: read_data(theta_paths[3])}
    # phi_values = {1: read_data(phi_paths[1]), 3: read_data(phi_paths[3])}

    theta_values = theta_paths
    phi_values = phi_paths
    
    # Apply mask if provided
    if mask is not None:
        theta_values = {key: theta[mask] for key, theta in theta_values.items()}
        phi_values = {key: phi[mask] for key, phi in phi_values.items()}

    A_coefficients = {1: {}, 3: {}}
    C_coefficients = {}
    A_unc = {1: {}, 3: {}}
    C_unc = {}

    # Compute A coefficients
    for dataset in [1, 3]:
        for l in l_values:
            for m in m_values[l]:
                alpha = np.mean(sph_harm_y(l, m, theta_values[dataset], phi_values[dataset]))
                # alpha_unc = np.std(sph_harm_y(l, m, theta_values[dataset], phi_values[dataset])) / np.sqrt(len(theta_values[dataset]))
                if l == 1:
                    A_coefficients[dataset][(l, m)] = -np.sqrt(8 * np.pi) * alpha / ETA
                    # A_unc[dataset][(l, m)] = np.sqrt(8 * np.pi) * alpha_unc / ETA
                elif l == 2:
                    A_coefficients[dataset][(l, m)] = np.sqrt(40 * np.pi) * alpha
                    # A_unc[dataset][(l, m)] = np.sqrt(40 * np.pi) * alpha_unc

    # Compute C coefficients
    for l1, l3 in [(1, 1), (2, 2), (1, 2), (2, 1)]:
        for m1 in m_values[l1]:
            for m3 in m_values[l3]:
                sph_harm_1 = sph_harm_y(l1, m1, theta_values[1], phi_values[1])
                sph_harm_3 = sph_harm_y(l3, m3, theta_values[3], phi_values[3])
                product = sph_harm_1 * sph_harm_3
                gamma = np.mean(product)
                # gamma_unc = np.std(product) / np.sqrt(len(theta_values[1]))

                if l1 == l3:
                    if l1 == 1:
                        C_coefficients[(l1, m1, l3, m3)] = 8 * np.pi * gamma / (ETA ** 2)
                        # C_unc[(l1, m1, l3, m3)] = 8 * np.pi * gamma_unc / (ETA ** 2)
                    elif l1 == 2:
                        C_coefficients[(l1, m1, l3, m3)] = 40 * np.pi * gamma
                        # C_unc[(l1, m1, l3, m3)] = 40 * np.pi * gamma_unc
                else:
                    C_coefficients[(l1, m1, l3, m3)] = - 8 * np.pi * np.sqrt(5) * gamma / ETA
                    # C_unc[(l1, m1, l3, m3)] = 8 * np.pi * np.sqrt(5) * gamma_unc / ETA

    return A_coefficients, C_coefficients

def save_coefficients(A_coefficients, C_coefficients, ZZ_path):
    """
    Save the A and C coefficients to CSV files.
    """
    A_coeff_path = os.path.join(ZZ_path, "A_coefficients_run_1.csv")
    C_coeff_path = os.path.join(ZZ_path, "C_coefficients_run_1.csv")

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

def read_masked_data(cos_psi_data, ZZ_inv_mass, psi_range, mass_range):
    """
    Apply a mask based on psi and ZZ invariant mass.
    """
    return (cos_psi_data > psi_range[0]) & (cos_psi_data < psi_range[1]) & (ZZ_inv_mass > mass_range[0]) & (ZZ_inv_mass < mass_range[1])



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
    value = (1 / (4 * np.sqrt(3))) * (-5 + plus_minus(dataset) * 12 * np.cos(theta) - 15 * np.cos(2*theta))
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


def calculate_coefficients_fgh(theta_paths, phi_paths):
    """
    Calculate the f, g, and h coefficients and return them as dictionaries.
    """
    # Read data and apply mask if provided
    theta_values = {1: np.loadtxt(theta_paths[1]), 3: np.loadtxt(theta_paths[3])}
    phi_values = {1: np.loadtxt(phi_paths[1]), 3: np.loadtxt(phi_paths[3])}
    
    # Initialize coefficients
    f_coefficients = np.zeros(8)
    g_coefficients = np.zeros(8)

    h_coefficients = np.zeros((8, 8))

    # Calculate projector vectors
    p_1 = projector_vector(theta_values[1], phi_values[1], 1)
    p_3 = projector_vector(theta_values[3], phi_values[3], 3)

    # Calculate f and g coefficients for non-vanishing indices
    for i in range(8):
        #if i in wp_indices:
        f_coefficients[i] = 0.5 * np.mean(p_1[i])
        
    for j in range(8):
        # if j in wm_indices:
        g_coefficients[j] = 0.5 * np.mean(p_3[j])
        

    # Calculate h coefficients for non-vanishing indices
    for i in range(8):
        for j in range(8):
            # if (i, j) in double_indices:
            h_coefficients[i,j] = 0.25 * np.mean(p_1[i] * p_3[j])
        
    return f_coefficients, g_coefficients, h_coefficients

def find_nonzero_trace_terms(O, threshold=1e-8):
    nonzero_f = []
    nonzero_g = []
    nonzero_h = []

    I = np.eye(3)

    for i in range(8):
        trace_f = np.trace(np.kron(lambda_operators[i], I) @ O)
        if abs(trace_f) > threshold:
            nonzero_f.append((i, trace_f / 3))  # Derivative ∂/∂f[i] = Tr(λ_i ⊗ I O) / 3

        trace_g = np.trace(np.kron(I, lambda_operators[i]) @ O)
        if abs(trace_g) > threshold:
            nonzero_g.append((i, trace_g / 3))  # Derivative ∂/∂g[i] = Tr(I ⊗ λ_j O) / 3

    for i in range(8):
        for j in range(8):
            trace_h = np.trace(np.kron(lambda_operators[i], lambda_operators[j]) @ O)
            if abs(trace_h) > threshold:
                nonzero_h.append(((i, j), trace_h))  # Derivative ∂/∂h[i,j] = Tr(λ_i ⊗ λ_j O)

    return nonzero_f, nonzero_g, nonzero_h

def calculate_variance_fgh(theta_paths, phi_paths, O):
    """
    Calculate the variance for bell operator.
    """
    # Load data
    theta_values = {1: np.loadtxt(theta_paths[1]), 3: np.loadtxt(theta_paths[3])}
    phi_values = {1: np.loadtxt(phi_paths[1]), 3: np.loadtxt(phi_paths[3])}
    n_samples = len(theta_values[1])

    # Projector vectors
    p_1 = 0.5 * projector_vector(theta_values[1], phi_values[1], 1)
    p_3 = 0.5 * projector_vector(theta_values[3], phi_values[3], 3)

    # Normalize projectors
    p_1 /= np.linalg.norm(p_1, axis=0)
    p_3 /= np.linalg.norm(p_3, axis=0)

    non_zero_f, non_zero_g, non_zero_h = find_nonzero_trace_terms(O)

    # Build coefficient matrix and trace vector
    coeff_columns = []
    trace_vector = []

    for i, trace_i in non_zero_f:
        coeff_columns.append(p_1[i])       
        trace_vector.append(trace_i)

    for j, trace_j in non_zero_g:
        coeff_columns.append(p_3[j])
        trace_vector.append(trace_j)

    for (i, j), trace_ij in non_zero_h:
        coeff_columns.append(p_1[i] * p_3[j])
        trace_vector.append(trace_ij)

    # Stack into (n_samples, n_coeffs) matrix
    coeff_matrix = np.column_stack(coeff_columns)
    trace_vector = np.array(trace_vector)

    # Covariance matrix of coefficients (shape: n_coeffs x n_coeffs)
    cov_matrix = np.cov(coeff_matrix, rowvar=False)

    # Variance = a^T C a
    variance = trace_vector.T @ cov_matrix @ trace_vector

    return variance

