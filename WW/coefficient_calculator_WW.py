import numpy as np
import os

from core.density_matrix_calculator import lambda_operators
from config import WW_ETA as ETA  # noqa: F401  bootstraps src/ onto sys.path
from diboson.physics.projectors import (
    plus_minus, projector_1, projector_2, projector_3, projector_4,
    projector_5, projector_6, projector_7, projector_8, projector_vector,
)



def calculate_coefficients_fgh(theta_paths, phi_paths):
    """
    Calculate the f, g, and h coefficients and return them as dictionaries.
    """
    # Read data
    theta_values = {1: np.loadtxt(theta_paths[1]), 3: np.loadtxt(theta_paths[3])}
    phi_values = {1: np.loadtxt(phi_paths[1]), 3: np.loadtxt(phi_paths[3])}
    
    # Initialize coefficients
    f_coefficients = np.zeros(8)
    g_coefficients = np.zeros(8)

    h_coefficients = np.zeros((8, 8))

    # Calculate projector vectors
    p_1 = projector_vector(theta_values[1], phi_values[1], 1)
    p_3 = projector_vector(theta_values[3], phi_values[3], 3)

    # Calculate f and g coefficients
    for i in range(8):
        f_coefficients[i] = 0.5 * np.mean(p_1[i])
        
    for j in range(8):
        g_coefficients[j] = 0.5 * np.mean(p_3[j])
        

    # Calculate h coefficients
    for i in range(8):
        for j in range(8):
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
    cov_matrix = np.cov(coeff_matrix, rowvar=False) / n_samples
    # cov_matrix = np.diag(np.diag(full_cov_matrix))

    # Variance = 
    variance = trace_vector.T @ cov_matrix @ trace_vector

    return variance

