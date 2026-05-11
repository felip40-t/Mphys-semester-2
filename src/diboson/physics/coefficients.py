import numpy as np
from scipy.special import sph_harm_y

from diboson.physics.density_matrix import T1_operators, T2_operators, lambda_operators
from diboson.config import ZZ_ETA as _ZZ_ETA, ZZ_G_L as _g_L, ZZ_G_R as _g_R
from diboson.physics.projectors import (
    plus_minus, projector_1, projector_2, projector_3, projector_4,
    projector_5, projector_6, projector_7, projector_8, projector_vector,
    read_masked_data,
)


# ---------------------------------------------------------------------------
# ZZ (AC) module-level constants
# ---------------------------------------------------------------------------

l_values = [1, 2]
m_values = {1: [-1, 0, 1], 2: [-2, -1, 0, 1, 2]}

a_matrix = (1 / (_g_R**2 - _g_L**2)) * np.array([
    [_g_R**2, 0, 0, 0, 0, _g_L**2, 0, 0],
    [0, _g_R**2, 0, 0, 0, 0, _g_L**2, 0],
    [0, 0, _g_R**2 - 0.5 * _g_L**2, 0, 0, 0, 0, (np.sqrt(3) / 2) * _g_L**2],
    [0, 0, 0, _g_R**2 - _g_L**2, 0, 0, 0, 0],
    [0, 0, 0, 0, _g_R**2 - _g_L**2, 0, 0, 0],
    [_g_L**2, 0, 0, 0, 0, _g_R**2, 0, 0],
    [0, _g_L**2, 0, 0, 0, 0, _g_R**2, 0],
    [0, 0, (np.sqrt(3) / 2) * _g_L**2, 0, 0, 0, 0, 0.5 * _g_L**2 - _g_R**2],
])


# ---------------------------------------------------------------------------
# ZZ: AC coefficients
# ---------------------------------------------------------------------------

def calculate_coefficients_AC(theta_paths, phi_paths):
    """
    Calculate the A and C coefficients and return them as dictionaries.
    If a mask is provided, it will be applied to the data.
    Important to note that the angular data is stored in .npy files.
    """

    theta_values = {1: np.load(theta_paths[1]), 3: np.load(theta_paths[3])}
    phi_values = {1: np.load(phi_paths[1]), 3: np.load(phi_paths[3])}

    A_coefficients = {1: {}, 3: {}}
    C_coefficients = {}

    for dataset in [1, 3]:
        for l in l_values:
            for m in m_values[l]:
                alpha = np.mean(sph_harm_y(l, m, theta_values[dataset], phi_values[dataset]))
                if l == 1:
                    A_coefficients[dataset][(l, m)] = -np.sqrt(8 * np.pi) * alpha / _ZZ_ETA
                elif l == 2:
                    A_coefficients[dataset][(l, m)] = np.sqrt(40 * np.pi) * alpha

    for l1, l3 in [(1, 1), (2, 2), (1, 2), (2, 1)]:
        for m1 in m_values[l1]:
            for m3 in m_values[l3]:
                sph_harm_1 = sph_harm_y(l1, m1, theta_values[1], phi_values[1])
                sph_harm_3 = sph_harm_y(l3, m3, theta_values[3], phi_values[3])
                gamma = np.mean(sph_harm_1 * sph_harm_3)

                if l1 == l3:
                    if l1 == 1:
                        coeff = 8 * np.pi * gamma / (_ZZ_ETA ** 2)
                    elif l1 == 2:
                        coeff = 40 * np.pi * gamma
                else:
                    coeff = -8 * np.pi * np.sqrt(5) * gamma / _ZZ_ETA

                C_coefficients[(l1, m1, l3, m3)] = coeff

    return A_coefficients, C_coefficients


def _find_nonzero_trace_terms_AC(O, threshold=1e-5):
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
    Calculate the variance of the Bell operator (ZZ/AC basis).
    """
    theta_values = {1: np.load(theta_paths[1]), 3: np.load(theta_paths[3])}
    phi_values = {1: np.load(phi_paths[1]), 3: np.load(phi_paths[3])}
    n_samples = len(theta_values[1])

    non_zero_A1, non_zero_A3, non_zero_C = _find_nonzero_trace_terms_AC(O)

    coeff_columns = []
    trace_vector = []

    for (l, m, trace_value) in non_zero_A1:
        const = -np.sqrt(8 * np.pi) / _ZZ_ETA if l == 1 else np.sqrt(40 * np.pi)
        coeff_columns.append(const * sph_harm_y(l, m, theta_values[1], phi_values[1]))
        trace_vector.append(trace_value)

    for (l, m, trace_value) in non_zero_A3:
        const = -np.sqrt(8 * np.pi) / _ZZ_ETA if l == 1 else np.sqrt(40 * np.pi)
        coeff_columns.append(const * sph_harm_y(l, m, theta_values[3], phi_values[3]))
        trace_vector.append(trace_value)

    for (l1, m1, l3, m3, trace_value) in non_zero_C:
        if l1 == l3:
            const = 8 * np.pi / (_ZZ_ETA ** 2) if l1 == 1 else 40 * np.pi
        else:
            const = -8 * np.pi * np.sqrt(5) / _ZZ_ETA
        C_coeff = const * sph_harm_y(l1, m1, theta_values[1], phi_values[1]) * sph_harm_y(l3, m3, theta_values[3], phi_values[3])
        coeff_columns.append(C_coeff)
        trace_vector.append(trace_value)

    coeff_matrix = np.column_stack(coeff_columns)
    trace_vector = np.array(trace_vector) / 9

    covariance_matrix = np.cov(coeff_matrix, rowvar=False) / n_samples
    variance = np.conj(trace_vector.T) @ covariance_matrix @ trace_vector

    return variance


# ---------------------------------------------------------------------------
# WW: fgh coefficients
# ---------------------------------------------------------------------------

def calculate_coefficients_fgh(theta_paths, phi_paths):
    """
    Calculate the f, g, and h coefficients and return them as dictionaries.
    """
    theta_values = {1: np.load(theta_paths[1]), 3: np.load(theta_paths[3])}
    phi_values = {1: np.load(phi_paths[1]), 3: np.load(phi_paths[3])}

    f_coefficients = np.zeros(8)
    g_coefficients = np.zeros(8)
    h_coefficients = np.zeros((8, 8))

    p_1 = projector_vector(theta_values[1], phi_values[1], 1)
    p_3 = projector_vector(theta_values[3], phi_values[3], 3)

    for i in range(8):
        f_coefficients[i] = 0.5 * np.mean(p_1[i])

    for j in range(8):
        g_coefficients[j] = 0.5 * np.mean(p_3[j])

    for i in range(8):
        for j in range(8):
            h_coefficients[i, j] = 0.25 * np.mean(p_1[i] * p_3[j])

    return f_coefficients, g_coefficients, h_coefficients


def _find_nonzero_trace_terms_fgh(O, threshold=1e-8):
    nonzero_f = []
    nonzero_g = []
    nonzero_h = []

    I = np.eye(3)

    for i in range(8):
        trace_f = np.trace(np.kron(lambda_operators[i], I) @ O)
        if abs(trace_f) > threshold:
            nonzero_f.append((i, trace_f / 3))

        trace_g = np.trace(np.kron(I, lambda_operators[i]) @ O)
        if abs(trace_g) > threshold:
            nonzero_g.append((i, trace_g / 3))

    for i in range(8):
        for j in range(8):
            trace_h = np.trace(np.kron(lambda_operators[i], lambda_operators[j]) @ O)
            if abs(trace_h) > threshold:
                nonzero_h.append(((i, j), trace_h))

    return nonzero_f, nonzero_g, nonzero_h


def calculate_variance_fgh(theta_paths, phi_paths, O):
    """
    Calculate the variance of the Bell operator (WW/fgh basis).
    """
    theta_values = {1: np.load(theta_paths[1]), 3: np.load(theta_paths[3])}
    phi_values = {1: np.load(phi_paths[1]), 3: np.load(phi_paths[3])}
    n_samples = len(theta_values[1])

    p_1 = 0.5 * projector_vector(theta_values[1], phi_values[1], 1)
    p_3 = 0.5 * projector_vector(theta_values[3], phi_values[3], 3)

    non_zero_f, non_zero_g, non_zero_h = _find_nonzero_trace_terms_fgh(O)

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

    coeff_matrix = np.column_stack(coeff_columns)
    trace_vector = np.array(trace_vector)

    cov_matrix = np.cov(coeff_matrix, rowvar=False) / n_samples
    variance = trace_vector.T @ cov_matrix @ trace_vector

    return variance
