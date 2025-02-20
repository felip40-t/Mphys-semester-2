import numpy as np
import os
import csv
from scipy.optimize import minimize
from Unitary_Matrix import euler_unitary_matrix

sqrt3 = np.sqrt(3)
sqrt3_2 = np.sqrt(3/2)
sqrt1_2 = 1/np.sqrt(2)

T1_m1 = sqrt3_2 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
T1_0 = sqrt3_2 * np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
T1_1 = sqrt3_2 * np.array([[0, -1, 0], [0, 0, -1], [0, 0, 0]])

T2_m2 = sqrt3 * np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
T2_m1 = sqrt3_2 * np.array([[0, 0, 0], [1, 0, 0], [0, -1, 0]])
T2_0 = sqrt1_2 * np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]])
T2_1 = sqrt3_2 * np.array([[0, -1, 0], [0, 0, 1], [0, 0, 0]])
T2_2 = sqrt3 * np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])

T1_operators = {(-1): T1_m1, 0: T1_0, 1: T1_1}
T2_operators = {(-2): T2_m2, (-1): T2_m1, 0: T2_0, 1: T2_1, 2: T2_2}
I_3 = np.identity(3)


def kron(T1_op, T2_op):
    return np.kron(T1_op, T2_op)

# Function to read the A and C coefficients from CSV
def read_coefficients(file_path):
    coefficients = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            if 'A' in file_path:  # A coefficients
                dataset, l, m, value = int(row[0]), int(row[1]), int(row[2]), complex(row[3])
                coefficients[(dataset, l, m)] = value
            else:  # C coefficients
                l1, m1, l3, m3, value = int(row[0]), int(row[1]), int(row[2]), int(row[3]), complex(row[4])
                coefficients[(l1, m1, l3, m3)] = value
    return coefficients

WW_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_WW_SM/Plots and data"
A_coefficients_file = os.path.join(WW_path, "A_coefficients_run_1.csv")
C_coefficients_file = os.path.join(WW_path, "C_coefficients_run_1.csv")

A_coefficients = read_coefficients(A_coefficients_file)
C_coefficients = read_coefficients(C_coefficients_file)


density_matrix = kron(I_3, I_3).astype(complex)

# Compute the density matrix A coefficients
for (dataset, l, m), A_value in A_coefficients.items():
    if l == 1:
        T_op = T1_operators[m]
    elif l == 2:
        T_op = T2_operators[m]

    if dataset == 1:
        density_matrix += A_value * np.kron(T_op, I_3)
    elif dataset == 3:
        density_matrix += A_value * np.kron(I_3, T_op)

# Compute the density matrix for C coefficients
for (l1, m1, l3, m3), C_value in C_coefficients.items():
    T1_op = T1_operators[m1] if l1 == 1 else T2_operators[m1]
    T2_op = T1_operators[m3] if l3 == 1 else T2_operators[m3]
    density_matrix += C_value * np.kron(T1_op, T2_op)

density_matrix *= 1/9

O_bell_prime = ( 4 / np.sqrt(27)) * ( np.kron(T1_1, T1_1) + np.kron(T1_m1, T1_m1) ) + ( 2 / 3 ) * ( np.kron(T2_2, T2_2) + np.kron(T2_m2, T2_m2) )


def inequality_function(parameters):
    U_params = parameters[:8]
    V_params = parameters[8:]

    U = euler_unitary_matrix(U_params[0], U_params[1], U_params[2], U_params[3], U_params[4], U_params[5], U_params[6], U_params[7])
    V = euler_unitary_matrix(V_params[0], V_params[1], V_params[2], V_params[3], V_params[4], V_params[5], V_params[6], V_params[7])
    U_cross_V = kron(U, V)

    O_bell = U_cross_V.conj().T @ O_bell_prime @ U_cross_V
    bell_inequality = np.trace(density_matrix @ O_bell)
    return - np.real(bell_inequality)

parameters = np.random.rand(16)

result = minimize(inequality_function, parameters, method='trust-constr')
optimal_params = result.x
U_params = optimal_params[:8]
V_params = optimal_params[8:]
U = euler_unitary_matrix(U_params[0], U_params[1], U_params[2], U_params[3], U_params[4], U_params[5], U_params[6], U_params[7])
V = euler_unitary_matrix(V_params[0], V_params[1], V_params[2], V_params[3], V_params[4], V_params[5], V_params[6], V_params[7])
print(f"Maximized expectation value: {-result.fun}")
print("U matrix:\n")
for row in U:
    print(" ".join(f"{x:6.2f}" for x in row))
print("\nV matrix:\n")
for row in V:
    print(" ".join(f"{x:6.2f}" for x in row))