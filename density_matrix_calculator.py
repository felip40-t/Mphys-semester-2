import numpy as np
import csv

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

S_x = sqrt1_2 *  np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
S_y = sqrt1_2 *  np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])

lambda_1 = np.array([[0,1,0],[1,0,0],[0,0,0]])
lambda_2 = np.array([[0,-1j,0],[1j,0,0],[0,0,0]])
lambda_3 = np.array([[1,0,0],[0,-1,0],[0,0,0]])
lambda_4 = np.array([[0,0,1],[0,0,0],[1,0,0]])
lambda_5 = np.array([[0,0,-1j],[0,0,0],[1j,0,0]])
lambda_6 = np.array([[0,0,0], [0,0,1], [0,1,0]])
lambda_7 = np.array([[0,0,0], [0,0,-1j], [0,1j,0]])
lambda_8 = 1/np.sqrt(3) * np.array([[1,0,0],[0,1,0],[0,0,-2]])

lambda_operators = {0: lambda_1, 1: lambda_2, 2: lambda_3, 3: lambda_4, 4: lambda_5, 5: lambda_6, 6: lambda_7, 7: lambda_8}

O_bell_prime1 = -2/sqrt3 * (np.kron(S_x, S_x) + np.kron(S_y, S_y)) + np.kron(lambda_4, lambda_4) + np.kron(lambda_5, lambda_5)
 
#O_bell_prime2 = ( 4 / np.sqrt(27)) * ( np.kron(T1_1, T1_1) + np.kron(T1_m1, T1_m1) ) + ( 2 / 3 ) * ( np.kron(T2_2, T2_2) + np.kron(T2_m2, T2_m2) )

# np.set_printoptions(precision=3, suppress=True)
# print("O_bell_prime1:")
# print(O_bell_prime1.real)
# print("\nO_bell_prime2:")
# print(O_bell_prime2)


def calculate_density_matrix_AC(A_coefficients, C_coefficients):
    """
    Construct the density matrix using the A and C coefficients.
    """
    density_matrix = np.kron(I_3, I_3).astype(complex)
    
    for dataset, inner_dict in A_coefficients.items():
        for (l, m), A_value in inner_dict.items():
            T_op = T1_operators[m] if l == 1 else T2_operators[m]
            if dataset == 1:
                density_matrix += A_value * np.kron(T_op, I_3)
            elif dataset == 3:
                density_matrix += A_value * np.kron(I_3, T_op)

    for (l1, m1, l3, m3), C_value in C_coefficients.items():
        T1_op = T1_operators[m1] if l1 == 1 else T2_operators[m1]
        T2_op = T1_operators[m3] if l3 == 1 else T2_operators[m3]
        density_matrix += C_value * np.kron(T1_op, T2_op)

    density_matrix *= 1/9
    return density_matrix

def calculate_density_matrix_fgh(f_coefficients, g_coefficients, h_coefficients):
    """
    Construct the density matrix using the f, g, and h coefficients.
    """
    density_matrix = (1/9) * np.kron(I_3, I_3).astype(complex)
    for i in range(8):
        density_matrix += f_coefficients[i] * np.kron(lambda_operators[i], I_3)
        density_matrix += g_coefficients[i] * np.kron(I_3, lambda_operators[i])
    for (i, j) in [(i, j) for i in range(8) for j in range(8)]:
        density_matrix += h_coefficients[i, j] * np.kron(lambda_operators[i], lambda_operators[j])

    return density_matrix