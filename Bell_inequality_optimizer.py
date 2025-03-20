import numpy as np
from scipy.optimize import minimize
from Unitary_Matrix import euler_unitary_matrix

def inequality_function(density_matrix, O_bell_prime, parameters):
        U_params = parameters[:8]
        V_params = parameters[8:]
        U = euler_unitary_matrix(*U_params)
        V = euler_unitary_matrix(*V_params)
        U_cross_V = np.kron(U, V)

        O_bell = U_cross_V.conj().T @ O_bell_prime @ U_cross_V
        bell_inequality = np.trace(density_matrix @ O_bell)
        return bell_inequality.real()

def bell_inequality_optimization(density_matrix, O_bell_prime):
    """
    Perform the optimization procedure to maximize the Bell inequality.
    """
    def inequality_function_pseudo(parameters):
        U_params = parameters[:8]
        V_params = parameters[8:]
        U = euler_unitary_matrix(*U_params)
        V = euler_unitary_matrix(*V_params)
        U_cross_V = np.kron(U, V)

        O_bell = U_cross_V.conj().T @ O_bell_prime @ U_cross_V
        bell_inequality = np.trace(density_matrix @ O_bell)
        return - np.real(bell_inequality)
    
    parameters = np.random.rand(16)
    result = minimize(inequality_function_pseudo, parameters, method='L-BFGS-B')
    optimal_params = result.x
    bell_value = - result.fun

    return bell_value, optimal_params
