import numpy as np
from scipy.optimize import differential_evolution
from Unitary_Matrix import euler_unitary_matrix

def optimal_bell_operator(O_bell_prime, parameters):
        """
        Calculate the optimal Bell operator using the given parameters.
        """
        U_params = parameters[:6]
        V_params = parameters[6:]
        U = euler_unitary_matrix(*U_params)
        V = euler_unitary_matrix(*V_params)
        U_cross_V = np.kron(U, V)
        
        O_bell = U_cross_V.conj().T @ O_bell_prime @ U_cross_V
        return O_bell

def inequality_function(density_matrix, O_bell_prime, parameters):
        O_bell = optimal_bell_operator(O_bell_prime, parameters)
        bell_inequality = np.trace(density_matrix @ O_bell)
        return bell_inequality

def inequality_function_pseudo(parameters, density_matrix, O_bell_prime):
        O_bell = optimal_bell_operator(O_bell_prime, parameters)
        bell_inequality = np.trace(density_matrix @ O_bell)
        return - np.real(bell_inequality)

def bell_inequality_optimization(density_matrix, O_bell_prime):
    """
    Perform the optimization procedure to maximize the Bell inequality.
    """
    bounds = [(0, 2 * np.pi)] * 12  # Define bounds for the parameters

    # Enable parallelization by setting workers=-1 in differential_evolution
    result = differential_evolution(inequality_function_pseudo, bounds, args=(density_matrix, O_bell_prime), workers=-1)
    optimal_params = result.x
    bell_value = - result.fun

    return bell_value, optimal_params
