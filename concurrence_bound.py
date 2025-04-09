from tabnanny import check
import numpy as np

def purity(density_matrix):
    """
    Calculate the purity of a density matrix.
    """
    return np.trace(density_matrix @ density_matrix)

def partial_trace(density_matrix, subsystem):
    """
    Calculate the partial trace of a density matrix over a subsystem.
    """
    rho_sub = np.zeros((3, 3), dtype=np.complex128)
    if subsystem == 1:
        for i in range(3):
            for j in range(3):
                rho_sub[i, j] = np.sum(density_matrix[k * 3 + i, k * 3 + j] for k in range(3))
        return rho_sub
    elif subsystem == 2:
        for i in range(3):
            for j in range(3):
                rho_sub[i, j] = np.sum(density_matrix[i * 3 + k, j * 3 + k] for k in range(3))
        return rho_sub
    else:
        raise ValueError("subsystem must be 1 or 2")

def check_density_matrix(rho):
    # Check if Hermitian: rho == rho^dagger
    is_hermitian = np.allclose(rho, np.conjugate(rho.T))
    print("Is Hermitian:", is_hermitian)
    # Check if normalized: Tr(rho) == 1
    trace = np.trace(rho)
    print("Trace:", trace)
    is_normalized = np.isclose(trace, 1)
    print("Is normalized (Trace = 1):", is_normalized)
    # Check if positive semi-definite: all eigenvalues >= 0
    eigenvalues = np.linalg.eigvalsh(rho) 
    is_positive_semi_definite = np.all(eigenvalues >= -1e-10)  # Allow small numerical tolerance
    print("Eigenvalues:", eigenvalues)
    print("Is positive semi-definite:", is_positive_semi_definite)
    
    return is_hermitian and is_positive_semi_definite and is_normalized

def concurrence_lower(density_matrix, uncertainty_matrix_real, uncertainty_matrix_imag):
    """
    Calculate the lower bound of the concurrence of a bipartite qutrit state.
    """
    rho_A = partial_trace(density_matrix, 1)
    # rho_A_var = partial_trace(uncertainty_matrix_real**2, 1) + partial_trace(uncertainty_matrix_imag**2, 1)
    rho_B = partial_trace(density_matrix, 2)
    # rho_B_var = partial_trace(uncertainty_matrix_real**2, 2) + partial_trace(uncertainty_matrix_imag**2, 2)
    purity_A = np.real(purity(rho_A))
    # purity_A_var = 4 * sum(np.abs(rho_A[i,j])**2 * rho_A_var[i,j] for i in range(3) for j in range(3))
    purity_B = np.real(purity(rho_B))
    # purity_B_var = 4 * sum(np.abs(rho_B[i,j])**2 * rho_B_var[i,j] for i in range(3) for j in range(3))
    total_purity = np.real(purity(density_matrix))
    #mtotal_purity_var = 4 * sum(np.abs(density_matrix[i,j])**2 * (uncertainty_matrix_real[i,j]**2 + uncertainty_matrix_imag[i,j]**2) for i in range(9) for j in range(9))
    conc_lb = 2 * max(0, total_purity - purity_A, total_purity - purity_B)
    if (conc_lb == 0):
        return 0 #,0
    elif (conc_lb == 2 * (total_purity - purity_A)):
        return conc_lb #, np.sqrt(total_purity_var + purity_A_var)
    elif (conc_lb == 2 * (total_purity - purity_B)):
        return conc_lb #, np.sqrt(total_purity_var + purity_B_var)


def concurrence_upper(density_matrix):
    """
    Calculate the upper bound of the concurrence of a bipartite qutrit state.
    """
    rho_A = partial_trace(density_matrix, 1)
    rho_B = partial_trace(density_matrix, 2)
    purity_A = np.real(purity(rho_A))
    purity_B = np.real(purity(rho_B))
    return 2 * min(1 - purity_A, 1 - purity_B)