from tabnanny import check
import numpy as np

def purity(density_matrix):
    """
    Calculate the purity of a density matrix.
    """
    return np.trace(density_matrix.T @ density_matrix)

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

    # Check if positive semi-definite: all eigenvalues >= 0
    eigenvalues = np.linalg.eigvalsh(rho)  # More stable for Hermitian matrices
    is_positive_semi_definite = np.all(eigenvalues >= -1e-10)  # Allow small numerical tolerance
    print("Eigenvalues:", eigenvalues)
    print("Is positive semi-definite:", is_positive_semi_definite)

    # Check if normalized: Tr(rho) == 1
    trace = np.trace(rho)
    print("Trace:", trace)
    is_normalized = np.isclose(trace, 1)
    print("Is normalized (Trace = 1):", is_normalized)
    
    return is_hermitian and is_positive_semi_definite and is_normalized

def concurrence_lower(density_matrix):
    """
    Calculate the lower bound of the concurrence of a bipartite qutrit state.
    """
    rho_A = partial_trace(density_matrix, 1)
    rho_B = partial_trace(density_matrix, 2)
    purity_A = purity(rho_A)
    purity_B = purity(rho_B)
    total_purity = purity(density_matrix)
    if check_density_matrix(rho_A) and check_density_matrix(rho_B) and check_density_matrix(density_matrix):
        print("All matrices are valid.")
    else:
        print("Invalid matrices.")
    return 2 * max(0, total_purity - purity_A, total_purity - purity_B)