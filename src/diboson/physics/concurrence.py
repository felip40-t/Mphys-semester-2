import logging
import numpy as np

_log = logging.getLogger(__name__)


def purity(density_matrix):
    """
    Calculate the purity of a density matrix.
    """
    return np.trace(density_matrix @ density_matrix)


def partial_trace(density_matrix, subsystem, dim):
    """
    Calculate the partial trace of a density matrix over a subsystem.

    rho is indexed as rho[a*dim+b, c*dim+d] where (a,c) belong to subsystem 1
    and (b,d) to subsystem 2.  Reshape to rho4[a,b,c,d] and contract accordingly.
    """
    rho4 = density_matrix.reshape(dim, dim, dim, dim)
    if subsystem == 1:
        return np.einsum('kikj->ij', rho4)
    elif subsystem == 2:
        return np.einsum('ikjk->ij', rho4)
    else:
        raise ValueError("subsystem must be 1 or 2")


def check_density_matrix(rho):
    is_hermitian = np.allclose(rho, np.conjugate(rho.T))
    trace = np.trace(rho)
    is_normalized = np.isclose(trace, 1)
    eigenvalues = np.linalg.eigvalsh(rho)
    is_positive_semi_definite = np.all(eigenvalues >= -1e-10)
    _log.debug("Is Hermitian: %s", is_hermitian)
    _log.debug("Trace: %s", trace)
    _log.debug("Is normalized (Trace = 1): %s", is_normalized)
    _log.debug("Eigenvalues: %s", eigenvalues)
    _log.debug("Is positive semi-definite: %s", is_positive_semi_definite)
    return is_hermitian and is_positive_semi_definite and is_normalized

def concurrence_lower(density_matrix):
    """
    Calculate the lower bound of the concurrence of a bipartite qutrit state.
    """
    rho_A = partial_trace(density_matrix, 1, 3)
    rho_B = partial_trace(density_matrix, 2, 3)
    purity_A = np.real(purity(rho_A))
    purity_B = np.real(purity(rho_B))
    total_purity = np.real(purity(density_matrix))
    conc_lb = 2 * max(0, total_purity - purity_A, total_purity - purity_B)
    if (conc_lb == 0):
        return 0
    else:
        return np.sqrt(conc_lb)



