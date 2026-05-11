import numpy as np
from scipy.optimize import minimize
from diboson.physics.unitary_matrix import euler_unitary_matrix

_N_STARTS = 40
_BOUNDS = [(0, 2 * np.pi)] * 12


def optimal_bell_operator(O_bell_prime, parameters):
    """Calculate the optimal Bell operator using the given parameters."""
    U = euler_unitary_matrix(*parameters[:6])
    V = euler_unitary_matrix(*parameters[6:])
    U_cross_V = np.kron(U, V)
    return U_cross_V.conj().T @ O_bell_prime @ U_cross_V


def _objective(parameters, rho4d, O4d):
    U = euler_unitary_matrix(*parameters[:6])
    V = euler_unitary_matrix(*parameters[6:])
    # Tr(rho @ (U⊗V)† O' (U⊗V)) via tensor contraction — avoids forming the 9×9 kron product
    val = np.einsum('abcd,ic,jd,ijkl,ka,lb->', rho4d, U.conj(), V.conj(), O4d, U, V, optimize='optimal')
    return -np.real(val)


def bell_inequality_optimization(density_matrix, O_bell_prime, seed=0):
    """
    Maximize the Bell inequality over U(3)×U(3) using multistart L-BFGS-B.

    The objective is smooth (trig functions), so gradient-based local search
    from multiple random starts is faster than differential evolution.

    seed: used to seed the RNG for the random starting points.
    """
    rho4d = density_matrix.reshape(3, 3, 3, 3)
    O4d = O_bell_prime.reshape(3, 3, 3, 3)

    rng = np.random.default_rng(seed)
    starts = rng.uniform(0, 2 * np.pi, size=(_N_STARTS, 12))

    best_val = -np.inf
    best_params = starts[0]

    for x0 in starts:
        result = minimize(
            _objective, x0,
            args=(rho4d, O4d),
            method='L-BFGS-B',
            bounds=_BOUNDS,
        )
        if -result.fun > best_val:
            best_val = -result.fun
            best_params = result.x

    return best_val, best_params
