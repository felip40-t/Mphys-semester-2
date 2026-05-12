import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
    Maximize the Bell inequality over U(3)×U(3) using parallel multistart L-BFGS-B.

    The objective is smooth (trig functions), so gradient-based local search
    from multiple random starts is faster than differential evolution. Starts
    are run in parallel via ThreadPoolExecutor (numpy releases the GIL during
    array operations, giving real concurrency without pickling overhead).

    seed: used to seed the RNG for the random starting points.
    """
    rho4d = density_matrix.reshape(3, 3, 3, 3)
    O4d = O_bell_prime.reshape(3, 3, 3, 3)

    # Derive the optimal contraction order once; reuse it on every function call
    # instead of re-searching on each evaluation (which optimize='optimal' would do).
    _dummy = np.ones((3, 3), dtype=complex)
    path, _ = np.einsum_path(
        'abcd,ic,jd,ijkl,ka,lb->', rho4d, _dummy, _dummy, O4d, _dummy, _dummy,
        optimize='optimal',
    )

    def objective(parameters):
        U = euler_unitary_matrix(*parameters[:6])
        V = euler_unitary_matrix(*parameters[6:])
        val = np.einsum('abcd,ic,jd,ijkl,ka,lb->', rho4d, U.conj(), V.conj(), O4d, U, V, optimize=path)
        return -np.real(val)

    def run_one(x0):
        result = minimize(objective, x0, method='L-BFGS-B', bounds=_BOUNDS)
        return result.x, -result.fun

    rng = np.random.default_rng(seed)
    starts = rng.uniform(0, 2 * np.pi, size=(_N_STARTS, 12))

    n_workers = min(_N_STARTS, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(run_one, starts))

    best_params, best_val = max(results, key=lambda r: r[1])
    return best_val, best_params
